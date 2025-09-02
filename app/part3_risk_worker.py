# part3_risk_worker.py
# ------------------------------------------------------------
# 24×7 Risk worker for a crypto symbol on Alpaca.
#
# What it does each loop:
#   • Snapshot account equity and position
#   • Fetch last VAR_WL minutes of 1‑min bars to estimate 1‑day VaR
#   • Compute exposure and rolling drawdown
#   • Persist snapshots to DB tables used by the dashboard
#   • If limits are breached, optionally cancel & flatten
#
# Environment variables (see .env)
# ------------------------------------------------------------

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.types import RawData

from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

# DB
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# --------------------------- helpers ---------------------------

def _to_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    v = str(v).strip().lower()
    return v in ("1", "true", "t", "yes", "y", "on")

def _to_int(v: Optional[str], default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default

def _to_float(v: Optional[str], default: float) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def normalize_symbols(symbol_env: str) -> Tuple[str, str]:
    """Return (slashed, unslashed) variants; accept either input format."""
    s = symbol_env.strip().upper()
    if "/" in s:
        slashed = s
        unslashed = s.replace("/", "")
    else:
        unslashed = s
        # best effort for common crypto pairs like BTCUSD -> BTC/USD
        if len(s) >= 6:
            slashed = s[:-3] + "/" + s[-3:]
        else:
            slashed = s  # fallback
    return slashed, unslashed

def round_qty(x: float, dec: int) -> float:
    q = 10 ** dec
    return math.trunc(x * q) / q

# --------------------------- config ---------------------------

@dataclass
class Cfg:
    symbol_slashed: str
    symbol_unslashed: str
    price_dec: int
    qty_dec: int
    tif: str
    cost_rate_bps: float
    tzname: str

@dataclass
class Limits:
    var_wl_min: int
    var_conf: float
    max_var_pct: float
    max_exposure_usd: float
    max_drawdown_pct: float
    enforce: bool
    cancel_before_flatten: bool
    poll_sec: int

def load_cfg() -> Tuple[Cfg, Limits]:
    slashed, unslashed = normalize_symbols(os.getenv("SYMBOL", "BTC/USD"))
    cfg = Cfg(
        symbol_slashed=slashed,
        symbol_unslashed=unslashed,
        price_dec=_to_int(os.getenv("PRICE_DEC"), 2),
        qty_dec=_to_int(os.getenv("QTY_DEC"), 6),
        tif=os.getenv("DEFAULT_TIF", "GTC"),
        cost_rate_bps=_to_float(os.getenv("COST_RATE_BPS"), 0.0),
        tzname=os.getenv("TZ", "America/New_York"),
    )
    lim = Limits(
        var_wl_min=_to_int(os.getenv("VAR_WL"), 1440),
        var_conf=_to_float(os.getenv("VAR_CONF"), 0.95),
        max_var_pct=_to_float(os.getenv("MAX_VAR_PCT"), 0.20),
        max_exposure_usd=_to_float(os.getenv("MAX_EXPOSURE_USD"), 0.0),
        max_drawdown_pct=_to_float(os.getenv("MAX_DRAWDOWN_PCT"), 0.30),
        enforce=_to_bool(os.getenv("ENFORCE"), False),
        cancel_before_flatten=_to_bool(os.getenv("CANCEL_OPEN_BEFORE_FLATTEN"), True),
        poll_sec=_to_int(os.getenv("VAR_REFRESH_SEC"), 60),
    )
    return cfg, lim

# --------------------------- DB ---------------------------

def build_dsn() -> Optional[str]:
    dsn = os.getenv("DB_DSN")
    if dsn:
        return dsn
    host = os.getenv("DB_HOST")
    if not host:
        return None
    port = os.getenv("DB_PORT", "5432")
    db = os.getenv("DB_NAME", "trading")
    user = os.getenv("DB_USER", "postgres")
    pw = os.getenv("DB_PASSWORD", "postgres")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"

DDL = """
CREATE SCHEMA IF NOT EXISTS public;

CREATE TABLE IF NOT EXISTS public.pnl_equity (
  ts      TIMESTAMPTZ NOT NULL,
  equity  NUMERIC(38,10) NOT NULL
);

CREATE TABLE IF NOT EXISTS public.positions (
  ts             TIMESTAMPTZ NOT NULL,
  symbol         TEXT NOT NULL,
  qty            NUMERIC(38,10) NOT NULL,
  avg_price      NUMERIC(38,10),
  market_price   NUMERIC(38,10),
  market_value   NUMERIC(38,10),
  unrealized_pl  NUMERIC(38,10)
);

CREATE TABLE IF NOT EXISTS public.risk_snapshots (
  ts              TIMESTAMPTZ NOT NULL,
  symbol          TEXT NOT NULL,
  exposure_usd    NUMERIC(38,10),
  var_usd         NUMERIC(38,10),
  var_pct_equity  NUMERIC(38,10),
  drawdown_pct    NUMERIC(38,10)
);

CREATE TABLE IF NOT EXISTS public.risk_events (
  ts         TIMESTAMPTZ NOT NULL,
  symbol     TEXT NOT NULL,
  rule       TEXT NOT NULL,
  metric     NUMERIC(38,10),
  threshold  NUMERIC(38,10),
  message    TEXT
);
"""

def get_engine() -> Optional[Engine]:
    dsn = build_dsn()
    if not dsn:
        return None
    try:
        eng = create_engine(dsn, pool_pre_ping=True, pool_size=3, max_overflow=2)
        with eng.begin() as conn:
            conn.execute(text("SELECT 1"))
            conn.execute(text(DDL))
        return eng
    except SQLAlchemyError:
        return None

def insert_equity(engine: Engine, ts: datetime, equity: float) -> None:
    try:
        with engine.begin() as c:
            c.execute(text("INSERT INTO public.pnl_equity(ts,equity) VALUES (:ts,:eq)"),
                      {"ts": ts, "eq": equity})
    except SQLAlchemyError:
        pass

def insert_position(engine: Engine, ts: datetime, symbol: str, qty: float,
                    avg_price: Optional[float], mpx: Optional[float]) -> None:
    try:
        mv = (qty * (mpx or 0.0))
        upl = (0.0 if (avg_price is None or mpx is None) else (mpx - avg_price) * qty)
        with engine.begin() as c:
            c.execute(text("""
                INSERT INTO public.positions(ts,symbol,qty,avg_price,market_price,market_value,unrealized_pl)
                VALUES (:ts,:sym,:qty,:ap,:mpx,:mv,:upl)
            """), {"ts": ts, "sym": symbol, "qty": qty, "ap": avg_price, "mpx": mpx, "mv": mv, "upl": upl})
    except SQLAlchemyError:
        pass

def insert_risk_snapshot(engine: Engine, ts: datetime, symbol: str, exposure: float,
                         var_usd: Optional[float], var_pct: Optional[float],
                         dd_pct: Optional[float]) -> None:
    try:
        with engine.begin() as c:
            c.execute(text("""
                INSERT INTO public.risk_snapshots(ts,symbol,exposure_usd,var_usd,var_pct_equity,drawdown_pct)
                VALUES (:ts,:sym,:exp,:varu,:varp,:dd)
            """), {"ts": ts, "sym": symbol, "exp": exposure, "varu": var_usd, "varp": var_pct, "dd": dd_pct})
    except SQLAlchemyError:
        pass

def insert_event(engine: Engine, ts: datetime, symbol: str, rule: str,
                 metric: float, threshold: float, message: str) -> None:
    try:
        with engine.begin() as c:
            c.execute(text("""
                INSERT INTO public.risk_events(ts,symbol,rule,metric,threshold,message)
                VALUES (:ts,:sym,:rule,:m,:thr,:msg)
            """), {"ts": ts, "sym": symbol, "rule": rule, "m": metric, "thr": threshold, "msg": message})
    except SQLAlchemyError:
        pass

def load_equity_history(engine: Engine, days: int = 7) -> pd.Series:
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with engine.begin() as c:
            res = c.execute(text("""
                SELECT ts, equity
                FROM public.pnl_equity
                WHERE ts >= :cutoff
                ORDER BY ts
            """), {"cutoff": cutoff}).fetchall()
        if not res:
            return pd.Series(dtype=float)
        idx = pd.to_datetime([r[0] for r in res], utc=True)
        vals = [float(r[1]) for r in res]
        return pd.Series(vals, index=idx)
    except SQLAlchemyError:
        return pd.Series(dtype=float)

# --------------------------- data & trading ---------------------------

def make_trading_client() -> TradingClient:
    # Accept multiple env spellings (ALPACA_* preferred)
    key = (
        os.getenv("ALPACA_API_KEY_ID")
        or os.getenv("ALPACA_API_KEY")
        or os.getenv("APCA_API_KEY_ID")
        or ""
    )
    secret = (
        os.getenv("ALPACA_API_SECRET_KEY")
        or os.getenv("ALPACA_API_SECRET")
        or os.getenv("APCA_API_SECRET_KEY")
        or ""
    )
    paper = _to_bool(os.getenv("ALPACA_PAPER"), True)
    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca API keys. Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY "
            "(or APCA_API_KEY_ID / APCA_API_SECRET_KEY)."
        )
    print(f"[auth] Using {'paper' if paper else 'live'} endpoint; key starts with {key[:4]}…", flush=True)
    return TradingClient(key, secret, paper=paper)

def make_crypto_data_client() -> CryptoHistoricalDataClient:
    # For crypto, keys are optional for public data; pass none to use IP based.
    return CryptoHistoricalDataClient()

def get_account_and_position(tcli: TradingClient, symbol_unslashed: str) -> Tuple[float, float, Optional[float]]:
    """Return (equity, qty, avg_price). Uses unslashed symbol to match Trading API positions."""
    acct: RawData = tcli.get_account()  # type: ignore
    equity = float(acct.equity)  # type: ignore
    qty = 0.0
    avg_price = None

    try:
        positions: List[RawData] = tcli.get_all_positions()  # type: ignore
        for p in positions:
            psym = str(p.symbol).upper()
            if psym == symbol_unslashed:
                qty = float(p.qty)  # signed in alpaca-py>=0.21 (for short/equity), crypto long-only typically +
                try:
                    avg_price = float(p.avg_entry_price)
                except Exception:
                    avg_price = None
                break
    except Exception:
        pass
    return equity, qty, avg_price

def fetch_latest_price(dcli: CryptoHistoricalDataClient, symbol_slashed: str) -> Optional[float]:
    try:
        # Latest trade is robust/cheap; bars are next.
        req = CryptoLatestTradeRequest(symbol_or_symbols=symbol_slashed)
        out = dcli.get_crypto_latest_trade(req)
        px = None
        if hasattr(out, "symbol") and hasattr(out, "price"):
            px = float(out.price)  # type: ignore
        else:
            # mapping-like
            for _sym, tr in out.items():  # type: ignore
                px = float(tr.price)      # type: ignore
                break
        return px
    except Exception:
        return None

def fetch_minute_bars(dcli: CryptoHistoricalDataClient, symbol_slashed: str, minutes: int) -> pd.Series:
    """Return a Series of closes indexed by UTC timestamps for the last `minutes` minutes."""
    end = utcnow()
    start = end - timedelta(minutes=minutes + 5)  # small buffer
    try:
        req = CryptoBarsRequest(
            symbol_or_symbols=[symbol_slashed],
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            limit=minutes + 10,
        )
        bars = dcli.get_crypto_bars(req).df  # MultiIndex (symbol, ts)
        if bars is None or len(bars) == 0:
            return pd.Series(dtype=float)
        # Select this symbol, take 'close'
        if isinstance(bars.index, pd.MultiIndex):
            df = bars.xs(symbol_slashed, level=0)
        else:
            df = bars
        closes = df["close"].astype(float)
        closes.index = pd.to_datetime(closes.index, utc=True)
        closes = closes.sort_index()
        # De-dup & last N
        closes = closes[~closes.index.duplicated(keep="last")].tail(minutes)
        return closes
    except Exception:
        return pd.Series(dtype=float)

# --------------------------- risk math ---------------------------

def compute_daily_var(closes: pd.Series, conf: float, exposure_usd: float) -> Optional[float]:
    """Parametric 1‑day VaR using minute returns scaled by sqrt(1440). One‑sided."""
    if closes is None or len(closes) < 30 or exposure_usd <= 0:
        return None
    rets = closes.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 30:
        return None
    std_min = float(np.nanstd(rets, ddof=1))
    if not np.isfinite(std_min) or std_min <= 0:
        return None
    std_day = std_min * math.sqrt(1440.0)
    # Convert one‑sided confidence (e.g., 0.95) to z via inverse CDF of N(0,1).
    from statistics import NormalDist
    z = NormalDist().inv_cdf(conf)   # replaces the erfinv machinery
    var_pct = abs(z) * std_day
    var_usd = var_pct * exposure_usd
    return float(var_usd)

def compute_drawdown_pct(equity_series: pd.Series, current_equity: float) -> Optional[float]:
    if current_equity is None or current_equity <= 0:
        return None
    try:
        ser = equity_series.copy() if equity_series is not None else pd.Series(dtype=float)
        if ser is None or len(ser) == 0:
            base = pd.Series([current_equity], index=[utcnow()])
        else:
            base = pd.concat([ser, pd.Series([current_equity], index=[utcnow()])]).sort_index()
        base = base.replace([np.inf, -np.inf], np.nan).dropna()
        if len(base) == 0:
            return None
        roll_max = base.cummax()
        dd = (base - roll_max) / roll_max
        return float(dd.iloc[-1])
    except Exception:
        return None

# --------------------------- actions ---------------------------

def cancel_all_open_orders(tcli: TradingClient) -> None:
    try:
        tcli.cancel_orders()
    except Exception:
        pass

def flatten_position(tcli: TradingClient, symbol_unslashed: str, qty: float, qty_dec: int) -> Optional[str]:
    """Submit a MARKET order to flat the given quantity (signed qty -> side). Returns order id if any."""
    try:
        q = round_qty(abs(qty), qty_dec)
        if q <= 0:
            return None
        side = OrderSide.SELL if qty > 0 else OrderSide.BUY
        req = MarketOrderRequest(symbol=symbol_unslashed, qty=q, side=side, time_in_force=TimeInForce.GTC)
        order = tcli.submit_order(order_data=req)
        oid = str(order.id) if hasattr(order, "id") else None
        return oid
    except Exception:
        return None

# --------------------------- main loop ---------------------------

def main():
    cfg, lim = load_cfg()
    print(f"[risk_worker] symbol={cfg.symbol_slashed} (trade {cfg.symbol_unslashed}) conf={lim.var_conf} wl={lim.var_wl_min}m"
          f" max_var_pct={lim.max_var_pct:.2%} max_dd={lim.max_drawdown_pct:.2%} enforce={lim.enforce}")

    tcli = make_trading_client()
    dcli = make_crypto_data_client()
    eng = get_engine()

    while True:
        loop_t0 = time.time()
        now = utcnow()

        # 1) Account & position
        equity, qty, avg_price = get_account_and_position(tcli, cfg.symbol_unslashed)

        # 2) Market price & exposure
        mpx = fetch_latest_price(dcli, cfg.symbol_slashed)
        if mpx is None:
            # Fallback: last close from bars
            closes = fetch_minute_bars(dcli, cfg.symbol_slashed, max(15, min(lim.var_wl_min, 120)))
            if len(closes) > 0:
                mpx = float(closes.iloc[-1])
        exposure = abs(qty) * (mpx or 0.0)

        # 3) VaR from 1‑min bars
        closes = fetch_minute_bars(dcli, cfg.symbol_slashed, lim.var_wl_min)
        var_usd = compute_daily_var(closes, lim.var_conf, exposure) if exposure > 0 else 0.0
        var_pct = (var_usd / equity) if (equity and equity > 0 and var_usd is not None) else None

        # 4) Drawdown (uses last 7 days of equity from DB if available)
        dd_ratio = None
        if eng is not None:
            hist = load_equity_history(eng, days=7)
            dd_ratio = compute_drawdown_pct(hist, equity)

        # 5) Persist snapshots
        if eng is not None:
            insert_equity(eng, now, equity)
            insert_position(eng, now, cfg.symbol_unslashed, qty, avg_price, mpx)
            insert_risk_snapshot(eng, now, cfg.symbol_unslashed, exposure, var_usd if var_usd is not None else None,
                                 var_pct if var_pct is not None else None,
                                 dd_ratio if dd_ratio is not None else None)

        # 6) Evaluate limits
        breaches = []  # list[(rule, metric, threshold, message)]
        if var_pct is not None and var_pct > lim.max_var_pct:
            breaches.append(("VAR_PCT", float(var_pct), float(lim.max_var_pct),
                             f"VaR {var_pct:.2%} > limit {lim.max_var_pct:.2%}"))
        if lim.max_exposure_usd > 0 and exposure > lim.max_exposure_usd:
            breaches.append(("EXPOSURE_USD", float(exposure), float(lim.max_exposure_usd),
                             f"Exposure ${exposure:,.2f} > cap ${lim.max_exposure_usd:,.2f}"))
        if dd_ratio is not None and (-dd_ratio) > lim.max_drawdown_pct:  # dd_ratio is negative or zero
            breaches.append(("DRAWDOWN_PCT", float(-dd_ratio), float(lim.max_drawdown_pct),
                             f"Drawdown {(-dd_ratio):.2%} > limit {lim.max_drawdown_pct:.2%}"))

        if breaches:
            msg = " | ".join(b[3] for b in breaches)
            print(f"[RISK][BREACH] {msg}")
            if eng is not None:
                for rule, metric, thr, m in breaches:
                    insert_event(eng, now, cfg.symbol_unslashed, rule, metric, thr, m)

            if lim.enforce and abs(qty) > 0:
                if lim.cancel_before_flatten:
                    cancel_all_open_orders(tcli)
                oid = flatten_position(tcli, cfg.symbol_unslashed, qty, cfg.qty_dec)
                if oid:
                    if eng is not None:
                        insert_event(eng, utcnow(), cfg.symbol_unslashed, "FLATTEN", 0.0, 0.0,
                                     f"Flattened due to {', '.join(b[0] for b in breaches)} (order_id={oid})")
                    print(f"[RISK][ACTION] Flattened position (order_id={oid})")
        else:
            vf = f"{var_usd:,.2f}" if var_usd is not None else "—"
            vp = f"{var_pct:.2%}"  if var_pct is not None else "—"
            dd = f"{dd_ratio:.2%}" if dd_ratio is not None else "—"
            print(f"[ok] eq=${equity:,.2f} pos={qty:.{cfg.qty_dec}f} @ {mpx if mpx is not None else '—'} "
                  f"exp=${exposure:,.2f} VaR={vf} ({vp}) DD={dd}")

        # 7) Sleep remaining time
        elapsed = time.time() - loop_t0
        time.sleep(max(0.5, lim.poll_sec - elapsed))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\nbye 👋")

# part2_pseudo_live.py
# ------------------------------------------------------------
# Pseudo-live crypto trader for Alpaca using alpaca-py.
# - 24x7 (no market-hours logic)
# - EMA crossover with ADX filter (optional price-action confirmation)
# - Z-score mean reversion with **ADX + RSI** filters (long-only)
# - Long-only sizing based on equity, with caps
# - Places GTC limit orders with a small price offset
# - Avoids stacking duplicates; cancels stale open orders
#
# In-sync with part1_backtest.py for:
#   • indicator math (EMA / ADX / Z-score)
#   • symbol normalization (BTCUSD ↔ BTC/USD)
#   • rounding helpers and timeframe mapping
#
# ENV (typical):
#   SYMBOL="BTC/USD"         # or "BTCUSD" — both accepted
#   PRICE_DEC="2"            # price decimals
#   QTY_DEC="6"              # qty decimals
#   DEFAULT_TIF="GTC"
#   LIMIT_AWAY_PCT="0.001"   # 0.10% from last price
#   POLL_SEC="5"
#   BAR_LOOKBACK_MIN="200"   # minutes of history to compute indicators
#
#   STRATEGY="ema"           # or "zscore"
#   EMA_FAST="13" EMA_SLOW="150" ADX_LEN="7" ADX_MIN="26"
#   USE_PA_FILTER="1" PA_MIN_BODY_FRAC="0.5"
#   Z_N="288" Z_ENTRY="2.4" Z_EXIT="0.0" Z_STOP="3.5" ADX_SKIP="30.0" Z_ADX_LEN="14"
#   RSI_LEN="14" RSI_LOWER="30" RSI_UPPER="70"
#
#   MAX_LEVERAGE="1.0"
#   USD_NOTIONAL_CAP="500000"
#   MAX_QTY_PER_TRADE="99999999"  # per-order qty cap
#   MIN_NOTIONAL_USD="10"     # skip orders smaller than this notional
#
#   ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY
#   ALPACA_PAPER="0"         # use paper (default 1). Set 0 for live.
#
#   CANCEL_AFTER_SEC="300"    # cancel open orders older than this
#   DRY_RUN="0"              # set 1 to print but not send orders
#
# Requires: alpaca-py >= 0.21
# ------------------------------------------------------------

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from collections import defaultdict, deque

import math
import numpy as np
import pandas as pd

# Alpaca trading
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderStatus

# Alpaca data (crypto)
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # modern import


# --------------------------- Utils ---------------------------

def env_str(k: str, default: str) -> str:
    v = os.getenv(k, default)
    return v if isinstance(v, str) else default

def env_int(k: str, default: int) -> int:
    try:
        return int(str(os.getenv(k, default)))
    except Exception:
        return default

def env_float(k: str, default: float) -> float:
    try:
        return float(str(os.getenv(k, default)))
    except Exception:
        return default

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def get_timeframe(minutes: int):
    return TimeFrame(minutes, TimeFrameUnit.Minute)

def symbol_for_data(sym: str) -> str:
    """BTCUSD -> BTC/USD ; pass through if already slashed."""
    if "/" in sym:
        return sym
    s = sym.upper().strip()
    if s.endswith(("USDT", "USDC")) and len(s) > 4:
        return s[:-4] + "/" + s[-4:]
    if s.endswith("USD") and len(s) > 3:
        return s[:-3] + "/USD"
    return s

def symbol_for_trade(sym: str) -> str:
    """BTC/USD -> BTCUSD (for Trading API)."""
    return sym.replace("/", "").upper().strip()

def round_price(x: float, dec: int) -> float:
    return float(np.round(float(x), dec))

def round_qty(x: float, dec: int) -> float:
    r = float(np.round(float(x), dec))
    return max(0.0, r)

def floor_qty(x: float, dec: int) -> float:
    step = 10 ** (-dec)
    # floor to the allowed precision, keep non-negative
    return max(0.0, math.floor((float(x) + 1e-12) / step) * step)

def list_open_orders(tc: TradingClient, symbol: str):
    """
    Return orders for `symbol` that still reserve qty/funds.
    Fetch ALL; filter by symbol + active-ish broker states.
    """
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, nested=True)
        orders = list(tc.get_orders(req))
    except Exception:
        return []
    target = symbol_for_trade(symbol)  # 'BTCUSD'
    out = []
    for o in orders:
        st  = str(getattr(o, "status", "")).lower()
        sym = str(getattr(o, "symbol", "")).upper().replace("/", "")
        if sym == target and st in ACTIVE_ORDER_STATUSES:
            out.append(o)
    return out

def get_position_free_qty(tc: TradingClient, symbol: str, qty_dec: int) -> float:
    """
    Broker-reported FREE qty (qty_available). Falls back to total qty if not present.
    """
    try:
        target = symbol_for_trade(symbol)
        for p in (tc.get_all_positions() or []):
            sym = str(getattr(p, "symbol", "")).upper().replace("/", "")
            if sym == target:
                avail = getattr(p, "qty_available", None)
                if avail is not None:
                    return floor_qty(float(avail), qty_dec)
                return floor_qty(float(getattr(p, "qty", 0.0)), qty_dec)
    except Exception:
        pass
    return 0.0

def open_sell_qty(tc: TradingClient, symbol: str) -> float:
    """Total *remaining* qty on open SELL orders for this symbol."""
    tot = 0.0
    for o in list_open_orders(tc, symbol):
        try:
            if str(getattr(o, "side", "")).lower() == "sell":
                q  = float(getattr(o, "qty", 0.0) or 0.0)
                fq = float(getattr(o, "filled_qty", 0.0) or 0.0)
                rem = max(0.0, q - fq)
                tot += rem
        except Exception:
            pass
    return tot

def open_buy_notional(tc: TradingClient, symbol: str) -> float:
    """Total *remaining* NOTIONAL on open BUY orders for this symbol."""
    tot = 0.0
    for o in list_open_orders(tc, symbol):
        try:
            if str(getattr(o, "side", "")).lower() != "buy":
                continue
            q  = float(getattr(o, "qty", 0.0) or 0.0)
            fq = float(getattr(o, "filled_qty", 0.0) or 0.0)
            rem = max(0.0, q - fq)
            px  = float(getattr(o, "limit_price", 0.0) or 0.0) or float(getattr(o, "avg_fill_price", 0.0) or 0.0)
            tot += rem * px
        except Exception:
            pass
    return tot

def get_conservative_available_usd(acct) -> float:
    """
    Prefer the tightest cap for non-marginable assets (e.g., crypto).
    Fallback sanely if fields are missing.
    """
    candidates = []
    for f in ("non_marginable_buying_power", "cash_withdrawable"):
        v = getattr(acct, f, None)
        if v is not None:
            try:
                x = float(v)
                if x > 0:
                    candidates.append(x)
            except Exception:
                pass
    for f in ("buying_power", "effective_buying_power", "cash"):
        v = getattr(acct, f, None)
        if v is not None:
            try:
                x = float(v)
                if x > 0:
                    candidates.append(x)
            except Exception:
                pass
    return min(candidates) if candidates else 0.0

def get_cash_available(tc: TradingClient) -> float:
    acct = tc.get_account()
    # For crypto, cash is the USD you can spend
    try: return float(acct.cash)
    except: return 0.0

def cancel_all_orders_for_symbol(tc: TradingClient, symbol: str) -> int:
    n = 0
    for o in list_open_orders(tc, symbol):
        try:
            tc.cancel_order_by_id(o.id)
            print(f"[cancel] {o.side} {symbol} qty={o.qty} id={o.id} (flip)")
            n += 1
        except Exception as e:
            print(f"[cancel-error] id={o.id} {e}")
    return n

def inflight_side_qty(symbol: str, side_str: str) -> float:
    now = utcnow()
    dq = INFLIGHT[symbol]
    # purge old
    while dq and (now - dq[0]["ts"]).total_seconds() > INFLIGHT_TTL:
        dq.popleft()
    return sum(x["qty"] for x in dq if x["side"] == side_str)

# (duplicate helpers kept for compatibility with earlier file layout)
def env_str(k: str, default: str) -> str:
    import os
    v = os.getenv(k, default)
    return v if isinstance(v, str) else default

def env_int(k: str, default: int) -> int:
    import os
    try:
        return int(str(os.getenv(k, default)))
    except Exception:
        return default

def env_float(k: str, default: float) -> float:
    import os
    try:
        return float(str(os.getenv(k, default)))
    except Exception:
        return default

# -------------------- In-flight tracking ---------------------
INFLIGHT = defaultdict(deque)
INFLIGHT_TTL = env_int("INFLIGHT_TTL_SEC", 60)

# States that still reserve qty/funds (treat as “open enough”)
ACTIVE_ORDER_STATUSES = {
    "new", "partially_filled",
    "accepted", "pending_new", "accepted_for_bidding",
    "pending_replace",
    "held",
}

# Side-level cooldown after broker says “insufficient balance”
SIDE_COOLDOWN = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
INSUFF_COOLDOWN_SEC = env_int("INSUFF_COOLDOWN_SEC", 60)


def cancel_open_orders_for_symbol(trading_client: TradingClient, symbol: str) -> int:
    """
    Cancels all OPEN/working orders for a given symbol.
    Returns count of orders we requested to cancel.
    """
    req = GetOrdersRequest(
        status=OrderStatus.OPEN,      # all working orders (new/accepted/partially_filled but still live)
        symbols=[symbol],
        limit=500
    )
    open_orders = trading_client.get_orders(filter=req)
    cancelled = 0
    for o in open_orders:
        try:
            trading_client.cancel_order_by_id(o.id)
            cancelled += 1
        except Exception as e:
            # log and continue (idempotent, some may already be canceling)
            print(f"[cancel_open_orders_for_symbol] cancel failed {o.id}: {e}")
    return cancelled

# --------------------------- Config --------------------------

@dataclass
class Cfg:
    symbol: str
    price_dec: int
    qty_dec: int
    tif: TimeInForce
    limit_away_pct: float
    poll_sec: int
    bar_lookback_min: int
    strategy: str
    # EMA
    ema_fast: int
    ema_slow: int
    adx_len: int
    adx_min: float
    use_pa_filter: bool
    pa_min_body_frac: float
    # Z-score + filters
    z_n: int
    z_entry: float
    z_exit: float
    z_stop: float
    adx_skip: float
    z_adx_len: int       # NEW: ADX lookback for z-score path
    rsi_len: int         # NEW
    rsi_lower: float     # NEW
    rsi_upper: float     # NEW (reserved for shorts; not used in long-only)
    # Sizing
    max_leverage: float
    usd_notional_cap: float
    max_qty_per_trade: float
    min_notional_usd: float
    cancel_after_sec: int
    usd_cash_reserve: float          # keep a few USD unspent
    base_qty_reserve_ticks: int      # keep a few qty ticks unsold
    cancel_on_flip: bool             # cancel all open orders when signal flips
    # Modes
    dry_run: bool
    paper: bool

def build_cfg() -> Cfg:
    # Normalize symbol for Alpaca crypto ("BTC/USD" style)
    raw_symbol = env_str("SYMBOL", "BTC/USD").upper().strip()
    symbol = symbol_for_data(raw_symbol)
    tif_name = env_str("DEFAULT_TIF", "GTC").upper()
    try:
        tif = TimeInForce[tif_name]
    except Exception:
        tif = TimeInForce.GTC

    return Cfg(
        symbol=symbol,
        price_dec=env_int("PRICE_DEC", 2),
        qty_dec=env_int("QTY_DEC", 6),
        tif=tif,
        limit_away_pct=env_float("LIMIT_AWAY_PCT", 0.001),  # 0.10%
        poll_sec=env_int("POLL_SEC", 5),
        bar_lookback_min=env_int("BAR_LOOKBACK_MIN", 200),
        strategy=env_str("STRATEGY", "ema").lower(),
        # EMA
        ema_fast=env_int("EMA_FAST", 20),
        ema_slow=env_int("EMA_SLOW", 50),
        adx_len=env_int("ADX_LEN", 14),
        adx_min=env_float("ADX_MIN", 25.0),
        use_pa_filter=env_int("USE_PA_FILTER", 1) == 1,
        pa_min_body_frac=env_float("PA_MIN_BODY_FRAC", 0.5),
        # Z-score + gates
        z_n=env_int("Z_N", 50),
        z_entry=env_float("Z_ENTRY", 2.0),
        z_exit=env_float("Z_EXIT", 0.5),
        z_stop=env_float("Z_STOP", 3.0),
        adx_skip=env_float("ADX_SKIP", 30.0),
        z_adx_len=env_int("Z_ADX_LEN", env_int("ADX_LEN", 14)),
        rsi_len=env_int("RSI_LEN", 14),
        rsi_lower=env_float("RSI_LOWER", 30.0),
        rsi_upper=env_float("RSI_UPPER", 70.0),
        # Sizing/risk
        max_leverage=env_float("MAX_LEVERAGE", 1.0),
        usd_notional_cap=env_float("USD_NOTIONAL_CAP", 1000.0),
        max_qty_per_trade=env_float("MAX_QTY_PER_TRADE", 9e9),
        min_notional_usd=env_float("MIN_NOTIONAL_USD", 5.0),
        usd_cash_reserve=env_float("USD_CASH_RESERVE", 5.0),
        cancel_after_sec=env_int("CANCEL_AFTER_SEC", 90),
        base_qty_reserve_ticks=env_int("BASE_QTY_RESERVE_TICKS", 1),
        cancel_on_flip=env_int("CANCEL_ON_FLIP", 1) == 1,
        # Modes
        dry_run=env_int("DRY_RUN", 0) == 1,
        paper=env_int("ALPACA_PAPER", 1) == 1,
    )


# --------------------------- Clients -------------------------

def mk_trading_client(paper: bool) -> TradingClient:
    key = env_str("ALPACA_API_KEY_ID", os.getenv("ALPACA_API_KEY", ""))
    sec = env_str("ALPACA_API_SECRET_KEY", os.getenv("ALPACA_API_SECRET", ""))
    if not key or not sec:
        raise RuntimeError("Missing Alpaca API keys (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY).")
    return TradingClient(api_key=key, secret_key=sec, paper=paper)

def mk_data_client() -> CryptoHistoricalDataClient:
    key = env_str("ALPACA_API_KEY_ID", os.getenv("ALPACA_API_KEY", ""))
    sec = env_str("ALPACA_API_SECRET_KEY", os.getenv("ALPACA_API_SECRET", ""))
    return CryptoHistoricalDataClient(key, sec)


# --------------------------- Data ----------------------------

def fetch_bars(cli: CryptoHistoricalDataClient, symbol: str, n_minutes: int) -> pd.DataFrame:
    end = utcnow()
    start = end - timedelta(minutes=n_minutes + 10)
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol_for_data(symbol)],
        timeframe=get_timeframe(1),
        start=start,
        end=end
    )
    bars = cli.get_crypto_bars(req)
    df = bars.df
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index().set_index("timestamp")
        df = df[df["symbol"] == symbol_for_data(symbol)].drop(columns=["symbol"])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    # normalize names
    ren = {"open":"o","high":"h","low":"l","close":"c","volume":"v"}
    for k, v in ren.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    keep = [c for c in ["o","h","l","c","v"] if c in df.columns]
    df = df[keep].replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["o"]>0)&(df["h"]>0)&(df["l"]>0)&(df["c"]>0)]
    return df

def get_latest_trade_px(cli: CryptoHistoricalDataClient, symbol: str) -> Optional[float]:
    try:
        lt = cli.get_crypto_latest_trade(
            CryptoLatestTradeRequest(symbol_or_symbols=[symbol_for_data(symbol)])
        )
        rec = lt[symbol_for_data(symbol)]
        price = getattr(rec, "price", None)
        return float(price) if price is not None else None
    except Exception:
        return None

def get_latest_trade_with_ts(cli: CryptoHistoricalDataClient, symbol: str) -> Tuple[Optional[float], Optional[datetime]]:
    try:
        lt = cli.get_crypto_latest_trade(
            CryptoLatestTradeRequest(symbol_or_symbols=[symbol_for_data(symbol)])
        )
        rec = lt[symbol_for_data(symbol)]
        price = float(getattr(rec, "price", None)) if getattr(rec, "price", None) is not None else None
        # Alpaca objects expose a timestamp; support both 'timestamp' and 't'
        ts_raw = getattr(rec, "timestamp", None) or getattr(rec, "t", None)
        ts = pd.to_datetime(ts_raw, utc=True).to_pydatetime() if ts_raw is not None else None
        return price, ts
    except Exception:
        return None, None

# ----------------------- Indicators --------------------------

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _wilder_ewm(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (alpha = 1/n)."""
    return s.ewm(alpha=1.0/n, adjust=False).mean()

def compute_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["h"], df["l"], df["c"]
    up = h.diff()
    down = -l.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0.0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0.0)
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = _wilder_ewm(tr, n)
    plus_di = 100.0 * (_wilder_ewm(plus_dm, n) / atr).replace([np.inf, -np.inf], np.nan)
    minus_di = 100.0 * (_wilder_ewm(minus_dm, n) / atr).replace([np.inf, -np.inf], np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = _wilder_ewm(dx, n)
    return adx

def compute_rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Wilder RSI on close price (df['c']).
    """
    c = pd.to_numeric(df["c"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = _wilder_ewm(up, n)
    roll_down = _wilder_ewm(down, n)
    rs = (roll_up / roll_down.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _price_action_ok(df: pd.DataFrame, min_body_frac: float) -> pd.Series:
    body = (df["c"] - df["o"]).clip(lower=0.0)
    rng = (df["h"] - df["l"]).replace(0, np.nan)
    frac = (body / rng).fillna(0.0)
    return (df["c"] > df["o"]) & (frac >= min_body_frac)

def signals_ema_adx(cfg: Cfg, df: pd.DataFrame) -> pd.Series:
    out = df.copy()
    out["ema_fast"] = _ema(out["c"], cfg.ema_fast)
    out["ema_slow"] = _ema(out["c"], cfg.ema_slow)
    out["adx"] = compute_adx(out, cfg.adx_len)
    cross_up = (out["ema_fast"] > out["ema_slow"]) & (out["ema_fast"].shift(1) <= out["ema_slow"].shift(1))
    cross_down = (out["ema_fast"] < out["ema_slow"]) & (out["ema_fast"].shift(1) >= out["ema_slow"].shift(1))
    strong = _price_action_ok(out, cfg.pa_min_body_frac) if cfg.use_pa_filter else pd.Series(True, index=out.index)
    sig = []
    state = 0
    for i in range(len(out)):
        if state == 0:
            if bool(cross_up.iloc[i]) and bool(out["adx"].iloc[i] >= cfg.adx_min) and bool(strong.iloc[i]):
                state = 1
        else:
            if bool(cross_down.iloc[i]):
                state = 0
        sig.append(state)
    return pd.Series(sig, index=out.index).fillna(0).astype(int)

def signals_zscore(cfg: Cfg, df: pd.DataFrame) -> pd.Series:
    """
    Z-score mean reversion (long-only) with ADX + RSI filters:
      Entry: z crosses DOWN below -z_entry AND ADX < adx_skip AND RSI < rsi_lower
      Exit (reversion): z crosses UP above -z_exit AND ADX < adx_skip
      Exit (stop): z crosses DOWN below -z_stop  (unconditional)
    Z computed on mid-price ((H+L)/2) to match backtest.
    """
    out = df.copy()

    # Z on mid-price
    mid = ((pd.to_numeric(out["h"], errors="coerce") + pd.to_numeric(out["l"], errors="coerce")) / 2.0).astype(float)
    ma = mid.rolling(cfg.z_n, min_periods=cfg.z_n).mean()
    sd = mid.rolling(cfg.z_n, min_periods=cfg.z_n).std(ddof=0)
    out["z"] = ((mid - ma) / sd).replace([np.inf, -np.inf], np.nan)

    # Filters
    out["adx"] = compute_adx(out, cfg.z_adx_len)
    out["rsi"] = compute_rsi(out, cfg.rsi_len)

    th_entry = -abs(cfg.z_entry)
    th_exit  = -abs(cfg.z_exit)
    th_stop  = -abs(cfg.z_stop)

    # Gates
    adx_ok = (out["adx"] < cfg.adx_skip) | out["adx"].isna()
    rsi_ok = (out["rsi"] < cfg.rsi_lower) | out["rsi"].isna()

    # Cross helpers
    z_prev = out["z"].shift(1)

    enter = (z_prev > th_entry) & (out["z"] <= th_entry) & adx_ok & rsi_ok
    exit_rev = (z_prev < th_exit) & (out["z"] >= th_exit) & adx_ok
    exit_stop = (z_prev > th_stop) & (out["z"] <= th_stop)

    sig = []
    state = 0
    for i in range(len(out)):
        if state == 0:
            if bool(enter.iloc[i]):
                state = 1
        else:
            if bool(exit_rev.iloc[i]) or bool(exit_stop.iloc[i]):
                state = 0
        sig.append(state)
    return pd.Series(sig, index=out.index).fillna(0).astype(int)

def compute_signal(cfg: Cfg, df: pd.DataFrame) -> Tuple[int, float]:
    """Return (signal, ref_close). signal ∈ {0,1}."""
    if df.empty:
        return 0, float('nan')
    if cfg.strategy == "ema":
        sig_series = signals_ema_adx(cfg, df)
    elif cfg.strategy == "zscore":
        sig_series = signals_zscore(cfg, df)
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")
    return int(sig_series.iloc[-1]), float(df["c"].iloc[-1])

def manage_open_order(tc: TradingClient, cfg: Cfg, side: OrderSide, ref_px: float) -> bool:
    oo_all = list_open_orders(tc, cfg.symbol)
    if not oo_all:
        return False

    side_str = side.name.lower()
    oo = [o for o in oo_all if str(getattr(o, "side", "")).lower() == side_str]
    if not oo:
        return False  # no same-side order; caller may place a new one

    # newest same-side
    def _dt(x):
        try: return pd.to_datetime(getattr(x, "created_at", None), utc=True)
        except: return pd.Timestamp.utcnow()
    o = sorted(oo, key=_dt, reverse=True)[0]

    now = utcnow()
    try:
        ots = pd.to_datetime(o.created_at, utc=True).to_pydatetime()
    except:
        ots = now
    age = (now - ots).total_seconds()
    st  = str(getattr(o, "status", "")).lower()
    lim = float(getattr(o, "limit_price", 0.0))

    # Important: treat early states as present
    if st in {"pending_new","new","accepted","pending_replace","pending_cancel"}:
        print(f"[pending] 1 open {side.name} order age={int(age)}s — waiting (status={st})")
        return True

    MIN_REPRICE_SEC = env_int("MIN_REPRICE_SEC", 45)
    REPRICE_PCT     = env_float("REPRICE_PCT", 0.0005)
    MAX_LIFETIME_SEC= env_int("MAX_LIFETIME_SEC", 900)

    want_limit = desired_limit_for(side, ref_px, cfg)
    drift = pct_diff(lim, want_limit)

    need_reprice = (drift > REPRICE_PCT) and (age >= MIN_REPRICE_SEC) and (st in {"new","partially_filled"})
    if need_reprice:
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            tc.replace_order_by_id(o.id, ReplaceOrderRequest(limit_price=want_limit))
            print(f"[replace] id={o.id} limit {lim} -> {want_limit}")
            return True
        except Exception as e:
            print(f"[replace-fallback] {e} -> cancel+repost")
            try:
                tc.cancel_order_by_id(o.id)
                print(f"[cancel] {o.side} {cfg.symbol} qty={o.qty} id={o.id} age={int(age)}s (for reprice)")
            except Exception as ce:
                if "pending cancel" in str(ce).lower():
                    print(f"[cancel-wait] id={o.id} already pending cancel")
                    return True
                print(f"[cancel-error] id={o.id} {ce}")
                return True
            return False  # let caller place fresh

    if age >= MAX_LIFETIME_SEC:
        try:
            tc.cancel_order_by_id(o.id)
            print(f"[cancel] {o.side} {cfg.symbol} id={o.id} age={int(age)}s (max lifetime)")
        except Exception as e:
            print(f"[cancel-error] id={o.id} {e}")
        return False

    print(f"[pending] 1 open {side.name} order age={int(age)}s — not repricing (|Δ|={drift:.5f})")
    return True

# ------------------------ Trading ops ------------------------

def get_equity(tc: TradingClient) -> float:
    acct = tc.get_account()
    # prefer equity > cash since we might hold positions
    try:
        return float(acct.equity)
    except Exception:
        return float(acct.cash)

def get_symbol_position_qty(tc: TradingClient, symbol: str) -> float:
    try:
        target = symbol_for_trade(symbol)  # e.g. 'BTCUSD'
        poss = tc.get_all_positions() or []
        for p in poss:
            # normalize broker symbol to unslashed, uppercase
            sym = str(getattr(p, "symbol", "")).upper().replace("/", "")
            if sym == target:
                return float(getattr(p, "qty", 0.0))
        return 0.0
    except Exception:
        return 0.0

# Treat these as “active / reserving” for our purposes
ACTIVE_ORDER_STATUSES = {
    "new", "partially_filled",
    "accepted", "pending_new", "accepted_for_bidding",
    "pending_replace",
    # rare but seen in the wild:
    "held"
}

def cancel_stale_orders(tc, symbol, older_than_sec, ref_px, reprice_pct=0.0005):
    n = 0
    now = utcnow()
    want_side = None  # optional: infer from current signal if you like
    for o in list_open_orders(tc, symbol):
        try:
            created = pd.to_datetime(o.created_at, utc=True).to_pydatetime()
            age = (now - created).total_seconds()
            st  = str(getattr(o, "status", "")).lower()
            lim = float(getattr(o, "limit_price", 0.0))
        except Exception:
            continue
        if st in ("pending_cancel","pending_replace"):
            continue
        drift = abs(lim - ref_px) / max(ref_px, 1e-9)
        if age >= older_than_sec and drift > reprice_pct:
            try:
                tc.cancel_order_by_id(o.id)
                print(f"[cancel] {o.side} {symbol} id={o.id} age={int(age)}s (drift={drift:.5f})")
                n += 1
            except Exception as e:
                print(f"[cancel-error] id={o.id} {e}")
    return n

def get_position_and_free(tc: TradingClient, symbol: str, qty_dec: int) -> Tuple[float, float]:
    """
    Returns (total_position_qty, free_qty_available) for `symbol`.
    `free_qty_available` is what isn't reserved by open orders.
    Falls back to total qty if broker doesn't expose `qty_available`.
    """
    total = 0.0
    free  = 0.0
    try:
        target = symbol_for_trade(symbol)
        for p in (tc.get_all_positions() or []):
            sym = str(getattr(p, "symbol", "")).upper().replace("/", "")
            if sym == target:
                total = float(getattr(p, "qty", 0.0) or 0.0)
                qa    = getattr(p, "qty_available", None)
                free  = float(qa) if qa is not None else total
                break
    except Exception:
        pass
    return floor_qty(total, qty_dec), floor_qty(free, qty_dec)

def place_limit_order(tc: TradingClient, cfg: Cfg, side: OrderSide, symbol: str, qty: float, ref_price: float):
    # derive a conservative limit around last trade
    if side == OrderSide.BUY:
        limit = round_price(ref_price * (1 - cfg.limit_away_pct), cfg.price_dec)
    else:
        limit = round_price(ref_price * (1 + cfg.limit_away_pct), cfg.price_dec)

    if side == OrderSide.BUY:
        acct      = tc.get_account()
        raw_avail = get_conservative_available_usd(acct)
        committed = open_buy_notional(tc, symbol)
        reserve   = cfg.usd_cash_reserve
        haircut   = env_float("AVAILABLE_SAFETY_HAIRCUT", 0.003)  # 0.3% buffer

        afford = max(0.0, raw_avail - committed - reserve)
        afford *= max(0.0, 1.0 - haircut)

        # Max we can afford at this limit, floored to qty precision
        px = max(limit, 1e-9)
        max_qty = floor_qty(afford / px, cfg.qty_dec)

        if max_qty <= 0.0:
            print(f"[skip] buy skipped: avail ${raw_avail:.2f} - committed ${committed:.2f} - reserve ${reserve:.2f}")
            return None

        # --- HARD GUARD: broker/App minimum notional (cfg.min_notional_usd) ---
        tick = 10 ** (-cfg.qty_dec)
        min_qty = math.ceil((cfg.min_notional_usd / px) / tick) * tick
        min_qty = round(min_qty, cfg.qty_dec)

        if max_qty < min_qty:
            print(f"[skip] BUY notional ${max_qty * limit:.2f} < MIN_NOTIONAL_USD ${cfg.min_notional_usd:.2f}")
            return None

        if qty > max_qty:
            print(f"[resize] BUY qty {qty} -> {max_qty} based on avail ${raw_avail:.2f} "
                  f"(committed ${committed:.2f}) @ limit ${limit:.2f}")

        # Ensure we send at least broker minimum, but not above what we can afford
        qty = max(min(qty, max_qty), min_qty)
        qty = round_qty(qty, cfg.qty_dec)

    # SELL path is unchanged in your file; qty stays as passed in.

    if qty <= 0.0:
        print(f"[skip] {side.name} qty=0 after sizing/flooring")
        return None

    req = LimitOrderRequest(
        symbol=symbol_for_trade(symbol),
        qty=qty,
        side=side,
        time_in_force=cfg.tif,
        limit_price=limit,
        client_order_id=f"psl-{symbol_for_data(symbol)}-{int(time.time())}-{side.name.lower()}",
    )

    if cfg.dry_run:
        print(f"[dry-run] submit {side.name} {symbol} qty={qty} limit={limit} tif={cfg.tif.name}")
        return None

    try:
        order = tc.submit_order(req)
        print(f"[order] {side.name} {symbol} qty={qty} limit={limit} tif={cfg.tif.name} id={order.id}")
        INFLIGHT[symbol_for_data(symbol)].append({
            "id": str(getattr(order, "id", "")),
            "coid": req.client_order_id,
            "side": side.name.lower(),
            "qty": float(qty),
            "ts": utcnow(),
        })
        return order
    except Exception as e:
        print(f"[order-error] {side.name} {symbol} qty={qty} limit={limit} -> {e}")
        try:
            import json
            err = json.loads(str(e))
            msg = str(err.get("message","")).lower()
            if int(err.get("code", 0)) == 40310000 and "insufficient balance" in msg:
                SIDE_COOLDOWN[(symbol_for_data(symbol), side.name.lower())] = utcnow()
                print(f"[cooldown] {side.name} for {symbol} due to broker reservation ({INSUFF_COOLDOWN_SEC}s)")
        except Exception:
            pass
        return None

def desired_limit_for(side, ref_px, cfg):
    """Where we *want* our limit given the ref price + policy."""
    off = 1 - cfg.limit_away_pct if side == OrderSide.BUY else 1 + cfg.limit_away_pct
    return round_price(ref_px * off, cfg.price_dec)

def pct_diff(a, b):
    return abs(a - b) / max(1e-9, b)

def manage_open_order(tc: TradingClient, cfg: Cfg, side: OrderSide, ref_px: float) -> bool:
    """
    If an open order exists for the symbol:
      - If price drift > REPRICE_PCT and order age > MIN_REPRICE_SEC -> replace (preferred) or cancel+repost.
      - Else let it sit, even past CANCEL_AFTER_SEC (no churn).
    Returns True if an open order still exists after management.
    """
    oo = list_open_orders(tc, cfg.symbol)
    if not oo:
        return False

    # env knobs (sensible defaults)
    MIN_REPRICE_SEC = env_int("MIN_REPRICE_SEC", 45)
    REPRICE_PCT     = env_float("REPRICE_PCT", 0.0005)  # 5 bps
    MAX_LIFETIME_SEC= env_int("MAX_LIFETIME_SEC", 900)  # 15 min absolute cap

    now = utcnow()
    want_limit = desired_limit_for(side, ref_px, cfg)

    # Pick the newest open order for this direction
    o = sorted(oo, key=lambda x: str(getattr(x, "created_at", "")), reverse=True)[0]
    try:
        ots = pd.to_datetime(o.created_at, utc=True).to_pydatetime()
    except Exception:
        ots = now
    age = (now - ots).total_seconds()
    st  = str(getattr(o, "status", "")).lower()
    lim = float(getattr(o, "limit_price", want_limit))

    # Skip if broker is already canceling/replacing
    if st in ("pending_cancel", "pending_replace"):
        print(f"[pending] 1 open orders ages=[{int(age)}]s — waiting (status={st})")
        return True

    need_reprice = (pct_diff(lim, want_limit) > REPRICE_PCT) and (age >= MIN_REPRICE_SEC)

    if need_reprice:
        # Try in-place replace first; if not available, fall back to cancel+reissue
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            req = ReplaceOrderRequest(limit_price=want_limit)
            tc.replace_order_by_id(o.id, req)
            print(f"[replace] id={o.id} limit {lim} -> {want_limit}")
            return True
        except Exception as e:
            print(f"[replace-fallback] {e} -> cancel+repost")
            try:
                tc.cancel_order_by_id(o.id)
                print(f"[cancel] {o.side} {cfg.symbol} qty={o.qty} id={o.id} age={int(age)}s (for reprice)")
            except Exception as ce:
                if "pending cancel" in str(ce).lower():
                    print(f"[cancel-wait] id={o.id} already pending cancel")
                    return True
                print(f"[cancel-error] id={o.id} {ce}")
                return True  # keep treating as open until broker updates
            return False  # allow caller to place a fresh order this loop

    # Absolute max lifetime guard (rare)
    if age >= MAX_LIFETIME_SEC:
        try:
            tc.cancel_order_by_id(o.id)
            print(f"[cancel] {o.side} {cfg.symbol} id={o.id} age={int(age)}s (max lifetime)")
        except Exception as e:
            print(f"[cancel-error] id={o.id} {e}")
        return False

    # Otherwise: keep it; don’t churn
    print(f"[pending] 1 open orders ages=[{int(age)}]s — not repricing (|Δ|={pct_diff(lim, want_limit):.5f})")
    return True

# ------------------------- Sizing logic ----------------------

def desired_qty(cfg: Cfg, equity: float, ref_price: float, signal: int) -> float:
    if not (ref_price and ref_price > 0):
        return 0.0
    target_notional = (equity * cfg.max_leverage) if signal == 1 else 0.0
    target_notional = min(target_notional, cfg.usd_notional_cap)
    q = target_notional / ref_price
    return round_qty(q, cfg.qty_dec)

def clamp_trade_qty(cfg: Cfg, delta_qty: float, pos_qty: float) -> float:
    """
    Clamp by MAX_QTY_PER_TRADE and by available position for sells.
    delta_qty is SIGNED (tgt - pos). Long-only (no shorts).
    """
    if delta_qty > 0:  # BUY
        return min(delta_qty, cfg.max_qty_per_trade)

    elif delta_qty < 0:  # SELL
        sellable = max(0.0, pos_qty)  # (optional: subtract pending sell qty)
        want     = abs(delta_qty)
        return min(want, cfg.max_qty_per_trade, sellable)

    else:
        return 0.0

# --------------------------- Main loop -----------------------

def main():
    cfg = build_cfg()
    print("=== Pseudo-Live Crypto Trader ===")
    print(f"Symbol={cfg.symbol}  Strategy={cfg.strategy}  Paper={cfg.paper}  DryRun={cfg.dry_run}")
    print(f"TIF={cfg.tif.name}  LimitAway={cfg.limit_away_pct:.4f}  Poll={cfg.poll_sec}s")
    print(f"Decimals: price={cfg.price_dec} qty={cfg.qty_dec}")
    print(f"Sizing: MaxLev={cfg.max_leverage} NotionalCap=${cfg.usd_notional_cap} MaxQtyPerTrade={cfg.max_qty_per_trade}")
    print(f"CancelAfter={cfg.cancel_after_sec}s  MinNotional=${cfg.min_notional_usd}")
    print("------------------------------------------------------------")

    tcli = mk_trading_client(cfg.paper)
    dcli = mk_data_client()

    last_bar_ts: Optional[pd.Timestamp] = None
    last_signal: Optional[int] = None  # to log transitions

    while True:
        loop_start = time.time()

        # 2) Get latest bars (only if new bar to save rate limits)
        try:
            bars = fetch_bars(dcli, cfg.symbol, cfg.bar_lookback_min)
        except Exception as e:
            print(f"[error] fetch_bars: {e}")
            time.sleep(cfg.poll_sec)
            continue

        if bars.empty:
            print("[warn] no bars; retrying…")
            time.sleep(cfg.poll_sec)
            continue

        cur_bar_ts = bars.index[-1]
        new_bar = (last_bar_ts is None) or (cur_bar_ts > last_bar_ts)
        if new_bar:
            last_bar_ts = cur_bar_ts

        # 3) Compute signal on latest data
        try:
            sig, ref_close = compute_signal(cfg, bars)
        except Exception as e:
            print(f"[error] compute_signal: {e}")
            time.sleep(cfg.poll_sec)
            continue
        if last_signal is None or sig != last_signal:
            print(f"[signal] ts={cur_bar_ts} -> {sig}")
            if last_signal is not None and cfg.cancel_on_flip:
                try:
                    n = cancel_all_orders_for_symbol(tcli, cfg.symbol)
                    if n > 0:
                        print(f"[info] canceled {n} open orders due to signal flip")
                except Exception as e:
                    print(f"[warn] cancel_on_flip: {e}")
            last_signal = sig

        # 4) Use latest trade if fresh; otherwise fall back to bar close (degraded mode)
        TRADE_STALE_S = env_int("TRADE_STALE_SEC", 180)   # ~3 min
        BAR_FRESH_S   = env_int("BAR_FRESH_SEC", 180)

        last_trade_px, last_trade_ts = get_latest_trade_with_ts(dcli, cfg.symbol)
        now_utc = utcnow()
        bar_age   = (now_utc - last_bar_ts).total_seconds() if last_bar_ts else 9e9
        trade_age = (now_utc - last_trade_ts).total_seconds() if last_trade_ts else 9e9

        if trade_age <= TRADE_STALE_S:
            last_px = last_trade_px
            mode = "normal"
        elif bar_age <= BAR_FRESH_S:
            last_px = ref_close
            mode = "bar_proxy"
            print(f"[data_guard] WARN: Latest trade stale ({int(trade_age)}s) but bars fresh ({int(bar_age)}s). Using bar close.")
        else:
            # both stale — skip this loop to avoid acting on bad data
            print(f"[data_guard] WARN: Both trades ({int(trade_age)}s) and bars ({int(bar_age)}s) are stale. Skipping.")
            time.sleep(cfg.poll_sec)
            continue

        last_px = round_price(last_px or ref_close, cfg.price_dec)

        # 5) Read account & current position
        try:
            eq = get_equity(tcli)
            pos_qty = get_symbol_position_qty(tcli, cfg.symbol)
        except Exception as e:
            print(f"[error] account/position: {e}")
            time.sleep(cfg.poll_sec)
            continue

        # 6) Decide desired qty (long-only)
        tgt_qty = desired_qty(cfg, eq, last_px, sig)
        delta = tgt_qty - pos_qty

        # 7) Skip if tiny or already aligned
        min_qty_from_notional = cfg.min_notional_usd / max(last_px, 1e-9)
        # require at least 2 ticks or min-notional, whichever is larger
        min_delta = max(2 * (10 ** (-cfg.qty_dec)), min_qty_from_notional)
        # If we're flat, kill any working orders so we don't drift
        if sig == 0:
            try:
                oo = list_open_orders(tcli, cfg.symbol)
                if oo:
                    n = cancel_all_orders_for_symbol(tcli, cfg.symbol)
                    if n > 0:
                        print(f"[info] canceled {n} open orders (signal=0)")
            except Exception as e:
                print(f"[warn] cancel_on_flat: {e}")
        if abs(delta) < min_delta:
            print(f"[align] ts={cur_bar_ts} px={last_px} sig={sig} eq={eq:.2f} pos={pos_qty:.{cfg.qty_dec}f} tgt={tgt_qty:.{cfg.qty_dec}f} (Δ<{min_delta:.{cfg.qty_dec}f})")
        else:
            # 8) If an open order exists, manage it (maybe replace), else place fresh
            oo_exists = manage_open_order(
                tcli,
                cfg,
                OrderSide.BUY if delta > 0 else OrderSide.SELL,
                last_px
            )
            if oo_exists:
                pass  # do not place a new order this loop
            else:
                # 9) Place order (clamped)
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                qty  = clamp_trade_qty(cfg, delta, pos_qty)

                # Cooldown guard after a 40310000 on this side
                cd_key = (symbol_for_data(cfg.symbol), side.name.lower())
                last_cd = SIDE_COOLDOWN[cd_key]
                if (utcnow() - last_cd).total_seconds() < INSUFF_COOLDOWN_SEC:
                    left = INSUFF_COOLDOWN_SEC - int((utcnow() - last_cd).total_seconds())
                    print(f"[skip] {side.name} cooldown {left}s (broker reservation)")
                    elapsed = time.time() - loop_start
                    time.sleep(max(0.5, cfg.poll_sec - elapsed))
                    continue

                if side == OrderSide.SELL:
                    # Derive pending directly from position:
                    # pending_from_pos = total position - broker free (qty_available)
                    pos_total, pos_free = get_position_and_free(tcli, cfg.symbol, cfg.qty_dec)
                    pending_from_pos = max(0.0, pos_total - pos_free)
                    # Also add what we can see via order listing / inflight (belt & suspenders)
                    pending_from_orders = open_sell_qty(tcli, cfg.symbol) + inflight_side_qty(cfg.symbol, "sell")
                    pend = max(pending_from_pos, pending_from_orders)

                    tick = 10 ** (-cfg.qty_dec)
                    sellable_free = max(0.0, pos_total - pend)
                    # keep a tiny reserve to avoid dust/rounding bumps
                    sellable_free = max(0.0, sellable_free - cfg.base_qty_reserve_ticks * tick)

                    if qty > sellable_free:
                        print(f"[resize] SELL qty {qty} -> {sellable_free} (pos_total={pos_total}, pos_free={pos_free}, pend={pend})")
                    qty = floor_qty(min(qty, sellable_free), cfg.qty_dec)
                    if qty <= 0.0:
                        print("[skip] SELL qty=0 (reserved by open/partial)")
                        elapsed = time.time() - loop_start
                        time.sleep(max(0.5, cfg.poll_sec - elapsed))
                        continue

                notional = qty * last_px
                if notional < cfg.min_notional_usd:
                    print(f"[skip] notional ${notional:.2f} < MIN_NOTIONAL_USD ${cfg.min_notional_usd:.2f}")
                else:
                    place_limit_order(tcli, cfg, side, cfg.symbol, qty, last_px)
        # 10) Sleep until next poll (account for time spent)
        elapsed = time.time() - loop_start
        to_sleep = max(0.5, cfg.poll_sec - elapsed)
        time.sleep(to_sleep)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye 👋")
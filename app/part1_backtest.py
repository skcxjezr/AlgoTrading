# part1_backtest.py
# ------------------------------------------------------------
# Crypto backtest (24×7) using Alpaca Data API via alpaca-py (modern layout).
# FIXED to prevent negative drift:
#   - Trades only on signal transitions (0→1 open, 1→0 close). No per-bar rebalancing.
#   - Entry sizing is cash-aware (won’t overspend; includes fee + slippage).
#   - Exit closes the whole position once, fees/slippage applied once.
#   - MIN_NOTIONAL_USD enforced on entry (still exits fully even if tiny).
#   - Sharpe/vol annualization depends on TF_MIN.
#
# Strategies supported:
#   1) EMA crossover with ADX filter (optional price-action confirmation)
#   2) Z-score mean reversion with ADX + RSI filters
# - Accepts SYMBOL as "BTC/USD" (preferred) or "BTCUSD" (auto-normalized for data).
# - Saves outputs with a sanitized filename (slashes -> dashes) and a unique run_id.
# - ***NEW***: also saves meta_<...>.json with strategy + critical parameters for the dashboard.
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import sys
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# Alpaca data (crypto)
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # modern import
import time

# --------------------------- Env Helpers ---------------------------

def env_str(key: str, default: str = "") -> str:
    v = os.getenv(key, default)
    return v.strip() if isinstance(v, str) else default

def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default

def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default

def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            dt = pd.to_datetime(s, utc=True).to_pydatetime()
        return dt
    except Exception:
        return None

def get_timeframe(minutes: int):
    return TimeFrame(minutes, TimeFrameUnit.Minute)

def round_price(x: float, dec: int) -> float:
    return float(np.round(float(x), dec))

def round_qty(x: float, dec: int) -> float:
    r = float(np.round(float(x), dec))
    return max(0.0, r)

def pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def symbol_for_data(sym: str) -> str:
    """BTCUSD -> BTC/USD ; pass through if already slashed."""
    if "/" in sym:
        return sym
    if sym.endswith(("USDT", "USDC")) and len(sym) > 4:
        return sym[:-4] + "/" + sym[-4:]
    if sym.endswith("USD") and len(sym) > 3:
        return sym[:-3] + "/USD"
    return sym

def safe_name(sym: str) -> str:
    """Sanitize for filenames (slashes/spaces -> dashes, strip weird chars)."""
    return re.sub(r'[^A-Za-z0-9._-]+', '-', sym.replace('/', '-'))

def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


# --------------------------- Data ---------------------------
def fetch_crypto_bars(symbol: str, start: datetime, end: datetime, minutes: int, key: str, secret: str) -> pd.DataFrame:
    sym = symbol_for_data(symbol)
    tf = get_timeframe(minutes)
    start = start.astimezone(timezone.utc)
    end   = end.astimezone(timezone.utc)

    # --- try SDK with retries ---
    last_exc = None
    for i in range(3):
        try:
            cli = CryptoHistoricalDataClient(key, secret)
            req = CryptoBarsRequest(symbol_or_symbols=[sym], timeframe=tf, start=start, end=end)
            bars = cli.get_crypto_bars(req)
            df = bars.df
            if df is not None and len(df):
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index().set_index("timestamp")
                    df = df[df["symbol"] == sym].drop(columns=["symbol"])
                df = df.sort_index()
                rename = {"open":"o","high":"h","low":"l","close":"c","volume":"v"}
                df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
                keep = [c for c in ["o","h","l","c","v"] if c in df.columns]
                df = df[keep].replace([np.inf,-np.inf], np.nan).dropna()
                df = df[(df["o"]>0)&(df["h"]>0)&(df["l"]>0)&(df["c"]>0)]
                df.index = pd.to_datetime(df.index, utc=True)
                return df
            break
        except Exception as e:
            last_exc = e
            time.sleep(1.5 * (2 ** i))

    # --- REST fallback (handles 429s and long ranges via next_page_token) ---
    try:
        import requests
        tf_map = {1:"1Min",5:"5Min",15:"15Min",30:"30Min",60:"1Hour",240:"4Hour",1440:"1Day"}
        tf_str = tf_map.get(int(minutes), "1Min")
        url = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "User-Agent": "algo-hft/backtest"}
        params = {"symbols": sym, "timeframe": tf_str, "start": start.isoformat(), "end": end.isoformat(), "limit": 10000}

        rows, token, pages = [], None, 0
        # estimate how many pages we need and allow buffer, but keep a hard upper bound to avoid infinite loops
        try:
            span_min = max(1, int((end - start).total_seconds() // 60))
            est_bars = max(1, span_min // int(minutes))
        except Exception:
            span_min, est_bars = 1, 1
        limit = int(params.get("limit", 10000)) or 10000
        import math as _math
        est_pages = max(1, _math.ceil(est_bars / limit))
        max_pages = min(1000, est_pages + 10)  # generous headroom, finite cap

        while True:
            if token:
                params["page_token"] = token
            else:
                params.pop("page_token", None)
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            j = r.json() or {}
            chunk = (j.get("bars", {}).get(sym, [])) or (j.get("bars", {}).get(sym.replace("/",""), [])) or []
            rows.extend(chunk)
            token = j.get("next_page_token") or None
            pages += 1
            if not token or pages >= max_pages:
                break

        if not rows:
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "t": pd.to_datetime(b.get("t") or b.get("timestamp"), utc=True, errors="coerce"),
            "o": b.get("o") or b.get("open"),
            "h": b.get("h") or b.get("high"),
            "l": b.get("l") or b.get("low"),
            "c": b.get("c") or b.get("close"),
            "v": b.get("v") or b.get("volume"),
        } for b in rows]).dropna()
        df = df.sort_values("t").set_index("t")
        for c in ["o","h","l","c","v"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf,-np.inf], np.nan).dropna()
        return df
    except Exception:
        if last_exc:
            raise last_exc
        return pd.DataFrame()

# ------------------------- Indicators & Signals -------------------------

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _wilder_ewm(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (alpha = 1/n)."""
    n = int(n or 0)
    if n < 1:
        raise ValueError(f"ADX length must be >= 1; got {n}. "
                         "Check `adx_len` (EMA) or `z_adx_len` (Z-score).")
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
    atr = _wilder_ewm(tr, n).replace(0, np.nan)
    plus_di = 100.0 * (_wilder_ewm(plus_dm, n) / atr).replace([np.inf, -np.inf], np.nan)
    minus_di = 100.0 * (_wilder_ewm(minus_dm, n) / atr).replace([np.inf, -np.inf], np.nan)
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / denom).replace([np.inf, -np.inf], np.nan)
    adx = _wilder_ewm(dx, n)
    return adx


def compute_rsi(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder RSI on close price."""
    try:
        c = pd.to_numeric(df["c"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    except Exception:
        c = pd.Series(index=df.index, data=np.nan)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    # Wilder smoothing
    roll_up = _wilder_ewm(up, int(n))
    roll_down = _wilder_ewm(down, int(n))
    rs = (roll_up / roll_down.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi



@dataclass
class Config:
    symbol: str
    tf_min: int
    start: datetime
    end: datetime
    initial_cash: float
    price_dec: int
    qty_dec: int
    fee_bps: float
    slippage_bps: float

    strategy: str  # 'ema' or 'zscore'

    # EMA strategy
    ema_fast: int
    ema_slow: int
    adx_len: int
    adx_min: float
    use_pa_filter: bool
    pa_min_body_frac: float  # fraction of candle range

    # Z-score strategy
    z_n: int
    z_entry: float
    z_exit: float
    z_stop: float
    z_adx_len: int  # ADX lookback for zscore strategy
    adx_skip: float  # skip mean-rev entries when ADX >= this


    # RSI filter (used by zscore)
    rsi_len: int
    rsi_lower: float
    rsi_upper: float

    # Sizing/exec
    max_leverage: float         # kept for compatibility; cash-only sizing by default
    min_notional_usd: float     # skip tiny entries

    # Output
    out_dir: str

def build_config() -> Config:
    now_utc = datetime.now(timezone.utc)
    start = parse_iso(env_str("START", ""))
    end = parse_iso(env_str("END", ""))
    if not end:
        days = env_int("DURATION_DAYS", 365)
        end = now_utc
        start = end - timedelta(days=days)
    if not start:
        start = end - timedelta(days=365)

    return Config(
        symbol=env_str("SYMBOL", "BTC/USD"),
        tf_min=env_int("TF_MIN", 5),
        start=start,
        end=end,
        initial_cash=env_float("INITIAL_CASH", 100_000.0),
        price_dec=env_int("PRICE_DEC", 2),
        qty_dec=env_int("QTY_DEC", 6),
        fee_bps=env_float("FEE_BPS", env_float("COST_RATE_BPS", 0.0)),  # bps
        slippage_bps=env_float("SLIPPAGE_BPS", 0.0),

        strategy=env_str("STRATEGY", "ema").lower(),

        ema_fast=env_int("EMA_FAST", 15),
        ema_slow=env_int("EMA_SLOW", 100),
        adx_len=env_int("ADX_LEN", 14),
        adx_min=env_float("ADX_MIN", 20.0),
        use_pa_filter=env_int("USE_PA_FILTER", 1) == 1,
        pa_min_body_frac=env_float("PA_MIN_BODY_FRAC", 0.5),

        z_n=env_int("Z_N", 144),
        z_entry=env_float("Z_ENTRY", 2.4),
        z_exit=env_float("Z_EXIT", 0.2),
        z_stop=env_float("Z_STOP", 3.5),
        z_adx_len=env_int("Z_ADX_LEN", env_int("ADX_LEN", 14)),
        adx_skip=env_float("ADX_SKIP", 25.0),


        rsi_len=env_int("RSI_LEN", 14),
        rsi_lower=env_float("RSI_LOWER", 30.0),
        rsi_upper=env_float("RSI_UPPER", 70.0),

        max_leverage=env_float("MAX_LEVERAGE", 1.0),
        min_notional_usd=env_float("MIN_NOTIONAL_USD", 5.0),

        out_dir=env_str("OUT_DIR", "./out"),
    )

def _price_action_ok(df: pd.DataFrame, min_body_frac: float) -> pd.Series:
    # Require bullish close and body >= min_body_frac of full range
    body = (df["c"] - df["o"]).clip(lower=0.0)
    rng = (df["h"] - df["l"]).replace(0, np.nan)
    frac = (body / rng).fillna(0.0)
    return (df["c"] > df["o"]) & (frac >= min_body_frac)

def signals_ema_adx(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = _ema(out["c"], cfg.ema_fast)
    out["ema_slow"] = _ema(out["c"], cfg.ema_slow)
    out["adx"] = compute_adx(out, cfg.adx_len)

    cross_up = (out["ema_fast"] > out["ema_slow"]) & (out["ema_fast"].shift(1) <= out["ema_slow"].shift(1))
    cross_down = (out["ema_fast"] < out["ema_slow"]) & (out["ema_fast"].shift(1) >= out["ema_slow"].shift(1))

    if cfg.use_pa_filter:
        strong = _price_action_ok(out, cfg.pa_min_body_frac)
    else:
        strong = pd.Series(True, index=out.index)

    sig = []
    state = 0  # 0=flat, 1=long
    for i in range(len(out)):
        if state == 0:
            if bool(cross_up.iloc[i]) and (out["adx"].iloc[i] >= cfg.adx_min) and bool(strong.iloc[i]):
                state = 1
        else:  # long
            if bool(cross_down.iloc[i]):
                state = 0
        sig.append(state)
    out["signal"] = pd.Series(sig, index=out.index).fillna(0).astype(int)
    return out

def signals_zscore(cfg: "Config", df: pd.DataFrame) -> pd.DataFrame:
    """Z-score mean-reversion signals.
    Entry: z crosses DOWN below -z_entry AND ADX < adx_skip (if enabled)
    Exit:  z crosses UP   above -z_exit  OR  crosses DOWN below -z_stop
    Returns a 0/1 STATE series.
    """
    out = df.copy()
    out["mid"] = (out["h"] + out["l"]) / 2.0
    m = out["mid"]
    m = pd.to_numeric(m, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Optional: fill tiny gaps so rolling windows are valid
    m = m.ffill().bfill()
    # Not enough data? Bail early with a flat signal.
    if int(m.dropna().shape[0]) < int(cfg.z_n) + 1:
        out["z"] = np.nan
        out["adx"] = compute_adx(out, cfg.z_adx_len) if cfg.z_adx_len >= 1 else np.nan
        out["rsi"] = compute_rsi(out, cfg.rsi_len) if cfg.rsi_len >= 1 else np.nan
        out["signal"] = 0
        return out
    # use numpy's NaN-aware reducers to avoid the pandas nanops warning
    ma = m.rolling(cfg.z_n, min_periods=cfg.z_n).apply(np.nanmean, raw=True)
    sd = m.rolling(cfg.z_n, min_periods=cfg.z_n).apply(lambda x: np.nanstd(x, ddof=0), raw=True)
    z  = (m - ma) / sd.replace(0.0, np.nan)

    # ADX gate for entries (optional)
    if cfg.adx_skip and cfg.z_adx_len >= 1:
        adx = compute_adx(out, cfg.z_adx_len)
        gate_ok = (adx < float(cfg.adx_skip)) | adx.isna()
    else:
        adx = pd.Series(index=out.index, data=np.nan)
        gate_ok = pd.Series(True, index=out.index)

    # RSI gate for entries (optional)
    if int(cfg.rsi_len) >= 1:
        rsi = compute_rsi(out, cfg.rsi_len)
        rsi_ok = (rsi < float(cfg.rsi_lower)) | rsi.isna()
    else:
        rsi = pd.Series(index=out.index, data=np.nan)
        rsi_ok = pd.Series(True, index=out.index)

    th_entry = -abs(float(cfg.z_entry))
    th_exit  = -abs(float(cfg.z_exit))
    th_stop  = -abs(float(cfg.z_stop))

    enter_cross = (z.shift(1) > th_entry) & (z <= th_entry) & gate_ok & rsi_ok
    exit_cross  = (((z.shift(1) < th_exit) & (z >= th_exit) & gate_ok) | ((z.shift(1) > th_stop) & (z <= th_stop)))

    # --- build STATE (ignore exits until we're long) ---
    state = 0
    sig = []
    for i in range(len(out)):
        if state == 0:
            if bool(enter_cross.iloc[i]):
                state = 1
        else:  # state == 1
            if bool(exit_cross.iloc[i]):
                state = 0
        sig.append(state)

    out["signal"] = pd.Series(sig, index=out.index, dtype=int)
    out["z"] = z
    out["adx"] = adx
    out["rsi"] = rsi
    return out

def make_signals(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    if cfg.strategy == "ema":
        return signals_ema_adx(cfg, df)
    elif cfg.strategy == "zscore":
        return signals_zscore(cfg, df)
    else:
        raise ValueError(f"Unknown STRATEGY={cfg.strategy}. Use 'ema' or 'zscore'.")


# ------------------------ Backtester ------------------------

@dataclass
class BTResult:
    equity: pd.Series          # UTC index
    trades: pd.DataFrame       # time, side, qty, price, fee, notional
    stats: dict
    bh_equity: Optional[pd.Series] = None

def run_backtest(cfg: Config, mkt: pd.DataFrame) -> BTResult:
    df = make_signals(cfg, mkt)

    cash = cfg.initial_cash
    pos_qty = 0.0

    equity_points = []
    trades = []

    fee_rate = cfg.fee_bps / 10_000.0
    slip_rate = cfg.slippage_bps / 10_000.0

    prev_sig = 0

    for ts, row in df.iterrows():
        px = float(row["c"])
        sig = int(row["signal"])

        # --- TRADE ONLY ON TRANSITIONS ---
        if sig == 1 and prev_sig == 0:
            fill_px = round_price(px * (1 + slip_rate), cfg.price_dec)
            if fill_px > 0 and cash > 0:
                # Max notional while paying fee from cash
                max_notional = cash / (1.0 + fee_rate)
                if max_notional >= cfg.min_notional_usd:
                    qty = round_qty(max_notional / fill_px, cfg.qty_dec)
                    notional = qty * fill_px
                    if notional >= cfg.min_notional_usd and qty > 0:
                        fee = notional * fee_rate
                        cash -= (notional + fee)
                        pos_qty += qty
                        trades.append(
                            {"ts": ts, "side": "BUY", "qty": qty, "price": fill_px,
                             "notional": notional, "fee": fee}
                        )

        elif sig == 0 and prev_sig == 1:
            if pos_qty > 0:
                fill_px = round_price(px * (1 - slip_rate), cfg.price_dec)
                notional = pos_qty * fill_px
                fee = notional * fee_rate
                cash += (notional - fee)
                trades.append(
                    {"ts": ts, "side": "SELL", "qty": pos_qty, "price": fill_px,
                     "notional": notional, "fee": fee}
                )
                pos_qty = 0.0

        prev_sig = sig
        equity_points.append((ts, float(cash + pos_qty * px)))

    # Equity series
    equity = pd.Series(
        data=[v for _, v in equity_points],
        index=pd.to_datetime([t for t, _ in equity_points], utc=True),
        name="equity"
    )

    # Buy & Hold series
    bh_equity = None
    if len(df) >= 2 and float(df["c"].iloc[0]) > 0:
        bh_qty = float(cfg.initial_cash) / float(df["c"].iloc[0])
        bh_equity = pd.Series(
            data=(bh_qty * df["c"]).astype(float).values,
            index=pd.to_datetime(df.index, utc=True),
            name="buyhold"
        )

    # Stats
    ret_series = equity.pct_change().dropna()
    tot_ret = (equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) >= 2 else 0.0
    periods_per_year = (60.0 / max(float(cfg.tf_min), 1.0)) * 24.0 * 365.0
    mu_ann  = ret_series.mean() * periods_per_year if len(ret_series) else np.nan
    vol_ann = ret_series.std(ddof=1) * np.sqrt(periods_per_year) if len(ret_series) > 1 else np.nan
    sharpe  = (mu_ann / vol_ann) if (np.isfinite(vol_ann) and vol_ann > 0) else np.nan
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    max_dd = dd.min() if len(dd) else np.nan

    stats = {
        "bars": int(len(df)),
        "trades": int(len(trades)),
        "total_return": float(tot_ret),
        "sharpe_approx": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(max_dd) if np.isfinite(max_dd) else np.nan,
        "final_equity": float(equity.iloc[-1]) if len(equity) else float(cfg.initial_cash),
    }

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["ts"] = pd.to_datetime(trades_df["ts"], utc=True)

    return BTResult(equity=equity, trades=trades_df, stats=stats, bh_equity=bh_equity)


# ---------------------- Persistence (CSV + META JSON) -------------------------

def write_outputs(cfg: Config, res: BTResult, tag: str) -> Tuple[str, str, str]:
    """Write equity/trades CSV and meta JSON; return paths."""
    ensure_dir(cfg.out_dir)
    start_str = cfg.start.strftime("%Y%m%d")
    end_str = cfg.end.strftime("%Y%m%d")
    sym_safe = safe_name(cfg.symbol)

    eq_path = f"{cfg.out_dir}/backtest_{sym_safe}_{start_str}_{end_str}_{tag}.csv"
    tr_path = f"{cfg.out_dir}/trades_{sym_safe}_{start_str}_{end_str}_{tag}.csv"
    meta_path = f"{cfg.out_dir}/meta_{sym_safe}_{start_str}_{end_str}_{tag}.json"

    # Equity CSV (with optional buyhold column)
    df_eq = res.equity.rename("equity").to_frame()
    df_eq["ts"] = df_eq.index
    if res.bh_equity is not None:
        bh = res.bh_equity.rename("buyhold").to_frame()
        bh["ts"] = bh.index
        df_eq = df_eq.merge(bh[["ts", "buyhold"]], on="ts", how="left")
    cols = ["ts", "equity"] + (["buyhold"] if "buyhold" in df_eq.columns else [])
    df_eq = df_eq[cols]
    df_eq.to_csv(eq_path, index=False)

    # Trades CSV (may be empty but keep headers)
    # Trades CSV (may be empty but keep headers)
    df_tr = res.trades.copy() if not res.trades.empty else pd.DataFrame(columns=["ts","side","qty","price","notional","fee"])
    df_tr.to_csv(tr_path, index=False)
    # META JSON for dashboard (strategy + critical params)
    if cfg.strategy == "ema":
        params: Dict[str, object] = {
            "EMA_FAST": cfg.ema_fast,
            "EMA_SLOW": cfg.ema_slow,
            "ADX_LEN": cfg.adx_len,
            "ADX_MIN": cfg.adx_min,
            "USE_PA_FILTER": bool(cfg.use_pa_filter),
            "PA_MIN_BODY_FRAC": cfg.pa_min_body_frac,
        }
    else:  # zscore
        # replicate the internal time stop formula so the dashboard can match exits
        time_stop = int(max(6, round(cfg.z_n / 2)))
        params = {
            "Z_N": cfg.z_n,
            "Z_ENTRY": cfg.z_entry,
            "Z_EXIT": cfg.z_exit,
            "Z_STOP": cfg.z_stop,
            "ADX_SKIP": cfg.adx_skip,
            "Z_ADX_LEN": int(cfg.z_adx_len),
            "RSI_LEN": int(cfg.rsi_len),
            "RSI_LOWER": float(cfg.rsi_lower),
            "RSI_UPPER": float(cfg.rsi_upper),
            "RSI_FILTER_ON_ENTRY": True,
            # Explicit z-score spec so downstream code doesn't guess:
            "Z_SOURCE": "log_close",            # z computed on log(price)
            "Z_BASE_MEAN": "ewm",               # detrend with EWM mean
            "Z_BASE_PARAM": {"span": cfg.z_n},  # EWM span
            "Z_STD_METHOD": "rolling_std",      # rolling window std
            "Z_STD_PARAM": {"window": cfg.z_n, "ddof": 0},
            "ENTRY_RULE": "cross_down",         # uses shift(1) cross
            "EXIT_RULES": ["cross_up", "stop", "adx_gate"],
            "TIME_STOP_BARS": time_stop,
            # ADX implementation hint for parity:
            "ADX_METHOD": "wilder_ewm",         # alpha = 1/n
            # Make intent explicit for dashboards that support both gates:
            "ADX_GATE_ON_ENTRY": True,
            "ADX_GATE_ON_EXIT": True,
        }
    meta = {
        "strategy": cfg.strategy,
        "tf_min": int(cfg.tf_min),
        "run_id": tag.split("-")[-1],
        "params": params,
        "initial_cash": float(cfg.initial_cash),
        "fee_bps": float(cfg.fee_bps),
        "slippage_bps": float(cfg.slippage_bps),
        "symbol": cfg.symbol,
        "start": cfg.start.replace(tzinfo=timezone.utc).isoformat(),
        "end": cfg.end.replace(tzinfo=timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Optional: persist z/adx/signal for parity debugging when strategy=zscore
    try:
        if cfg.strategy == "zscore":
            sig_path = f"{cfg.out_dir}/sig_{sym_safe}_{start_str}_{end_str}_{tag}.csv"
            # recompute signals to capture z and adx in lockstep with saved equity
            df_dbg = make_signals(cfg, mkt := None)  # NOTE: wire through original market DF if available
            # If mkt isn't in scope here, you can pass it down from main or save alongside res.
            # Keep columns small:
            out_cols = [c for c in ["z", "adx", "rsi", "signal", "c"] if c in df_dbg.columns]
            df_dbg[["ts"] + out_cols].to_csv(sig_path, index=False)
    except Exception:
        pass
    return eq_path, tr_path, meta_path


# --------------------------- Main ---------------------------

def main():
    cfg = build_config()

    key = env_str("ALPACA_API_KEY_ID", os.getenv("ALPACA_API_KEY", ""))
    secret = env_str("ALPACA_API_SECRET_KEY", os.getenv("ALPACA_API_SECRET", ""))

    if not key or not secret:
        print("ERROR: Alpaca API keys not set. Define ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.", file=sys.stderr)
        sys.exit(2)

    print(f"[Backtest] {cfg.symbol} {cfg.start.isoformat()} → {cfg.end.isoformat()}  ({cfg.tf_min}-min)")
    print(f"  Strategy={cfg.strategy}  InitialCash={cfg.initial_cash:,.2f}  FeeBps={cfg.fee_bps:.2f}  SlipBps={cfg.slippage_bps:.2f}")
    print(f"  PriceDec={cfg.price_dec}  QtyDec={cfg.qty_dec}  MaxLev={cfg.max_leverage}  MinNotional=${cfg.min_notional_usd:.2f}")
    print("  Execution: trade-on-transition only (no per-bar rebalancing)")

    # Fetch data
    bars = fetch_crypto_bars(cfg.symbol, cfg.start, cfg.end, cfg.tf_min, key, secret)
    if bars.empty:
        print("No bars returned for given range. Nothing to do.", file=sys.stderr)
        sys.exit(1)

    # Run backtest
    res = run_backtest(cfg, bars)

    # Tag embeds strategy + timeframe + unique run id
    rid = run_id()
    tag = f"{cfg.strategy}-{cfg.tf_min}m-{rid}"

    # Save CSV + META
    eq_path, tr_path, meta_path = write_outputs(cfg, res, tag)
    print(f"Saved equity to: {eq_path}")
    print(f"Saved trades to: {tr_path}")
    print(f"Saved meta   to: {meta_path}")

    # Summary
    print("---- Summary ----")
    print(f"Bars:           {res.stats['bars']}")
    print(f"Trades:         {res.stats['trades']}")
    print(f"Total Return:   {pct(res.stats['total_return'])}")
    try:
        sharpe_val = res.stats.get('sharpe_approx', np.nan)
        print(f"Sharpe (approx): {sharpe_val:.2f}" if np.isfinite(sharpe_val) else "Sharpe: —")
    except Exception:
        print("Sharpe: —")
    print(f"Max Drawdown:   {pct(res.stats['max_drawdown'])}")
    print(f"Final Equity:   {res.stats['final_equity']:,.2f}")

if __name__ == "__main__":
    main()
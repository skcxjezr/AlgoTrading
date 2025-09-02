# part3_data_guard.py
# ------------------------------------------------------------
# Crypto data sanity guard for Alpaca (alpaca-py).
#
# What it checks (smart heuristics tuned for 24×7 crypto):
#   • Gaps: missing 1-min bars; severity depends on % missing and recency
#   • Freshness: staleness of latest trade vs. how fresh your bars are
#   • Spread: bid/ask spread (bps) capped by MAX_SPREAD_BPS*
#   • Jump: last close vs. prior close move (bps) with warn/fail bands
#   • Candle shape: negative, inverted, or zero-range OHLCs
#
# *We auto-pick MAX_SPREAD_BPS from several common env names if unset.
#
# Exit codes:
#   0 = OK (no errors; warnings allowed unless STRICT=1)
#   1 = Warnings only (STRICT=0)
#   2 = Errors (or warnings treated as errors when STRICT=1)
#
# ENV (typical):
#   SYMBOL="BTC/USD"              (BTCUSD also accepted)
#   LOOKBACK_MIN=120              (history window for gap/jump checks)
#   STALE_TRADE_SEC=300           (latest trade older than this is stale)
#   BARS_FRESH_SEC=180            (bars considered fresh within this)
#   GAP_TOL_MIN=10                (max consecutive missing minutes before concern)
#   GAP_RECENT_MIN=60             (only “recent long gaps” are hard errors)
#   GAP_PCT_WARN=10              (warn if >10% minutes missing)
#   GAP_PCT_FAIL=20              (error if >20% minutes missing)
#   MAX_SPREAD_BPS=50             (warn/error if ask-bid exceeds this)
#   JUMP_WARN_BPS=300             (3.00% last-bar jump warns)
#   JUMP_FAIL_BPS=1500            (15.0% last-bar jump errors)
#   PRICE_DEC=2                   (for printing/rounding)
#   POLL_SEC=30                   (loop sleep if ONE_SHOT=0)
#   ONE_SHOT=0/1                  (run once or loop)
#   STRICT=0/1                    (treat warnings as errors if 1)
#
# Requires: alpaca-py >= 0.21, pandas, numpy
# ------------------------------------------------------------

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# Alpaca data
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestTradeRequest,
    CryptoLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# ------------------------------ Utils ------------------------------

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

def env_float_or_none(k: str) -> Optional[float]:
    v = os.getenv(k)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def get_timeframe(minutes: int):
    return TimeFrame(minutes, TimeFrameUnit.Minute)

def symbol_for_data(sym: str) -> str:
    """BTCUSD -> BTC/USD; pass through if already slashed."""
    if "/" in sym:
        return sym.strip().upper()
    s = sym.strip().upper()
    if s.endswith(("USDT", "USDC")) and len(s) > 4:
        return s[:-4] + "/" + s[-4:]
    if s.endswith("USD") and len(s) > 3:
        return s[:-3] + "/USD"
    return s


# ------------------------------ Models ------------------------------

@dataclass
class GuardCfg:
    symbol: str
    lookback_min: int
    gap_tol_min: int
    gap_recent_min: int
    gap_pct_warn: float
    gap_pct_fail: float
    stale_trade_sec: int
    bars_fresh_sec: int
    spread_cap_bps: float
    jump_warn_bps: float
    jump_fail_bps: float
    price_dec: int
    poll_sec: int
    one_shot: bool
    strict: bool

@dataclass
class CheckResult:
    errors: Dict[str, str] = field(default_factory=dict)
    warnings: Dict[str, str] = field(default_factory=dict)
    info: Dict[str, str] = field(default_factory=dict)

    def worst_code(self, strict: bool) -> int:
        if self.errors:
            return 2
        if self.warnings and strict:
            return 2
        if self.warnings:
            return 1
        return 0


# ------------------------------ Client ------------------------------

def mk_data_client() -> CryptoHistoricalDataClient:
    key = env_str("ALPACA_API_KEY_ID", os.getenv("ALPACA_API_KEY", ""))
    sec = env_str("ALPACA_API_SECRET_KEY", os.getenv("ALPACA_API_SECRET", ""))
    return CryptoHistoricalDataClient(key, sec)


# ------------------------------ Fetchers ------------------------------

def fetch_recent_bars(cli: CryptoHistoricalDataClient, symbol: str, minutes: int) -> pd.DataFrame:
    end = utcnow()
    start = end - timedelta(minutes=minutes + 5)
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
        df = (
            df.reset_index()
              .set_index("timestamp")
        )
        df = df[df["symbol"] == symbol_for_data(symbol)].drop(columns=["symbol"])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    # normalize cols
    ren = {"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
    df.rename(columns={k: v for k, v in ren.items() if k in df.columns}, inplace=True)
    keep = [c for c in ["o", "h", "l", "c", "v"] if c in df.columns]
    df = df[keep].replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["o"] > 0) & (df["h"] > 0) & (df["l"] > 0) & (df["c"] > 0)]
    return df

def fetch_latest_trade(cli: CryptoHistoricalDataClient, symbol: str) -> Tuple[Optional[float], Optional[datetime]]:
    try:
        t = cli.get_crypto_latest_trade(CryptoLatestTradeRequest(symbol_or_symbols=[symbol_for_data(symbol)]))
        rec = t[symbol_for_data(symbol)]
        px = float(getattr(rec, "price", None)) if hasattr(rec, "price") else None
        ts = pd.to_datetime(getattr(rec, "timestamp", None), utc=True).to_pydatetime() if hasattr(rec, "timestamp") else None
        return px, ts
    except Exception:
        return None, None

def fetch_latest_quote(cli: CryptoHistoricalDataClient, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[datetime]]:
    try:
        q = cli.get_crypto_latest_quote(CryptoLatestQuoteRequest(symbol_or_symbols=[symbol_for_data(symbol)]))
        rec = q[symbol_for_data(symbol)]
        bid = float(getattr(rec, "bid_price", None)) if hasattr(rec, "bid_price") else None
        ask = float(getattr(rec, "ask_price", None)) if hasattr(rec, "ask_price") else None
        ts = pd.to_datetime(getattr(rec, "timestamp", None), utc=True).to_pydatetime() if hasattr(rec, "timestamp") else None
        return bid, ask, ts
    except Exception:
        return None, None, None


# ------------------------------ Checks ------------------------------

def check_candle_shapes(df: pd.DataFrame) -> CheckResult:
    res = CheckResult()
    if df.empty:
        res.errors["bars.empty"] = "No bars returned."
        return res
    # tiny epsilon to absorb float round-off
    eps = 10 * np.finfo(float).eps
    out_of_bounds = (
        (df["h"] + eps < df[["o","c"]].max(axis=1)) |
        (df["l"] - eps > df[["o","c"]].min(axis=1)) |
        (df["h"] + eps < df["l"] - eps)
    )
    zero_rng = np.isclose(df["h"] - df["l"], 0.0, atol=1e-12)
    if "v" in df.columns:          # only count if there was volume
        zero_rng = zero_rng & (df["v"] > 0)

    if zero_rng.any():
        res.warnings["bars.zero_range"] = f"{int(zero_rng.sum())} zero-range candles."
        
    n_bad = int(out_of_bounds.sum())
    if n_bad > 0:
        res.errors["bars.shape"] = f"{n_bad} candles with invalid OHLC ranges."
    if zero_rng.any():
        res.warnings["bars.zero_range"] = f"{int(zero_rng.sum())} zero-range candles."
    if (n_bad == 0) and (not zero_rng.any()):
        res.info["bars.shape"] = "All candle ranges valid."
    return res

def _missing_runs(idx: pd.DatetimeIndex) -> List[List[pd.Timestamp]]:
    # Build complete minute grid and find missing timestamps; group consecutive runs
    full = pd.date_range(start=idx[0].floor("min"), end=idx[-1].floor("min"), freq="1min", tz="UTC")
    idx_min = idx.floor("min")
    missing = sorted(set(full) - set(idx_min))
    if not missing:
        return []
    runs: List[List[pd.Timestamp]] = []
    cur = [missing[0]]
    for ts in missing[1:]:
        if (ts - cur[-1]) == pd.Timedelta(minutes=1):
            cur.append(ts)
        else:
            runs.append(cur)
            cur = [ts]
    runs.append(cur)
    return runs

def check_gaps_smart(df: pd.DataFrame,
                     gap_tol_min: int,
                     recent_min: int,
                     pct_warn: float,
                     pct_fail: float) -> CheckResult:
    res = CheckResult()
    if df.empty:
        res.errors["gaps.empty"] = "No bars to assess gaps."
        return res

    idx = df.index.floor("min")
    full = pd.date_range(start=idx[0], end=idx[-1], freq="1min", tz="UTC")
    missing = sorted(set(full) - set(idx))
    if not missing:
        res.info["gaps"] = "No missing minutes."
        return res

    runs = _missing_runs(idx)
    longest = max((len(r) for r in runs), default=0)
    miss_pct = (len(missing) / max(len(full), 1)) * 100.0

    recent_threshold = idx[-1] - pd.Timedelta(minutes=recent_min)
    any_recent_long = any((len(r) > gap_tol_min) and (r[-1] >= recent_threshold) for r in runs)

    if miss_pct > pct_fail or any_recent_long:
        res.errors["gaps.long"] = f"Missing {miss_pct:.2f}% minutes; longest run {longest} min (recent long gap={any_recent_long})."
        return res

    if miss_pct > pct_warn or longest > gap_tol_min:
        res.warnings["gaps.short"] = f"Missing {miss_pct:.2f}% minutes; longest run {longest} min."
    else:
        res.info["gaps"] = f"Missing {miss_pct:.2f}% minutes; longest run {longest} min."
    return res

def check_freshness(trade_px: Optional[float],
                    trade_ts: Optional[datetime],
                    bars_last_ts: Optional[pd.Timestamp],
                    stale_sec: int,
                    bars_fresh_sec: int,
                    price_dec: int) -> CheckResult:
    res = CheckResult()
    now = utcnow()
    if bars_last_ts is None:
        res.errors["fresh.bars"] = "No bars timestamp."
        return res

    bars_age = (now - bars_last_ts.to_pydatetime()).total_seconds()
    bars_fresh = bars_age <= bars_fresh_sec

    if trade_ts is None or trade_px is None:
        if bars_fresh:
            res.warnings["fresh.trade"] = "Bars fresh but latest trade missing."
        else:
            res.errors["fresh.all"] = "Trade missing and bars stale."
        return res

    trade_age = (now - trade_ts).total_seconds()
    trade_fresh = trade_age <= stale_sec

    if not trade_fresh and bars_fresh:
        res.warnings["fresh.trade"] = f"Latest trade stale ({int(trade_age)}s) but bars are fresh ({int(bars_age)}s). px={round(trade_px, price_dec)}"
    elif not trade_fresh and not bars_fresh:
        res.errors["fresh.all"] = f"Trade stale ({int(trade_age)}s) and bars stale ({int(bars_age)}s)."
    else:
        res.info["fresh.ok"] = f"Trade fresh ({int(trade_age)}s), bars age {int(bars_age)}s."
    return res

def check_spread(bid: Optional[float],
                 ask: Optional[float],
                 cap_bps: float) -> CheckResult:
    res = CheckResult()
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        res.warnings["spread.na"] = "Quote missing or invalid; cannot assess spread."
        return res
    mid = 0.5 * (bid + ask)
    bps = ((ask - bid) / mid) * 10000.0
    if bps > cap_bps:
        res.errors["spread"] = f"Spread {bps:.1f} bps > cap {cap_bps:.1f}."
    else:
        res.info["spread"] = f"Spread {bps:.1f} bps <= cap {cap_bps:.1f}."
    return res

def check_jump(df: pd.DataFrame,
               warn_bps: float,
               fail_bps: float) -> CheckResult:
    res = CheckResult()
    if len(df) < 2:
        res.info["jump.na"] = "Not enough bars to assess last-bar jump."
        return res
    p0, p1 = float(df["c"].iloc[-2]), float(df["c"].iloc[-1])
    if p0 <= 0 or p1 <= 0:
        res.warnings["jump.badpx"] = "Non-positive prices encountered."
        return res
    bps = abs((p1 - p0) / ((p1 + p0) / 2.0)) * 10000.0
    if bps >= fail_bps:
        res.errors["jump"] = f"Last-bar move {bps:.0f} bps >= fail {fail_bps:.0f}."
    elif bps >= warn_bps:
        res.warnings["jump"] = f"Last-bar move {bps:.0f} bps >= warn {warn_bps:.0f}."
    else:
        res.info["jump"] = f"Last-bar move {bps:.0f} bps."
    return res


# ------------------------------ Config ------------------------------

def build_guard_cfg() -> GuardCfg:
    # Allow several env names for spread cap (choose first present)
    spread_cap = env_float_or_none("SPREAD_CAP_BPS")
    if spread_cap is None:
        candidates = [
            env_float_or_none("MAX_SPREAD_BPS"),
            env_float_or_none("MAX_SPREAD_BPS_RTH"),
            env_float_or_none("MAX_SPREAD_BPS_EXT"),
            env_float_or_none("MAX_SPREAD_BPS_OVN"),
        ]
        spread_cap = next((c for c in candidates if c is not None), 50.0)

    return GuardCfg(
        symbol=env_str("SYMBOL", "BTC/USD"),
        lookback_min=env_int("LOOKBACK_MIN", 240),
        gap_tol_min=env_int("GAP_TOL_MIN", 10),
        gap_recent_min=env_int("GAP_RECENT_MIN", 60),
        gap_pct_warn=env_float("GAP_PCT_WARN", 1.0),
        gap_pct_fail=env_float("GAP_PCT_FAIL", 3.0),
        stale_trade_sec=env_int("STALE_TRADE_SEC", 300),
        bars_fresh_sec=env_int("BARS_FRESH_SEC", 180),
        spread_cap_bps=float(spread_cap),
        jump_warn_bps=env_float("JUMP_WARN_BPS", 300.0),
        jump_fail_bps=env_float("JUMP_FAIL_BPS", 1500.0),
        price_dec=env_int("PRICE_DEC", 2),
        poll_sec=env_int("POLL_SEC", 30),
        one_shot=(env_int("ONE_SHOT", 0) == 1),
        strict=(env_int("STRICT", 0) == 1),
    )


# ------------------------------ Helpers ------------------------------

def merge(a: CheckResult, b: CheckResult) -> CheckResult:
    return CheckResult(
        errors={**a.errors, **b.errors},
        warnings={**a.warnings, **b.warnings},
        info={**a.info, **b.info},
    )

def summarize(results: List[CheckResult], strict: bool) -> Tuple[Dict[str, str], int]:
    agg = CheckResult()
    for r in results:
        agg = merge(agg, r)
    code = agg.worst_code(strict)
    # Prefer errors, then warnings, then info
    if agg.errors:
        headline = next(iter(agg.errors.values()))
        level = "ERROR"
    elif agg.warnings:
        headline = next(iter(agg.warnings.values()))
        level = "WARN"
    else:
        headline = next(iter(agg.info.values())) if agg.info else "OK"
        level = "OK"
    return {"level": level, "headline": headline}, code

def print_summary(symbol: str, summary: Dict[str, str]) -> None:
    ts = utcnow().isoformat(timespec="seconds")
    print(f"[{ts}] [{symbol}] {summary['level']}: {summary['headline']}")


# ------------------------------ Runner ------------------------------

def run_once(cli: CryptoHistoricalDataClient, cfg: GuardCfg) -> Tuple[Dict[str, str], int]:
    sym = symbol_for_data(cfg.symbol)

    # Fetch
    df = fetch_recent_bars(cli, sym, cfg.lookback_min)
    trade_px, trade_ts = fetch_latest_trade(cli, sym)
    bid, ask, _qts = fetch_latest_quote(cli, sym)

    # Checks
    res_shape = check_candle_shapes(df)
    res_gaps = check_gaps_smart(df, cfg.gap_tol_min, cfg.gap_recent_min, cfg.gap_pct_warn, cfg.gap_pct_fail)
    last_ts = df.index[-1] if not df.empty else None
    res_fresh = check_freshness(trade_px, trade_ts, last_ts, cfg.stale_trade_sec, cfg.bars_fresh_sec, cfg.price_dec)
    res_spread = check_spread(bid, ask, cfg.spread_cap_bps)
    res_jump = check_jump(df, cfg.jump_warn_bps, cfg.jump_fail_bps)

    summary, code = summarize([res_shape, res_gaps, res_fresh, res_spread, res_jump], strict=cfg.strict)
    return summary, code


def main():
    cfg = build_guard_cfg()
    print("=== Data Guard (Crypto, 24×7) ===")
    print(f"Symbol={cfg.symbol}  Lookback={cfg.lookback_min}m")
    print(f"Gaps tol={cfg.gap_tol_min}m  recent={cfg.gap_recent_min}m  warn%={cfg.gap_pct_warn}  fail%={cfg.gap_pct_fail}")
    print(f"Freshness: trade_stale>{cfg.stale_trade_sec}s  bars_fresh<={cfg.bars_fresh_sec}s")
    print(f"SpreadCap={cfg.spread_cap_bps:.0f} bps  Jump warn/fail={cfg.jump_warn_bps:.0f}/{cfg.jump_fail_bps:.0f} bps")
    print(f"Loop: poll={cfg.poll_sec}s  OneShot={cfg.one_shot}  Strict={cfg.strict}")
    print("------------------------------------------------------------")

    try:
        cli = mk_data_client()
    except Exception as e:
        print(f"[fatal] Cannot create data client: {e}")
        raise SystemExit(2)

    if cfg.one_shot:
        summary, code = run_once(cli, cfg)
        print_summary(cfg.symbol, summary)
        raise SystemExit(code)

    # loop
    while True:
        start = time.time()
        try:
            summary, code = run_once(cli, cfg)
            print_summary(cfg.symbol, summary)
            if code == 2 and cfg.strict:
                # still continue looping, but reflect severity in logs/exit only on one-shot
                pass
        except Exception as e:
            print(f"[loop-error] {e}")
        elapsed = time.time() - start
        time.sleep(max(0.5, cfg.poll_sec - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye 👋")
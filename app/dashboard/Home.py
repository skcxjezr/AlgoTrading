# app/dashboard/Home.py
# ------------------------------------------------------------
# Crypto Trader Dashboard (Fixed Portfolio + Backtests Runner)
# - Portfolio prefers live broker (Alpaca) positions; falls back to DB.
# - Rolling PV uses Alpaca Portfolio History (equity incl. cash) when available.
# - Backtests tab lets you choose strategy/params and run a backtest.
# - Each run writes meta_*.json so strategy/params show in the UI.
# - Robust import of part1_backtest from sibling app/ folder.
# - Fallback file saving when part1_backtest.save_outputs is missing.
# - UTC-safe timestamps; resilient DB schema detection; Streamlit caching.
# ------------------------------------------------------------

from __future__ import annotations
import time
import os
import sys
import math
import json
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path as _Path
import altair as alt

import pandas as pd
import numpy as np
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
except Exception:
    _st_autorefresh = None

def safe_autorefresh(interval=None, *, key=None, interval_ms=None, **kwargs):
    """
    No-op while a backtest is running.
    Accepts either `interval` or `interval_ms` (and positional works too).
    Forwards any other st_autorefresh kwargs (e.g., limit).
    """
    # normalize the interval param
    if interval_ms is None:
        interval_ms = interval
    if not st.session_state.get("bt_running", False):
        if _st_autorefresh and interval_ms:
            _st_autorefresh(interval=interval_ms, key=key, **kwargs)
# --- optional TradingView Lightweight Charts ---
try:
    from streamlit_lightweight_charts import renderLightweightCharts
    _HAS_LWC = True
except Exception:
    _HAS_LWC = False
    
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy import inspect
import re

# ------------------------------ Optional broker SDK ------------------------------
try:
    from alpaca.trading.client import TradingClient
    from alpaca.common.exceptions import APIError
except Exception:  # pragma: no cover
    TradingClient = None
    class APIError(Exception):
        pass

# ------------------------------ Robust import of backtester ------------------------------
bt = None
try:
    import part1_backtest as bt
except Exception:
    try:
        from app import part1_backtest as bt
    except Exception:
        try:
            _self_dir = _Path(__file__).resolve().parent       # app/dashboard
            _app_dir = _self_dir.parent                        # app
            if str(_app_dir) not in sys.path:
                sys.path.insert(0, str(_app_dir))
            import part1_backtest as bt
        except Exception:
            bt = None

# ------------------------------ Utils ------------------------------

st.set_page_config(page_title="Crypto Trader Dashboard", layout="wide")

def _env_str(k: str, default: str = "") -> str:
    v = os.getenv(k, default)
    return v if isinstance(v, str) else default

def _env_bool(k: str, default: bool = False) -> bool:
    v = str(os.getenv(k, str(int(default)))).strip().lower()
    return v in ("1","true","t","yes","y")

def _env_bps(k: str, default: float = 0.0) -> float:
    """
    Read an env var as *basis points* (bps).
    Accepts values like: "0.1", "0.1  # comment", "10bps", "0.10%", "12 bp".
    - If '%' is present, interpret as percent and convert to bps (e.g., 0.10% -> 10 bps).
    - If 'bp'/'bps' present, interpret as bps directly.
    - Otherwise, treat the raw number as bps.
    """
    v = os.getenv(k, "")
    s = str(v or "")

    # strip inline comments
    s = s.split("#", 1)[0]
    s = s.split("//", 1)[0]
    s = s.strip().lower().replace(",", "")

    if not s:
        return default

    import re
    m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
    if not m:
        return default
    x = float(m.group(1))

    if "%" in s:
        return x * 100.0  # percent -> bps
    # if 'bp' or 'bps' present, x is already in bps
    return x

def _env_float(k: str, default: float = 0.0) -> float:
    """
    Parse a float from an env var, ignoring inline comments and symbols like $ or commas.
    Accepts: "5", "5.0", "$5", "5  # comment", "5e3", etc.
    """
    s = os.getenv(k, "")
    s = str(s if s is not None else "")
    # strip inline comments
    s = s.split("#", 1)[0]
    s = s.split("//", 1)[0]
    # clean adornments
    s = s.strip().lower().replace(",", "").replace("$", "")
    if not s:
        return default
    import re
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    try:
        return float(m.group(0)) if m else default
    except Exception:
        return default

def _env_int(k: str, default: int = 0) -> int:
    """Parse an int from an env var (uses _env_float then rounds)."""
    v = _env_float(k, float(default))
    try:
        return int(round(v))
    except Exception:
        return default
    
def to_utc(s: pd.Series | pd.DatetimeIndex | pd.Timestamp | str) -> pd.Series:
    dtv = pd.to_datetime(s, utc=True, errors="coerce")
    return dtv

def _ensure_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None or t.tz is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")

def _as_utc_timestamp(x) -> pd.Timestamp:
    """Return a UTC-aware pandas.Timestamp from any input without using tz= on aware inputs."""
    ts = pd.Timestamp(x)
    if ts.tzinfo is None or ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def _to_int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return int(x)
    except Exception:
        return None

def _time_window_controls(default: str = "3M"):
    presets = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "Max", "Custom"]
    c1, c2 = st.columns([2, 4])

    try:
        choice = c1.segmented_control("Window", presets, default=default if default in presets else "3M")
    except Exception:
        choice = c1.selectbox("Window", presets, index=presets.index(default if default in presets else "3M"))

    end_utc = _ensure_utc(pd.Timestamp.utcnow())

    if choice == "1D":
        start_utc = end_utc - pd.Timedelta(days=1);  days = 1
    elif choice == "1W":
        start_utc = end_utc - pd.Timedelta(days=7);  days = 7
    elif choice == "1M":
        start_utc = end_utc - pd.Timedelta(days=30); days = 30
    elif choice == "3M":
        start_utc = end_utc - pd.Timedelta(days=90); days = 90
    elif choice == "6M":
        start_utc = end_utc - pd.Timedelta(days=180); days = 180
    elif choice == "1Y":
        start_utc = end_utc - pd.Timedelta(days=365); days = 365
    elif choice == "YTD":
        jan1 = _ensure_utc(pd.Timestamp(end_utc.year, 1, 1))
        start_utc = jan1; days = max((end_utc - start_utc).days, 1)
    elif choice == "Max":
        start_utc = None; days = 1825
    else:
        default_start = (end_utc - pd.Timedelta(days=90)).date()
        default_end   = end_utc.date()
        dr = c2.date_input("Date range (UTC)", (default_start, default_end))
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            s, e = dr
        else:
            s, e = default_start, default_end

        s = _ensure_utc(pd.Timestamp(s))
        e = _ensure_utc(pd.Timestamp(e)) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        if e < s:
            s, e = e, s
        start_utc, end_utc = s, e
        days = max((end_utc - start_utc).days, 1)

    c3, _ = st.columns([2, 3])
    normalize = c3.checkbox("Normalize to 100", value=False,
                            help="Rebase series so the first visible point = 100.")
    smooth = c3.checkbox("7-day smooth", value=False,
                         help="Show a 7-day rolling mean overlay.")
    return start_utc, end_utc, days, normalize, smooth

def _ensure_ts_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean 2-col DataFrame ['ts','close'] with UTC datetimes and numeric close.
    Robust to MultiIndex, duplicate columns, and object/dict/list cell values.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ts","close"])

    d = df.copy()

    # Flatten MultiIndex columns -> single strings
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = ["__".join([str(x) for x in tup if x is not None]) for tup in d.columns]

    # If row index is MultiIndex, flatten it
    if isinstance(d.index, pd.MultiIndex):
        d = d.reset_index()

    # Ensure there's a 'ts' column
    if "ts" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={"index": "ts"})
        else:
            for cand in ["timestamp", "time", "Datetime", "Date", "date"]:
                if cand in d.columns:
                    d = d.rename(columns={cand: "ts"})
                    break
    if "ts" not in d.columns:
        # Try to detect a datetime-like column
        for c in list(d.columns):
            try:
                probe = pd.to_datetime(d[c], utc=True, errors="coerce")
                if probe.notna().sum() >= max(3, int(len(d) * 0.1)):
                    d.insert(0, "ts", probe)
                    break
            except Exception:
                pass

    # Choose a single 'close' series (handle duplicates and MultiIndex columns)
    close_series = None
    for cand in ["close", "Close", "Adj Close", "adj_close", "c", "price"]:
        if cand in d.columns:
            obj = d[cand]
            # If duplicate columns produced a DataFrame, pick the one with most non-nulls
            if isinstance(obj, pd.DataFrame):
                counts = obj.notna().sum()
                obj = obj[counts.idxmax()]
            close_series = obj
            break
    if close_series is None:
        # Case-insensitive fallback
        for c in d.columns:
            if str(c).lower() == "close":
                obj = d[c]
                if isinstance(obj, pd.DataFrame):
                    counts = obj.notna().sum()
                    obj = obj[counts.idxmax()]
                close_series = obj
                break

    # If still missing required pieces, bail out cleanly
    if "ts" not in d.columns or close_series is None:
        return pd.DataFrame(columns=["ts","close"])

    # Convert types
    ts = pd.to_datetime(d["ts"], utc=True, errors="coerce")

    # Squeeze any remaining 2D into a 1D Series
    if isinstance(close_series, pd.DataFrame):
        if close_series.shape[1] == 1:
            close_series = close_series.iloc[:, 0]
        else:
            counts = close_series.notna().sum()
            close_series = close_series[counts.idxmax()]

    # If cells are dict/list, extract a scalar
    if getattr(close_series, "dtype", None) == "object":
        def _extract(x):
            if isinstance(x, dict):
                for k in ("close", "c", "price"):
                    if k in x:
                        return x[k]
            if isinstance(x, (list, tuple)) and len(x) > 0:
                return x[-1]  # last value
            return x
        close_series = close_series.map(_extract)

    close_vals = pd.to_numeric(close_series, errors="coerce")

    out = pd.DataFrame({"ts": ts, "close": close_vals}).dropna(subset=["ts", "close"])
    return out


# ------------------------------ Database ------------------------------

def _dsn_from_env() -> str:
    dsn = os.getenv("DB_DSN") or os.getenv("DATABASE_URL") or ""
    if dsn:
        return dsn
    user = _env_str("DB_USER", "postgres")
    pwd  = _env_str("DB_PASSWORD", "postgres")
    host = _env_str("DB_HOST", "db")
    port = _env_str("DB_PORT", "5432")
    name = _env_str("DB_NAME", "trading")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"

@st.cache_resource(show_spinner=False)
def get_engine() -> Optional[Engine]:
    try:
        dsn = _dsn_from_env()
        eng = create_engine(dsn, pool_pre_ping=True, pool_recycle=180)
        with eng.connect() as c:
            c.exec_driver_sql("SELECT 1")
        return eng
    except Exception as e:
        st.warning(f"DB unavailable: {e}")
        return None

def table_exists(engine: Optional[Engine], table: str) -> bool:
    if engine is None:
        return False
    try:
        insp = inspect(engine)
        return insp.has_table(table, schema="public") or insp.has_table(table)
    except Exception:
        return False

def columns_of(engine: Engine, table: str) -> List[str]:
    try:
        insp = inspect(engine)
        cols = [c["name"] for c in insp.get_columns(table, schema="public")] or \
               [c["name"] for c in insp.get_columns(table)]
        return [str(c) for c in cols]
    except Exception:
        return []

def first_existing(candidates: List[str], existing: List[str]) -> Optional[str]:
    e = [c.lower() for c in existing]
    for name in candidates:
        if name.lower() in e:
            for ex in existing:
                if ex.lower() == name.lower():
                    return ex
    return None

# ------------------------------ Alpaca (Broker) ------------------------------

def _alpaca_client() -> Optional[Any]:
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or ""
    sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET") or ""
    if not key or not sec or TradingClient is None:
        return None
    base_url = os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_BASE_URL") or ""
    paper = "paper" in (base_url or "").lower() or _env_bool("ALPACA_PAPER", True)
    try:
        return TradingClient(key, sec, paper=paper)
    except Exception:
        return None

def _alpaca_headers() -> Dict[str, str]:
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        return {}
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}

def _detect_alpaca_base_url() -> Optional[str]:
    try:
        import requests
    except Exception:
        return None
    headers = _alpaca_headers()
    if not headers:
        return None
    base_env = (os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_BASE_URL") or "").strip()
    candidates = [base_env] if base_env else [
        "https://paper-api.alpaca.markets",
        "https://api.alpaca.markets",
    ]
    for url in [c for c in candidates if c]:
        u = url.rstrip("/")
        try:
            r = requests.get(f"{u}/v2/account", headers=headers, timeout=20)
            if r.status_code == 200:
                return u
        except Exception:
            continue
    return None

def _period_from_days(days: int) -> str:
    d = int(max(1, days))
    if d <= 1:   return "1D"
    if d <= 30:  return "1M"
    if d <= 90:  return "3M"
    if d <= 180: return "6M"
    if d <= 365: return "1A"
    return "all"

def _get_portfolio_history_rest(
    period: Optional[str] = None,
    timeframe: str = "1D",
    extended_hours: bool = True,
) -> pd.DataFrame:
    try:
        import requests
    except Exception:
        return pd.DataFrame(columns=["ts","portfolio_value"])
    headers = _alpaca_headers()
    base = _detect_alpaca_base_url()
    if not headers or not base:
        return pd.DataFrame(columns=["ts","portfolio_value"])
    url = f"{base}/v2/account/portfolio/history"
    params = {"timeframe": timeframe, "extended_hours": str(bool(extended_hours)).lower()}
    if period:
        params["period"] = period
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            try:
                err = r.json()
            except Exception:
                err = {"raw": r.text}
            st.warning(f"Alpaca portfolio history failed ({r.status_code}): {err}")
            return pd.DataFrame(columns=["ts","portfolio_value"])
        data = r.json() or {}
        ts = pd.to_datetime(data.get("timestamp", []), unit="s", utc=True)
        eq = pd.Series(data.get("equity", []), index=ts, name="portfolio_value")
        df = pd.DataFrame({"ts": eq.index, "portfolio_value": pd.to_numeric(eq.values, errors="coerce")})
        df = df.dropna(subset=["ts", "portfolio_value"]).sort_values("ts").drop_duplicates("ts", keep="last")
        return df[["ts","portfolio_value"]]
    except Exception:
        return pd.DataFrame(columns=["ts","portfolio_value"])

def load_positions_broker() -> pd.DataFrame:
    tc = _alpaca_client()
    if tc is None:
        return pd.DataFrame()
    try:
        positions = tc.get_all_positions()
    except APIError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if not positions:
        return pd.DataFrame()
    rows = []
    for p in positions:
        def f(x):
            try: return float(x)
            except Exception: return np.nan
        sym = (getattr(p, "symbol", "") or "").replace("/", "")
        qty = f(getattr(p, "qty", "0"))
        avg = f(getattr(p, "avg_entry_price", np.nan))
        mv  = f(getattr(p, "market_value", np.nan))
        upl = f(getattr(p, "unrealized_pl", np.nan))
        uplpc = f(getattr(p, "unrealized_plpc", np.nan))
        last = f(getattr(p, "current_price", np.nan))
        if (math.isnan(last) or last == 0) and (not math.isnan(mv)) and qty not in (0, np.nan):
            last = abs(mv) / abs(qty)
        rows.append(dict(symbol=sym, qty=qty, avg_cost=avg, last=last,
                         market_value=mv, unrealized_pl=upl, unrealized_plpc=uplpc))
    df = pd.DataFrame(rows)
    for c in ("qty","avg_cost","last","market_value","unrealized_pl","unrealized_plpc"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_account_broker() -> Dict[str, Optional[float]]:
    tc = _alpaca_client()
    if tc is None:
        return dict(cash=None, equity=None, portfolio_value=None, buying_power=None)
    try:
        acct = tc.get_account()
        def f(x):
            try: return float(x)
            except Exception: return None
        return dict(
            cash=f(getattr(acct, "cash", None)),
            equity=f(getattr(acct, "equity", None)),
            portfolio_value=f(getattr(acct, "portfolio_value", None)),
            buying_power=f(getattr(acct, "buying_power", None)),
        )
    except Exception:
        return dict(cash=None, equity=None, portfolio_value=None, buying_power=None)

@st.cache_data(ttl=30, show_spinner=False)
def load_portfolio_history_broker(days: int = 30) -> pd.DataFrame:
    period = _period_from_days(days)
    timeframe = "1Min" if period == "1D" else "1D"

    df_rest = _get_portfolio_history_rest(period=period, timeframe=timeframe, extended_hours=True)
    if not df_rest.empty:
        return df_rest

    tc = _alpaca_client()
    if tc is None:
        return pd.DataFrame(columns=["ts","portfolio_value"])

    end = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    start = end - dt.timedelta(days=max(1, days))

    def _df_from_payload(ph) -> pd.DataFrame:
        if ph is None:
            return pd.DataFrame()
        try:
            eq = getattr(ph, "equity", None)
            ts = getattr(ph, "timestamp", None)
            if eq is None and isinstance(ph, dict):
                eq = ph.get("equity"); ts = ph.get("timestamp") or ph.get("time")
            if eq is None:
                phd = dict(ph)
                eq = phd.get("equity"); ts = phd.get("timestamp") or phd.get("time")
            if not eq or ts is None:
                return pd.DataFrame()
            df = pd.DataFrame({
                "ts": pd.to_datetime(ts, unit="s", utc=True),
                "portfolio_value": pd.to_numeric(eq, errors="coerce")
            }).dropna()
            return df
        except Exception:
            return pd.DataFrame()

    try:
        ph = tc.get_portfolio_history(period=period, timeframe=timeframe, extended_hours=True)
        df = _df_from_payload(ph)
        if not df.empty:
            return df
    except Exception:
        pass

    try:
        ph = tc.get_portfolio_history(date_start=start, date_end=end, timeframe=timeframe, extended_hours=True)
        df = _df_from_payload(ph)
        if not df.empty:
            return df
    except Exception:
        pass

    return pd.DataFrame(columns=["ts","portfolio_value"])

# ------------------------------ DB: Fills → FIFO positions ------------------------------
@st.cache_data(ttl=20, show_spinner=False)
def load_fills_db(
    _engine: Optional[Engine],
    symbol_hint: Optional[str] = None,
    hours: int = 24,
    max_pages: int = 10
) -> pd.DataFrame:
    try:
        import requests
        key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
        if key and sec:
            base_url = (os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
            url = f"{base_url}/v2/account/activities"
            headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
            since = (dt.datetime.utcnow() - dt.timedelta(hours=max(1, int(hours)))).replace(microsecond=0).isoformat() + "Z"
            params = {"activity_types": "FILL", "after": since, "direction": "desc", "page_size": 100}
            results, page_token, pages = [], None, 0
            while True:
                p = params.copy()
                if page_token:
                    p["page_token"] = page_token
                r = requests.get(url, headers=headers, params=p, timeout=30)
                r.raise_for_status()
                batch = r.json()
                if not isinstance(batch, list) or not batch:
                    break
                results.extend(batch)
                page_token = batch[-1].get("id")
                pages += 1
                if pages >= max_pages or not page_token:
                    break
            if results:
                rows = []
                for a in results:
                    if a.get("activity_type") != "FILL":
                        continue
                    rows.append({
                        "ts": a.get("transaction_time"),
                        "symbol": (a.get("symbol") or "").replace("/", ""),
                        "side": (a.get("side") or "").lower() or None,
                        "qty": float(a["qty"]) if a.get("qty") not in (None, "") else np.nan,
                        "price": float(a["price"]) if a.get("price") not in (None, "") else np.nan,
                        "venue": a.get("exchange") or a.get("venue") or None,
                        "liquidity": a.get("liquidity") or None,
                        "order_id": a.get("order_id") or a.get("id") or None,
                    })
                df_api = pd.DataFrame(rows)
                if not df_api.empty:
                    df_api["ts"] = pd.to_datetime(df_api["ts"], utc=True, errors="coerce")
                    df_api = df_api.dropna(subset=["ts"]).sort_values("ts", ascending=False).reset_index(drop=True)
                    df_api["qty"] = pd.to_numeric(df_api["qty"], errors="coerce")
                    df_api["price"] = pd.to_numeric(df_api["price"], errors="coerce")
                    if symbol_hint:
                        sym_clean = symbol_hint.replace("/", "")
                        df_api = df_api[df_api["symbol"].astype(str) == sym_clean]
                    for col in ["venue","liquidity","order_id"]:
                        if col not in df_api.columns:
                            df_api[col] = pd.NA
                    st.caption("Fills source → Alpaca REST (recent)")
                    return df_api[["ts","symbol","side","qty","price","venue","liquidity","order_id"]]
    except Exception:
        pass

    if _engine is None:
        return pd.DataFrame(columns=["ts","symbol","side","qty","price","venue","liquidity","order_id"])

    insp = inspect(_engine)

    def qident(name: Optional[str]) -> Optional[str]:
        if not name: return None
        return '"' + str(name).replace('"','""') + '"'

    def find_schema(table: str) -> Optional[str]:
        if insp.has_table(table, schema="public"):
            return "public"
        if insp.has_table(table):
            return None
        return None

    candidates = ["fills", "executions", "trade_fills", "orders_filled"]

    for table in candidates:
        schema = find_schema(table)
        if schema is None and not insp.has_table(table):
            continue

        cols = columns_of(_engine, table)
        ts_col   = first_existing(["ts","timestamp","time","filled_at","executed_at","created_at"], cols)
        sym_col  = first_existing(["symbol","asset","ticker"], cols)
        side_col = first_existing(["side","buy_sell","direction","is_buy"], cols)
        qty_col  = first_existing(["qty","quantity","filled_qty","size"], cols)
        px_col   = first_existing(["price","fill_price","avg_price","executed_price"], cols)
        ven_col  = first_existing(["venue","exchange","source","broker"], cols)
        liq_col  = first_existing(["liquidity","liquidity_flag","taker"], cols)
        oid_col  = first_existing(["order_id","client_order_id","clordid","orderid"], cols)

        if not (ts_col and sym_col and qty_col and px_col):
            continue

        try:
            select_cols = [
                f"{qident(ts_col)}   AS ts",
                f"{qident(sym_col)}  AS symbol",
                (f"{qident(side_col)} AS side") if side_col else "NULL AS side",
                f"{qident(qty_col)}  AS qty",
                f"{qident(px_col)}   AS price",
                (f"{qident(ven_col)} AS venue") if ven_col else "NULL AS venue",
                (f"{qident(liq_col)} AS liquidity") if liq_col else "NULL AS liquidity",
                (f"{qident(oid_col)} AS order_id") if oid_col else "NULL AS order_id",
            ]

            where_clauses = []
            params = {}
            if symbol_hint:
                sym_clean = symbol_hint.replace("/", "")
                where_clauses.append(f"(REPLACE({qident(sym_col)}, '/', '') = :sym OR {qident(sym_col)} = :sym_slash)")
                params.update({"sym": sym_clean, "sym_slash": symbol_hint})

            if hours and int(hours) > 0:
                where_clauses.append(f"{qident(ts_col)} >= NOW() - INTERVAL '{int(hours)} hours'")

            where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            from_clause = (f'"{schema}".{qident(table)}' if schema else qident(table))
            q = text(f"""
                SELECT {', '.join(select_cols)}
                FROM {from_clause}
                {where}
                ORDER BY ts DESC
                LIMIT 10000
            """)

            with _engine.connect() as c:
                df = pd.read_sql(q, c, params=params)

            if df.empty:
                continue

            df["ts"] = to_utc(df["ts"])
            df = df.dropna(subset=["ts"])
            df["symbol"] = df["symbol"].astype(str).str.replace("/", "", regex=False)

            side = df["side"].astype(str).str.strip().str.lower()
            bool_map = {"true":"buy","t":"buy","1":"buy","false":"sell","f":"sell","0":"sell"}
            side = side.map(lambda s: bool_map.get(s, s))
            df["side"] = side.replace({
                "buy_long":"buy","b":"buy","bot":"buy","long":"buy",
                "sell_short":"sell","s":"sell","sold":"sell","short":"sell"
            })
            if side_col is None:
                if "qty" in df:
                    qnum = pd.to_numeric(df["qty"], errors="coerce")
                    df.loc[qnum < 0, "side"] = "sell"
                    df.loc[qnum >= 0, "side"] = df.loc[qnum >= 0, "side"].fillna("buy")

            for col in ["qty","price"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            for opt in ["venue","liquidity","order_id"]:
                if opt not in df.columns:
                    df[opt] = pd.NA

            st.caption(f"Fills source → table: `{table}` schema: `{schema or 'search_path'}` rows: {len(df)}")
            return df[["ts","symbol","side","qty","price","venue","liquidity","order_id"]]

        except Exception:
            continue

    return pd.DataFrame(columns=["ts","symbol","side","qty","price","venue","liquidity","order_id"])

@dataclass
class Lot:
    qty: float
    price: float

def _fifo_cost_basis(fills: pd.DataFrame) -> pd.DataFrame:
    if fills is None or fills.empty:
        return pd.DataFrame(columns=["symbol","qty","avg_cost"])
    f = fills[["ts","symbol","side","qty","price"]].copy()
    f = f.sort_values("ts")
    out: Dict[str, Dict[str, float]] = {}

    for sym, g in f.groupby("symbol"):
        lots: List[Lot] = []
        for _, row in g.iterrows():
            side = str(row["side"]).lower()
            qty  = float(row["qty"])
            px   = float(row["price"])
            if side.startswith("buy"):
                i = 0
                while qty > 0 and i < len(lots) and lots[i].qty < 0:
                    cover = min(qty, -lots[i].qty)
                    lots[i].qty += cover
                    qty -= cover
                    if abs(lots[i].qty) < 1e-12:
                        lots.pop(i)
                    else:
                        i += 1
                if qty > 0:
                    lots.append(Lot(qty=qty, price=px))
            elif side.startswith("sell"):
                i = 0
                while qty > 0 and i < len(lots) and lots[i].qty > 0:
                    close = min(qty, lots[i].qty)
                    lots[i].qty -= close
                    qty -= close
                    if abs(lots[i].qty) < 1e-12:
                        lots.pop(i)
                    else:
                        i += 1
                if qty > 0:
                    lots.append(Lot(qty=-qty, price=px))
            else:
                continue

        net_qty = sum(l.qty for l in lots)
        if abs(net_qty) < 1e-12:
            continue
        total_cost = sum(abs(l.qty) * l.price for l in lots)
        avg_cost = total_cost / abs(net_qty)
        out[sym] = dict(symbol=sym, qty=net_qty, avg_cost=avg_cost)

    if not out:
        return pd.DataFrame(columns=["symbol","qty","avg_cost"])
    df = pd.DataFrame(out.values())
    return df.sort_values("symbol")

@st.cache_data(ttl=15, show_spinner=False)
def load_last_prices_db(_engine: Optional[Engine]) -> pd.DataFrame:
    if _engine is None:
        return pd.DataFrame(columns=["symbol","last","ts"])

    def _try(table: str, price_col_candidates: List[str]) -> pd.DataFrame:
        if not table_exists(_engine, table):
            return pd.DataFrame()
        cols = columns_of(_engine, table)
        ts_col = first_existing(["ts","timestamp","time","dt"], cols)
        sym_col = first_existing(["symbol","asset","ticker"], cols)
        px_col = first_existing(price_col_candidates, cols)
        if table.startswith("quotes"):
            bid_col = first_existing(["bid","bid_price","best_bid_price"], cols)
            ask_col = first_existing(["ask","ask_price","best_ask_price"], cols)
            if bid_col and ask_col:
                px_col = None
        if not (ts_col and sym_col):
            return pd.DataFrame()
        q = text(f"SELECT * FROM public.{table} WHERE {ts_col} >= NOW() - INTERVAL '1 day'")
        try:
            with _engine.connect() as c:
                df = pd.read_sql(q, c)
            if df.empty:
                return pd.DataFrame()
            df["ts"] = to_utc(df[ts_col])
            df["symbol"] = df[sym_col].astype(str).str.replace("/", "", regex=False)
            if px_col:
                df["last"] = pd.to_numeric(df[px_col], errors="coerce")
            else:
                bid_col2 = first_existing(["bid","bid_price","best_bid_price"], df.columns.tolist())
                ask_col2 = first_existing(["ask","ask_price","best_ask_price"], df.columns.tolist())
                if bid_col2 and ask_col2:
                    df["last"] = (pd.to_numeric(df[bid_col2], errors="coerce") + pd.to_numeric(df[ask_col2], errors="coerce")) / 2.0
                else:
                    return pd.DataFrame()
            return df[["symbol","last","ts"]].dropna()
        except Exception:
            return pd.DataFrame()

    for table, cand in [
        ("last_prices", ["last","price","close"]),
        ("quotes", ["mid","price"]),
        ("bars_1m", ["close","last"]),
        ("bars_5m", ["close","last"]),
        ("bars_15m", ["close","last"]),
        ("bars", ["close","last"]),
    ]:
        df = _try(table, cand)
        if not df.empty:
            df = df.sort_values("ts").drop_duplicates("symbol", keep="last")
            return df
    return pd.DataFrame(columns=["symbol","last","ts"])

@st.cache_data(ttl=15, show_spinner=False)
def load_equity_db(_engine: Optional[Engine]) -> pd.DataFrame:
    if _engine is None:
        return pd.DataFrame(columns=["ts","portfolio_value"])

    total_pv_candidates = [
        "portfolio_value", "net_liquidation", "nav", "account_value",
        "equity", "total_equity", "account_equity", "equity_total",
    ]
    positions_only_candidates = [
        "positions_equity", "securities_equity", "net_value", "net_position_value", "positions_mv",
    ]
    cash_candidates = [
        "cash", "cash_balance", "available_funds", "free_cash", "available_cash"
    ]
    tables = [
        "portfolio_values", "account_snapshots", "pnl_equity", "account_equity", "equity_curve"
    ]

    for table in tables:
        if not table_exists(_engine, table):
            continue
        cols = columns_of(_engine, table)
        if not cols:
            continue
        ts_col  = first_existing(["ts","timestamp","time","dt"], cols)
        pv_col  = first_existing(total_pv_candidates, cols)
        pos_col = first_existing(positions_only_candidates, cols)
        cash_col= first_existing(cash_candidates, cols)
        if not ts_col:
            continue

        select_cols = [f"{ts_col} AS ts"]
        if pv_col:  select_cols.append(f"{pv_col} AS pv_total")
        if pos_col: select_cols.append(f"{pos_col} AS pv_positions")
        if cash_col:select_cols.append(f"{cash_col} AS cash")

        try:
            q = text(f"SELECT {', '.join(select_cols)} FROM public.{table} ORDER BY 1 DESC LIMIT 5000")
            with _engine.connect() as c:
                df = pd.read_sql(q, c)
            if df.empty:
                continue
            df["ts"] = to_utc(df["ts"])
            for cnum in [c for c in ["pv_total","pv_positions","cash"] if c in df.columns]:
                df[cnum] = pd.to_numeric(df[cnum], errors="coerce")

            if "pv_total" in df.columns and df["pv_total"].notna().any():
                df["portfolio_value"] = df["pv_total"]
            elif "pv_positions" in df.columns and df["pv_positions"].notna().any():
                if "cash" in df.columns and df["cash"].notna().any():
                    df["portfolio_value"] = df["pv_positions"].fillna(0) + df["cash"].fillna(0)
                else:
                    df["portfolio_value"] = df["pv_positions"]
            else:
                continue

            out = df.dropna(subset=["ts","portfolio_value"]).sort_values("ts")
            if out.empty:
                continue
            out = out.drop_duplicates("ts")
            out["portfolio_value"] = pd.to_numeric(out["portfolio_value"], errors="coerce")
            out.loc[out["portfolio_value"] <= 0, "portfolio_value"] = np.nan
            out["portfolio_value"] = out["portfolio_value"].ffill()
            out = out.dropna(subset=["portfolio_value"])
            return out[["ts","portfolio_value"]]
        except Exception:
            continue

    return pd.DataFrame(columns=["ts","portfolio_value"])

# ------------------------------ Portfolio Rendering ------------------------------

def _fmt_money(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        return f"${x:,.2f}"
    except Exception:
        return "—"

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"

def _portfolio_summary(pos: pd.DataFrame,
                       account_equity: Optional[float],
                       account_cash: Optional[float]) -> Dict[str, Optional[float]]:
    if pos is None or pos.empty:
        cash_val = account_cash if account_cash is not None else (account_equity if account_equity is not None else None)
        pv_total = 0.0 + (cash_val or 0.0)
        lev = None if (account_equity in (None, 0)) else 0.0
        return dict(portfolio_value=pv_total, cash=cash_val, long_exposure=0.0, short_exposure=0.0,
                    unrealized_pl=0.0, leverage=lev, positions=0)

    pos = pos.copy()
    if "symbol" in pos.columns:
        non_cash_mask = ~pos["symbol"].astype(str).str.upper().eq("CASH")
    else:
        non_cash_mask = pd.Series(True, index=pos.index)

    if "position_value" not in pos.columns or pos["position_value"].isna().all():
        last = pd.to_numeric(pos.get("last"), errors="coerce")
        qty  = pd.to_numeric(pos.get("qty"), errors="coerce")
        pos["position_value"] = (qty.abs() * last.abs())

    mv_all = pd.to_numeric(pos.get("position_value"), errors="coerce").fillna(0.0)
    mv_nc = mv_all[non_cash_mask]

    cash_val = account_cash
    if cash_val is None:
        if "symbol" in pos.columns:
            cash_rows = pos[~non_cash_mask]
            if not cash_rows.empty:
                cash_val = float(pd.to_numeric(cash_rows.get("position_value"), errors="coerce").fillna(0.0).sum())
    if cash_val is None and account_equity is not None:
        cash_val = float(account_equity - float(mv_nc.sum()))

    pv_total = float(mv_nc.sum()) + (cash_val or 0.0)

    uplift = pd.to_numeric(pos.get("unrealized_pl"), errors="coerce").fillna(0.0)
    upl_nc = uplift[non_cash_mask] if len(uplift) == len(non_cash_mask) else uplift

    long_exp = float(mv_nc[mv_nc > 0].sum())
    short_exp = float(-mv_nc[mv_nc < 0].sum())
    upl_sum = float(upl_nc.sum())
    positions = int((mv_nc != 0).sum())

    lev = None
    if account_equity and account_equity != 0:
        lev = (long_exp + short_exp) / account_equity

    return dict(portfolio_value=pv_total, cash=cash_val, long_exposure=long_exp, short_exposure=short_exp,
                unrealized_pl=upl_sum, leverage=lev, positions=positions)

def render_portfolio(engine: Optional[Engine]) -> None:
    st.subheader("Portfolio")
    src = "broker"

    broker_pos = load_positions_broker()
    acct = load_account_broker()

    df = pd.DataFrame()
    if not broker_pos.empty:
        df = broker_pos.copy()
        if "last" not in df or df["last"].isna().all():
            df["last"] = np.nan
        df["position_value"] = (pd.to_numeric(df["qty"], errors="coerce").abs() *
                                 pd.to_numeric(df["last"], errors="coerce").abs()).round(2)
    else:
        src = "database (FIFO from fills)"
        pos_df = pd.DataFrame()
        if engine is not None and table_exists(engine, "positions"):
            try:
                with engine.connect() as c:
                    pos_df = pd.read_sql(text("SELECT * FROM public.positions ORDER BY 1 DESC LIMIT 10000"), c)
            except Exception:
                pos_df = pd.DataFrame()

        if pos_df.empty:
            fills = load_fills_db(engine)
            pos_df = _fifo_cost_basis(fills)

        if not pos_df.empty:
                    # --- De-dupe & drop dust positions ---
            # Keep only the most recent row per symbol (and account, if present)
            ts_candidates = [c for c in ["ts","timestamp","time","dt","asof","created_at","updated_at"] if c in pos_df.columns]
            ts_col = ts_candidates[0] if ts_candidates else None
            key_cols = ["symbol"] + (["account"] if "account" in pos_df.columns else [])
            if ts_col:
                pos_df = (pos_df.sort_values(ts_col).drop_duplicates(key_cols, keep="last"))

            # Drop near-zero qty (dust) rows
            pos_df["qty"] = pd.to_numeric(pos_df.get("qty", pd.Series(np.nan, index=pos_df.index)), errors="coerce").fillna(0.0)
            EPS = 1e-6
            dust_mask = pos_df["qty"].abs() < EPS
            for _col in ["unrealized_pl","market_value","position_value","avg_cost","last"]:
                if _col in pos_df.columns:
                    pos_df.loc[dust_mask, _col] = 0.0
            pos_df = pos_df.loc[~dust_mask].copy()
            last_px = load_last_prices_db(engine)
            if not last_px.empty:
                pos_df["symbol"] = pos_df["symbol"].astype(str).str.replace("/", "", regex=False)
                last_px = last_px.sort_values("ts").drop_duplicates("symbol", keep="last")
                pos_df = pos_df.merge(last_px[["symbol","last"]], on="symbol", how="left")
            if "last" in pos_df.columns:
                pos_df["position_value"] = (pd.to_numeric(pos_df["qty"], errors="coerce").abs() *
                                             pd.to_numeric(pos_df["last"], errors="coerce").abs()).round(2)
            df = pos_df
        else:
            st.info("No positions found in DB and broker positions unavailable.")

    cash_val = acct.get("cash") if isinstance(acct, dict) else None
    positions_mv = (
        float(
            pd.to_numeric(
                df.get("position_value", pd.Series(0.0, index=df.index)),
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )
        if not df.empty
        else 0.0
    )

    if (cash_val is None or (isinstance(cash_val, float) and (np.isnan(cash_val) or np.isinf(cash_val)))) and not df.empty:
        eq_series = load_equity_db(engine)
        if not eq_series.empty:
            latest_pv = pd.to_numeric(eq_series.sort_values("ts")["portfolio_value"].iloc[-1], errors="coerce")
            if pd.notna(latest_pv):
                cash_val = float(latest_pv - positions_mv)

    if cash_val is None and isinstance(acct, dict) and acct.get("equity") is not None:
        try:
            cash_val = float(acct.get("equity") - positions_mv)
        except Exception:
            pass

    if cash_val is not None and not (isinstance(cash_val, float) and np.isnan(cash_val)):
        cash_row = {
            "symbol": "CASH",
            "qty": float(cash_val),
            "avg_cost": 1.0,
            "last": 1.0,
            "position_value": float(cash_val),
            "unrealized_pl": 0.0,
            "unrealized_plpc": 0.0,
        }
        df = pd.concat([df, pd.DataFrame([cash_row])], ignore_index=True)

    if not df.empty:
        show_cols = ["symbol","qty","avg_cost","last","position_value","unrealized_pl","unrealized_plpc"]
        for c in show_cols:
            if c not in df.columns: df[c] = np.nan
        st.dataframe(df[show_cols], use_container_width=True, height=320)

    cash_for_summary = acct.get("cash") if isinstance(acct, dict) else None
    summary = _portfolio_summary(df, acct.get("equity") if isinstance(acct, dict) else None, cash_for_summary)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Portfolio Value", _fmt_money(summary["portfolio_value"]))
    c2.metric("Cash", _fmt_money(summary["cash"]))
    c3.metric("Long Exposure", _fmt_money(summary["long_exposure"]))
    c4.metric("Short Exposure", _fmt_money(summary["short_exposure"]))
    c5.metric("Unrealized P&L", _fmt_money(summary["unrealized_pl"]))
    c6.metric("Leverage", f"{summary['leverage']:.2f}×" if summary['leverage'] is not None else "—")
    st.caption(f"Source: {src} (cash included as part of portfolio)")

# ---------- Benchmark (SPY) loader ----------
@st.cache_data(ttl=300, show_spinner=False)
def load_spy_prices(start_utc: pd.Timestamp, end_utc: pd.Timestamp, interval: str = "1h") -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ['ts','close'] at UTC timestamps.
    Tries Alpaca Stocks data first (if keys present), otherwise falls back to yfinance.
    """
    start_utc = _ensure_utc(pd.Timestamp(start_utc))
    end_utc   = _ensure_utc(pd.Timestamp(end_utc))

    # Try Alpaca Stocks
    try:
        key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
        if key and sec:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            tf = {"1m": TimeFrame.Minute, "15m": TimeFrame(15, TimeFrameUnit.Minute),
                  "1h": TimeFrame.Hour, "1d": TimeFrame.Day}[interval]
            client = StockHistoricalDataClient(key, sec)
            req = StockBarsRequest(symbol_or_symbols="SPY",
                                   timeframe=tf,
                                   start=start_utc.to_pydatetime(),
                                   end=end_utc.to_pydatetime(),
                                   limit=10_000)
            bars = client.get_stock_bars(req)
            df = bars.df.reset_index()
            if "symbol" in df.columns:
                df = df[df["symbol"] == "SPY"]
            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "ts"})
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.rename(columns={"close": "close"})
            return df[["ts", "close"]].dropna().sort_values("ts")
    except Exception:
        pass

    # Fallback: yfinance (make sure 'yfinance' is in requirements)
    try:
        import yfinance as yf
        # Map our desired interval to yfinance granularity
        yf_int = {"1m": "1m", "15m": "15m", "1h": "60m", "1d": "1d"}[interval]
        # yfinance uses local-naive DatetimeIndex; we’ll convert to UTC
        data = yf.download("SPY", start=start_utc.tz_convert(None).to_pydatetime(),
                           end=end_utc.tz_convert(None).to_pydatetime(),
                           interval=yf_int, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return pd.DataFrame(columns=["ts","close"])
        df = data.rename(columns={"Close": "close"}).reset_index()
        ts_col = "Datetime" if "Datetime" in df.columns else "Date"
        df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        return df[["ts","close"]].dropna().sort_values("ts")
    except Exception:
        return pd.DataFrame(columns=["ts","close"])


# ---------- Stat helpers ----------
def _returns_from_equity(eq: pd.Series) -> pd.Series:
    # ensure numeric, don’t let inf/-inf leak into std/mean
    s = pd.to_numeric(eq, errors="coerce")
    r = s.pct_change()
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r

def _beta_to_spy(
    eq_window: pd.Series,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    spy_close: pd.Series,          # <-- NEW: pass SPY closes in
    rule: str = "1h",
) -> tuple[float, float, int]:
    """
    Compute beta and correlation of portfolio vs SPY over [start_utc, end_utc].
    Inputs:
      - eq_window: Series of portfolio equity with UTC DatetimeIndex.
      - spy_close: Series of SPY close with UTC DatetimeIndex.
      - rule: resample frequency (e.g., '1h', '15min'). Uses lowercase spellings.
    Returns: (beta, corr, n_points)
    """

    # normalize freq spellings to avoid FutureWarnings
    if rule.endswith("H"):  rule = rule[:-1] + "h"
    if rule.endswith("T"):  rule = rule[:-1] + "min"
    if rule.endswith("S"):  rule = rule[:-1] + "s"

    # ensure window bounds are tz-aware UTC
    if start_utc.tzinfo is None: start_utc = start_utc.tz_localize("UTC")
    if end_utc.tzinfo   is None: end_utc   = end_utc.tz_localize("UTC")

    # portfolio equity -> window slice -> resample last
    eq_series = eq_window.loc[(eq_window.index >= start_utc) & (eq_window.index <= end_utc)]
    if eq_series.empty:
        return float("nan"), float("nan"), 0
    eq_rs = (pd.to_numeric(eq_series, errors="coerce")
               .replace([np.inf, -np.inf], np.nan)
               .resample(rule).last())

    # SPY close -> window slice -> resample last
    spy_series = spy_close.loc[(spy_close.index >= start_utc) & (spy_close.index <= end_utc)]
    spy_rs = (pd.to_numeric(spy_series, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .resample(rule).last())

    # returns (don’t dropna yet), then align pairwise and clean
    # Do NOT forward-fill inside pct_change; make behavior explicit
    r_p = eq_rs.pct_change(fill_method=None)
    r_s = spy_rs.pct_change(fill_method=None)

    # Align pairwise and remove any NaN/±inf rows before stats
    both = pd.concat([r_p, r_s], axis=1, keys=["rp", "rs"])
    both = (both.astype("float64")
                .replace([np.inf, -np.inf], np.nan)
                .dropna(how="any"))
    # If your source can produce duplicate stamps, guard once:
    if both.index.has_duplicates:
        both = both[~both.index.duplicated(keep="last")]

    n = len(both)
    if n < 3:
        return float("nan"), float("nan"), n

    var_s = float(both["rs"].var(ddof=0))
    if not np.isfinite(var_s) or var_s == 0.0:
        return float("nan"), float("nan"), n

    cov_ps = float(both.cov(ddof=0).loc["rp", "rs"])
    beta = cov_ps / var_s
    corr = float(both.corr().loc["rp", "rs"])
    return beta, corr, n

def _max_drawdown(eq: pd.Series) -> tuple[pd.Series, float]:
    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    return dd, float(dd.min()) if len(dd) else float("nan")


def _var_series(eq: pd.Series, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> dict:
    """
    Historical & (optionally) normal VaR over intraday returns if available,
    else daily returns. Returns dict with 95%/99% VaR (as negative fractions).
    """
    eq = eq.copy()
    eq.index = pd.to_datetime(eq.index, utc=True)
    span = (end_utc - start_utc).total_seconds()

    # Choose cadence for VaR: intraday if we reasonably can
    if span <= 7 * 86400 and len(eq) >= 200:
        rule = "15min"  # 15-minute
    elif span <= 45 * 86400 and len(eq) >= 300:
        rule = "1h"   # hourly
    else:
        rule = "1D"   # daily

    eq_rs = eq.resample(rule).last().dropna()
    r = _returns_from_equity(eq_rs)
    if r.empty:
        return {"hist_95": float("nan"), "hist_99": float("nan")}

    hist_95 = float(-np.percentile(r.values, 5))
    hist_99 = float(-np.percentile(r.values, 1))
    return {"hist_95": hist_95, "hist_99": hist_99}


def _turnover_and_costs(engine: Optional["Engine"], start_utc: pd.Timestamp,
                        end_utc: pd.Timestamp, eq_window: pd.Series) -> tuple[float, float, int]:
    """
    Turnover = sum(abs(notional)) / average_equity over window.
    Trading costs = sum(fee) if present else fee_bps * notional.
    """
    # Look back from NOW to the window start (not just the window length),
    # so weekend/holiday windows still fetch the prior session’s fills.
    now_utc = pd.Timestamp.now(tz="UTC")          # simplest
    # or
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    # or (stdlib, version-agnostic)
    from datetime import datetime, timezone
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    hours_win = int(np.ceil((end_utc - start_utc).total_seconds() / 3600)) + 1
    hours_from_now = int(np.ceil((now_utc - start_utc).total_seconds() / 3600)) + 1
    hours = max(hours_win, hours_from_now)
    hours = min(hours, 24 * 30)  # safety cap: 30 days

    fills = load_fills_db(engine, hours=hours)
    if fills.empty:
        return 0.0, 0.0, 0
    f = fills.copy()
    f["ts"] = pd.to_datetime(f["ts"], utc=True, errors="coerce")
    f = f[(f["ts"] >= start_utc) & (f["ts"] <= end_utc)]
    if f.empty:
        return 0.0, 0.0, 0

    # Notional per fill
    f["qty"]   = pd.to_numeric(f["qty"], errors="coerce").abs()
    f["price"] = pd.to_numeric(f["price"], errors="coerce").abs()
    f["notional"] = f["qty"] * f["price"]

    # Fees: column if present else estimate from env fee_bps
    if "fee" in f.columns:
        fees = pd.to_numeric(f["fee"], errors="coerce").fillna(0.0).sum()
    else:
        fee_bps = _env_bps("FEE_BPS", 0.0)
        fees = float(f["notional"].sum() * (fee_bps / 10_000.0))

    avg_eq = float(pd.to_numeric(eq_window, errors="coerce").mean())
    if not np.isfinite(avg_eq) or avg_eq <= 0:
        return 0.0, float(fees), int(len(f))

    turnover = float(f["notional"].sum() / avg_eq)
    return max(0.0, turnover), float(fees), int(len(f))
    

def _concentration_df(engine: Optional["Engine"], eq_now: float) -> pd.DataFrame:
    """
    Current weights by symbol (incl. CASH if available).
    Robust to broker column naming differences and missing qty/last.
    """
    # Try broker first
    pos = load_positions_broker()

    if pos.empty and engine is not None:
        # Rebuild from recent fills (FIFO) + last prices in DB
        fills = load_fills_db(engine, hours=24*30)  # last month as a fallback
        last  = load_last_prices_db(engine)
        fifo  = _fifo_cost_basis(fills)
        if fifo.empty:
            return pd.DataFrame(columns=["symbol","weight"])
        df = fifo.merge(last, on="symbol", how="left")
        df["last"] = pd.to_numeric(df["last"], errors="coerce")
        df["qty"]  = pd.to_numeric(df["qty"], errors="coerce")
        df["position_value"] = (df["qty"].abs() * df["last"].abs())
        pos = df[["symbol","position_value"]].copy()
    else:
        if pos is None or pos.empty:
            return pd.DataFrame(columns=["symbol","weight"])

        # --- pick the symbol column robustly ---
        sym_col = first_existing(["symbol","asset","ticker","instrument","product"], list(pos.columns)) or "symbol"
        if sym_col not in pos.columns:
            return pd.DataFrame(columns=["symbol","weight"])

        # If we already have a usable notional, use it
        mv_col = first_existing(
            ["position_value","market_value","market_value_usd","marketvalue","notional","value"],
            list(pos.columns)
        )

        if mv_col:
            pos["position_value"] = pd.to_numeric(pos[mv_col], errors="coerce")
        else:
            # Fall back to qty * price with broad column name coverage
            qty_col = first_existing(["qty","quantity","position_qty","position","size","units"], list(pos.columns))
            px_col  = first_existing(["last","price","current_price","close","market_price"], list(pos.columns))

            if qty_col and px_col:
                pos["position_value"] = (
                    pd.to_numeric(pos[qty_col], errors="coerce").abs()
                    * pd.to_numeric(pos[px_col],  errors="coerce").abs()
                )
            else:
                # Nothing we can compute ⇒ no concentration
                return pd.DataFrame(columns=["symbol","weight"])

        # Keep only what we need, with canonical names
        pos = pos[[sym_col, "position_value"]].copy()
        pos = pos.rename(columns={sym_col: "symbol"})
        pos["position_value"] = pd.to_numeric(pos["position_value"], errors="coerce")

    # Final guards
    pos = pos.dropna(subset=["symbol","position_value"])
    if pos.empty or not np.isfinite(eq_now) or eq_now <= 0:
        return pd.DataFrame(columns=["symbol","weight"])

    # Aggregate & weights
    agg = pos.groupby("symbol", as_index=False)["position_value"].sum()
    agg["weight"] = (agg["position_value"] / float(eq_now)).clip(lower=-1e9, upper=1e9)
    agg = agg.sort_values("weight", ascending=False)
    return agg[["symbol","weight"]]
    """
    Current weights by symbol (incl. CASH if available).
    """
    pos = load_positions_broker()
    src = "broker"
    if pos.empty and engine is not None:
        # Rebuild from recent fills (FIFO) + last prices in DB
        fills = load_fills_db(engine, hours=24*30)  # last month as a fallback
        last  = load_last_prices_db(engine)
        fifo  = _fifo_cost_basis(fills)
        if fifo.empty:
            return pd.DataFrame(columns=["symbol","weight"])
        df = fifo.merge(last, on="symbol", how="left")
        df["last"] = pd.to_numeric(df["last"], errors="coerce")
        df["qty"]  = pd.to_numeric(df["qty"], errors="coerce")
        df["position_value"] = (df["qty"].abs() * df["last"].abs())
        pos = df[["symbol","position_value"]].copy()
        src = "db"
    else:
        if "position_value" not in pos.columns:
            pos["position_value"] = (pd.to_numeric(pos["qty"], errors="coerce").abs()
                                     * pd.to_numeric(pos["last"], errors="coerce").abs())
        pos = pos[["symbol","position_value"]].copy()

    if pos.empty or not np.isfinite(eq_now) or eq_now <= 0:
        return pd.DataFrame(columns=["symbol","weight"])

    agg = pos.groupby("symbol", as_index=False)["position_value"].sum()
    agg["weight"] = (agg["position_value"] / float(eq_now)).clip(lower=-1e9, upper=1e9)
    agg = agg.sort_values("weight", ascending=False)
    return agg[["symbol","weight"]]
# ------------------------------ Portfolio Value (rolling) ------------------------------
def render_equity(engine: Optional[Engine]) -> None:
    st.subheader("Portfolio Value (rolling)")
    
    # Refresh every 60s while on this tab
    # Disable auto-refresh while a backtest is running to avoid interruptions
    if not st.session_state.get("bt_running", False):
        safe_autorefresh(interval=60_000, key="pv_autorefresh")
    else:
        st.caption("Auto-refresh paused: backtest in progress")
        
    start_utc, end_utc, days, normalize, smooth = _time_window_controls(default="3M")

    broker_hist = load_portfolio_history_broker(days=days)
    src = "Alpaca Portfolio History" if not broker_hist.empty else "DB fallback"

    if not broker_hist.empty:
        df = broker_hist.copy()
    else:
        df = load_equity_db(engine)
        if df.empty:
            acct = load_account_broker()
            pv = acct.get("portfolio_value") or acct.get("equity")
            if pv is not None:
                df = pd.DataFrame(
                    [{"ts": pd.Timestamp.utcnow().tz_localize("UTC"), "portfolio_value": pv}]
                )
                src = "Account snapshot"
            else:
                df = pd.DataFrame(columns=["ts", "portfolio_value"])

    if df.empty:
        st.info("No portfolio value data available.")
        return

    df["ts"] = to_utc(df["ts"])
    df = (
        df.dropna(subset=["ts", "portfolio_value"])
          .sort_values("ts")
          .drop_duplicates("ts", keep="last")
    )

    # Normalize to UTC earlier (you already do df["ts"] = to_utc(df["ts"]))
    if start_utc is not None:
        df_win = df[(df["ts"] >= start_utc) & (df["ts"] <= end_utc)]
    else:
        df_win = df[df["ts"] <= end_utc]

    # Fallback to latest available window if empty (weekend/holiday/off-hours)
    if df_win.empty and not df.empty:
        dmax = df["ts"].max()
        dmin = df["ts"].min()
        lookback_days = max(int(days or 1), 1)  # 'days' comes from _time_window_controls
        start2 = max(dmin, dmax - pd.Timedelta(days=lookback_days))
        df_win = df[(df["ts"] >= start2) & (df["ts"] <= dmax)]

    df = df_win

    if df.empty:
        st.info("No data in selected window.")
        return

    # Use the ACTUAL shown window everywhere below
    win_start_utc = df["ts"].min()
    win_end_utc   = df["ts"].max()

    series = df.set_index("ts")["portfolio_value"].astype(float)
    if normalize:
        base = series.iloc[0]
        if base != 0 and pd.notna(base):
            series = 100.0 * series / base
        else:
            st.warning("Cannot normalize: first value is zero/NaN.")

    plot_df = pd.DataFrame({"ts": series.index, "value": series.values})
    if smooth and len(series) >= 7:
        plot_df["7d_smooth"] = series.rolling(7, min_periods=1).mean().values

    value_cols = ["value"] + (["7d_smooth"] if "7d_smooth" in plot_df.columns else [])
    plot_long = plot_df.melt(
        id_vars="ts", value_vars=value_cols, var_name="Series", value_name="y"
    )

    y_title = "Index (base=100)" if normalize else "Portfolio Value (USD)"

    brush = alt.selection_interval(encodings=["x"], clear="dblclick")
    nearest = alt.selection_point(nearest=True, on="pointermove",
                                  fields=["ts"], empty="none")

    base = alt.Chart(plot_long)

    lines = base.transform_calculate(
        ts_utc="utcFormat(toDate(datum.ts), '%Y-%m-%d %H:%M UTC')"
    ).mark_line().encode(
        x=alt.X("ts:T", title="Time (UTC)"),
        y=alt.Y("y:Q", title=y_title, scale=alt.Scale(zero=False, nice=True)),
        color=alt.Color("Series:N", legend=alt.Legend(title="Series", orient="top")),
        tooltip=[
            alt.Tooltip("ts_utc:N", title="Time"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("y:Q", title=y_title, format=",.2f"),
        ],
    ).transform_filter(brush)

    selectors = base.mark_point(opacity=0).encode(
        x="ts:T"
    ).add_params(nearest).transform_filter(brush)

    points = lines.mark_point().transform_filter(nearest)
    rule = base.mark_rule().encode(x="ts:T").transform_filter(nearest).transform_filter(brush)

    value_prefix = "" if normalize else "$"
    labels = (
        base.transform_filter(nearest)
            .transform_filter(brush)
            .transform_calculate(
                label=f"utcFormat(toDate(datum.ts), '%Y-%m-%d %H:%M UTC') + '  |  {value_prefix}' + format(datum.y, ',.2f')"
            )
            .mark_text(align="left", dx=6, dy=-6)
            .encode(
                x="ts:T",
                y="y:Q",
                text="label:N",
                color="Series:N",
            )
    )

    chart = (
        alt.layer(lines, selectors, points, rule, labels)
           .add_params(brush)
           .properties(height=360, width="container")
    )

    st.altair_chart(((chart).interactive()), use_container_width=True)
        # ---------------- Scorecard ("Pyfolio-style") ----------------
    st.divider()
    st.markdown(f"### Scorecard — since **{win_start_utc.date()}**")
    st.caption(
        f"Window shown: {win_start_utc.strftime('%Y-%m-%d %H:%M UTC')} → {win_end_utc.strftime('%Y-%m-%d %H:%M UTC')}"
    )
    # Equity window, metrics base
    eq_window = df.set_index("ts")["portfolio_value"].astype(float)
    if eq_window.empty:
        st.info("No equity data in the selected window for analytics.")
        return

    eq0 = float(eq_window.iloc[0])
    eqN = float(eq_window.iloc[-1])
    pnl = eqN - eq0
    ret = (eqN / eq0 - 1.0) if eq0 > 0 else float("nan")

    # Sharpe & DD
    bpyr = _infer_bars_per_year(eq_window.index.to_series())
    r = _returns_from_equity(eq_window)
    vol_ann = r.std(ddof=1) * math.sqrt(bpyr) if len(r) > 1 else float("nan")
    mu_ann  = r.mean() * bpyr if len(r) > 0 else float("nan")
    sharpe  = (mu_ann / vol_ann) if (vol_ann and not math.isnan(vol_ann) and vol_ann > 0) else float("nan")

    dd_series, mdd = _max_drawdown(eq_window)

    # You already have spy_df earlier (used to build spy charts/series)
    # Ensure UTC DatetimeIndex:
    # Beta & corr to SPY
    # Load SPY using your helper already in Home.py
    spy_df = load_spy_prices(win_start_utc, win_end_utc, interval="1h")
    spy_df = _ensure_ts_close(spy_df)     # ensures ['ts','close'] clean
    spy_close = (spy_df.set_index("ts")["close"].astype(float))
    spy_close.index = pd.to_datetime(spy_close.index, utc=True, errors="coerce")

    beta, corr, n_beta = _beta_to_spy(eq_window, win_start_utc, win_end_utc, spy_close, "1h")

    # Turnover & costs (uses fills)
    turnover, fees, n_trades = _turnover_and_costs(engine, win_start_utc, win_end_utc, eq_window)

    # VaR (intraday if available, else daily)
    var_dict = _var_series(eq_window, win_start_utc, win_end_utc)

    # ---- KPI row
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("P&L", f"${pnl:,.0f}", f"{ret*100:,.2f}%")
    k2.metric("Sharpe (approx.)", f"{sharpe:,.2f}")
    k3.metric("Max Drawdown", f"{mdd*100:,.2f}%")
    k4.metric("VaR 95% (hist)", f"{var_dict['hist_95']*100:,.2f}%")

    k5,k6,k7,k8 = st.columns(4)
    k5.metric("Beta to SPY", f"{beta:,.2f}" if not math.isnan(beta) else "—", help=f"n={n_beta}")
    k6.metric("Corr. to SPY", f"{corr:,.2f}" if not math.isnan(corr) else "—")
    k7.metric("Turnover", f"{turnover*100:,.2f}%" if turnover==turnover else "—")
    k8.metric("Trading Costs", f"${fees:,.0f}  |  trades: {n_trades}")

    # ---- Concentration chart
    st.markdown("#### Concentration (current)")
    conc = _concentration_df(engine, eq_now=eqN)
    if conc.empty:
        st.caption("No positions found.")
    else:
        topn = conc.head(12).copy()
        topn["pct"] = topn["weight"] * 100
        conc_chart = (
            alt.Chart(topn)
               .mark_bar()
               .encode(
                   x=alt.X("pct:Q", title="Weight (%)", axis=alt.Axis(format=",.1f")),
                   y=alt.Y("symbol:N", sort="-x", title=None),
                   tooltip=[alt.Tooltip("symbol:N", title="Symbol"),
                            alt.Tooltip("pct:Q", title="Weight (%)", format=",.2f")]
               )
               .properties(height=max(220, 18*len(topn)), width="container")
        )
        st.altair_chart(conc_chart, use_container_width=True)

    # ---- Drawdown curve
    st.markdown("#### Drawdown")
    dd_df = dd_series.reset_index().rename(columns={"index": "ts", 0: "dd"})
    dd_df.columns = ["ts","dd"]
    dd_chart = (
        alt.Chart(dd_df)
           .mark_area(opacity=0.5)
           .encode(x=alt.X("ts:T", title="Time (UTC)"),
                   y=alt.Y("dd:Q", title="Drawdown", scale=alt.Scale(domain=[dd_df["dd"].min()*1.05, 0], nice=False)),
                   tooltip=[alt.Tooltip("ts:T", title="Time (UTC)"),
                            alt.Tooltip("dd:Q", title="Drawdown", format=",.2%")])
           .properties(height=220, width="container")
    )
    st.altair_chart(dd_chart, use_container_width=True)

    # ---- Returns histogram (VaR visual)
    st.markdown("#### Returns distribution")
    hist = pd.DataFrame({"r": r.values})
    hist["r_pct"] = hist["r"] * 100.0
    hist_chart = (
        alt.Chart(hist)
           .mark_bar()
           .encode(x=alt.X("r_pct:Q", bin=alt.Bin(maxbins=60), title="Return (%)"),
                   y=alt.Y("count()", title="Count"),
                   tooltip=[alt.Tooltip("r_pct:Q", title="Return (%)", format=",.2f"), alt.Tooltip("count():Q")])
           .properties(height=220, width="container")
    )
    st.altair_chart(hist_chart, use_container_width=True)

    st.caption("• VaR shown uses historical method (negative % move). • All times UTC.")
    
    npts = len(series)
    left, right = st.columns([3, 2])
    with left:
        start_txt = df["ts"].iloc[0].strftime("%Y-%m-%d %H:%M UTC")
        end_txt = df["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
        st.caption(f"Window: {start_txt} → {end_txt} • Points: {npts} • Source: {src}")
    with right:
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="portfolio_value_window.csv",
            mime="text/csv",
        )

# ------------------------------ Fills ------------------------------
def render_fills(engine: Optional[Engine]) -> None:

    st.subheader("Recent Fills")

    # Auto-refresh every 30s while on this tab (unique key so it doesn't clash)
    try:
        # Disable auto-refresh while a backtest is running to avoid interruptions
        if not st.session_state.get("bt_running", False):
            safe_autorefresh(interval=30_000, key="fills_autorefresh")
        else:
            st.caption("Auto-refresh paused: backtest in progress")
    except Exception:
        pass

    c1, c2 = st.columns([1, 1])
    with c1:
        lookback = st.selectbox(
            "Lookback",
            ["24h", "3d", "7d", "30d"],
            index=0,
            help="Show fills within this rolling window."
        )
    with c2:
        if st.button("Refresh now"):
            # Clear just the fills cache + rerun for an immediate update
            try:
                load_fills_db.clear()  # type: ignore[attr-defined]
            except Exception:
                pass
            if not st.session_state.get("bt_running", False):
                st.rerun()            
            else:
                st.caption("Auto-refresh paused: backtest in progress")

    hours_map = {"24h": 24, "3d": 72, "7d": 168, "30d": 720}
    hours = hours_map.get(lookback, 24)

    df = load_fills_db(engine, hours=hours)
    if df.empty:
        st.write("No fills available in the selected window.")
        return

    d = df.copy()

    # ---- Standardize & derive the requested columns ----
    # Timestamps
    filled_at = pd.to_datetime(d.get("ts"), utc=True, errors="coerce")

    # Direct/fallback mappings
    asset = d.get("symbol")
    side = d.get("side")
    filled_qty = pd.to_numeric(d.get("filled_qty", d.get("qty")), errors="coerce")
    avg_fill_price = pd.to_numeric(d.get("avg_fill_price", d.get("price")), errors="coerce")
    status = d.get("status") if "status" in d.columns else "Filled"

    # Only keep the requested columns (removed: Order Type, Qty, Source, Submitted At)
    out = pd.DataFrame({
        "Asset": asset,
        "Side": side,
        "Filled Qty": filled_qty,
        "Avg. Fill Price": avg_fill_price,
        "Status": status,
        "Filled At": filled_at,
    })

    # Tidy up values/formatting
    out["Side"] = out["Side"].astype("string").str.lower().str.capitalize()
    out = out.sort_values("Filled At", ascending=False)

    st.dataframe(
        out,
        use_container_width=True,
        height=320,
        column_config={
            "Filled Qty": st.column_config.NumberColumn(format="%.8f"),
            "Avg. Fill Price": st.column_config.NumberColumn(format="$%.2f"),
            "Filled At": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss UTC"),
        },
    )
    st.caption(f"Window: last {lookback}")
# ------------------------------ Performance Helpers ------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _read_csv_fast(path: str) -> pd.DataFrame:
    from pathlib import Path as _P
    p = _P(path)
    if not p.exists():
        return pd.DataFrame()
    base = p.with_suffix("")
    pq = base.with_suffix(".parquet")

    try:
        if pq.exists() and pq.stat().st_mtime >= p.stat().st_mtime:
            try:
                df = pd.read_parquet(pq)
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                return df
            except Exception:
                pass
    except Exception:
        pass

    df = None
    try:
        import polars as pl  # type: ignore
        df_pl = pl.read_csv(path, try_parse_dates=True, low_memory=True)
        df = df_pl.to_pandas()
    except Exception:
        try:
            df = pd.read_csv(path, engine="pyarrow", memory_map=True)
        except Exception:
            df = pd.read_csv(path, low_memory=False, memory_map=True)

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    try:
        df.to_parquet(pq, index=False)
    except Exception:
        pass
    return df

def _lttb_indices(x: "np.ndarray", y: "np.ndarray", n_out: int) -> "np.ndarray":
    N = x.shape[0]
    if n_out >= N or n_out < 3:
        return np.arange(N, dtype=np.int64)
    bucket_size = (N - 2) / (n_out - 2)
    a = 0
    inds = [0]
    for i in range(0, n_out - 2):
        start = int(np.floor((i + 1) * bucket_size)) + 1
        end = int(np.floor((i + 2) * bucket_size)) + 1
        start = max(start, 1)
        end = min(end, N)
        if start >= end:
            start = max(1, start - 1)
            end = min(N, start + 1)

        avg_x = np.mean(x[start:end]) if end > start else x[start-1]
        avg_y = np.mean(y[start:end]) if end > start else y[start-1]

        range_offs = int(np.floor(i * bucket_size)) + 1
        range_to = int(np.floor((i + 1) * bucket_size)) + 1
        range_offs = max(inds[-1] + 1, range_offs)
        range_to = min(range_to, N - 1)
        if range_offs >= range_to:
            range_offs = max(inds[-1] + 1, range_offs - 1)
            range_to = min(N - 1, range_offs + 1)

        x_a = x[a]
        y_a = y[a]
        xs = x[range_offs:range_to]
        ys = y[range_offs:range_to]

        areas = np.abs((x_a - avg_x) * (ys - y_a) - (y_a - avg_y) * (xs - x_a))
        idx = np.argmax(areas)
        a = range_offs + idx
        inds.append(a)
    inds.append(N - 1)
    return np.array(inds, dtype=np.int64)

def _downsample_equity_for_plot(df_eq: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    if df_eq is None or df_eq.empty or "ts" not in df_eq.columns:
        return df_eq

    n = len(df_eq)
    if n <= max_points:
        return df_eq

    y_col = "equity" if "equity" in df_eq.columns else None
    if y_col is None:
        num_cols = [c for c in df_eq.columns if pd.api.types.is_numeric_dtype(df_eq[c])]
        y_col = num_cols[0] if num_cols else None
    if y_col is None:
        return df_eq.iloc[::max(1, n // max_points)]

    x = pd.to_datetime(df_eq["ts"], utc=True, errors="coerce").astype("int64").to_numpy()
    y = pd.to_numeric(df_eq[y_col], errors="coerce").to_numpy()
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return df_eq.iloc[::max(1, n // max_points)]

    x = x[mask]
    y = y[mask]
    idx_map = np.nonzero(mask)[0]

    keep_local = _lttb_indices(x, y, max_points)
    keep = idx_map[keep_local]
    keep.sort()

    return df_eq.iloc[keep].reset_index(drop=True)

# ------------------------------ Backtests (CSV/DB) ------------------------------

@st.cache_data(ttl=10, show_spinner=False)
def _list_backtest_runs_csv(out_dir: str = "./out") -> pd.DataFrame:
    out = []
    d = _Path(out_dir)
    if not d.exists():
        return pd.DataFrame(columns=["source","symbol","start","end","equity_path","trades_path","created_at",
                                     "strategy","tf_min","params"])
    pat = re.compile(r"^(backtest|trades)_(.+?)_(\d{8})_(\d{8})(?:_([A-Za-z0-9\-T]+))?\.csv$")
    files = list(d.glob("*.csv"))
    runs: Dict[Tuple[str,str,str,str], Dict[str, Any]] = {}
    for f in files:
        m = pat.match(f.name)
        if not m:
            continue
        kind, sym_safe, s, e, tag = m.groups()
        tag = tag or ""
        key = (sym_safe, s, e, tag)
        entry = runs.setdefault(key, {
            "symbol_safe": sym_safe, "symbol": sym_safe.replace('-', '/'),
            "start": s, "end": e, "equity_path": None, "trades_path": None
        })
        if kind == "backtest":
            entry["equity_path"] = str(f)
        else:
            entry["trades_path"] = str(f)

    for (sym_safe, s, e, tag), v in runs.items():
        if not v.get("equity_path"):
            continue
        created = dt.datetime.utcfromtimestamp(_Path(v["equity_path"]).stat().st_mtime).replace(tzinfo=dt.timezone.utc)

        meta_path = None
        try:
            base = os.path.basename(v["equity_path"])
            mm = re.match(r"^backtest_(.+?)_(\d{8})_(\d{8})_([A-Za-z0-9\-T]+)\.csv$", base)
            if mm:
                sym_safe2, s2, e2, run_id = mm.groups()
                meta_path = os.path.join(d, f"meta_{sym_safe2}_{s2}_{e2}_{run_id}.json")
        except Exception:
            meta_path = None

        strategy = None
        tf_min = None
        params: Any = None
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                strategy = meta.get("strategy")
                tf_min = meta.get("tf_min")
                params = meta.get("params")
            except Exception:
                pass

        out.append({
            "source": "csv",
            "symbol": v["symbol"],
            "start": dt.datetime.strptime(s, "%Y%m%d").replace(tzinfo=dt.timezone.utc),
            "end": dt.datetime.strptime(e, "%Y%m%d").replace(tzinfo=dt.timezone.utc),
            "equity_path": v.get("equity_path"),
            "trades_path": v.get("trades_path"),
            "created_at": created,
            "strategy": strategy,
            "tf_min": tf_min,
            "params": params,
        })

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def _load_backtest_equity_csv(path: str) -> pd.DataFrame:
    df = _read_csv_fast(path)
    if df.empty:
        return pd.DataFrame(columns=["ts","equity","buyhold"])
    if "ts" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ts"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "equity" not in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            df = df.rename(columns={num_cols[0]: "equity"})
    cols = [c for c in ["ts","equity","buyhold"] if c in df.columns]
    return df[cols].sort_values("ts").reset_index(drop=True)

@st.cache_data(ttl=1800, show_spinner=False)
def _load_backtest_trades_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["ts","side","qty","price","notional","fee"])
    try:
        df = _read_csv_fast(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=["ts","side","qty","price","notional","fee"])

def _infer_bars_per_year(ts: pd.Series) -> float:
    if ts.empty or len(ts) < 2:
        return 60*24*365
    dtmin = (ts.sort_values().diff().dropna().median()).total_seconds() / 60.0
    if dtmin <= 0:
        return 60*24*365
    return (60.0 / dtmin) * 24 * 365

def _compute_stats_from_equity(df_eq: pd.DataFrame) -> Dict[str, float]:
    if df_eq.empty or "equity" not in df_eq.columns:
        return {"bars": 0, "trades": 0, "total_return": 0.0, "sharpe_approx": float("nan"),
                "max_drawdown": float("nan"), "final_equity": float("nan")}
    eq = df_eq["equity"].astype(float)
    bars = len(eq)
    ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    bpyr = _infer_bars_per_year(df_eq["ts"])
    vol = ret.std(ddof=1) * math.sqrt(bpyr) if len(ret) > 1 else float("nan")
    sharpe = (ret.mean() * bpyr / vol) if (vol and not math.isnan(vol) and vol>0) else float("nan")
    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    stats = {
        "bars": int(bars),
        "trades": None,
        "total_return": float(eq.iloc[-1]/eq.iloc[0]-1.0) if bars >= 2 else 0.0,
        "sharpe_approx": float(sharpe) if sharpe==sharpe else float("nan"),
        "max_drawdown": float(dd.min()) if len(dd) else float("nan"),
        "final_equity": float(eq.iloc[-1]) if bars else float("nan"),
    }
    return stats

def _compute_stats_for_column(df_eq: pd.DataFrame, col: str) -> Dict[str, float]:
    if df_eq.empty or col not in df_eq.columns:
        return {"total_return": 0.0, "sharpe_approx": float("nan"), "max_drawdown": float("nan")}
    s = pd.to_numeric(df_eq[col], errors="coerce").dropna()
    if s.empty:
        return {"total_return": 0.0, "sharpe_approx": float("nan"), "max_drawdown": float("nan")}
    eq = s.astype(float)
    ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    bpyr = _infer_bars_per_year(df_eq["ts"]) if "ts" in df_eq.columns else 60*24*365
    vol = ret.std(ddof=1) * math.sqrt(bpyr) if len(ret) > 1 else float("nan")
    sharpe = (ret.mean() * bpyr / vol) if (vol and not math.isnan(vol) and vol>0) else float("nan")
    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    return {
        "total_return": float(eq.iloc[-1]/eq.iloc[0]-1.0) if len(eq) >= 2 else 0.0,
        "sharpe_approx": float(sharpe) if sharpe==sharpe else float("nan"),
        "max_drawdown": float(dd.min()) if len(dd) else float("nan"),
    }

# ---------------------- Backtest launcher helpers ----------------------

def _safe_name_for_file(sym: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '-', sym.replace('/', '-'))

def _save_outputs_fallback(cfg: Any, res: Any, out_dir: str) -> tuple[str, str]:
    """Save equity/trades when bt.save_outputs is unavailable."""
    os.makedirs(out_dir, exist_ok=True)

    sym_safe = _safe_name_for_file(getattr(cfg, "symbol", "SYMBOL"))
    start = getattr(cfg, "start", dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1))
    end   = getattr(cfg, "end",   dt.datetime.now(dt.timezone.utc))

    # Robust UTC conversion (avoid tz= on aware timestamps)
    ts_start = _as_utc_timestamp(start)
    ts_end   = _as_utc_timestamp(end)

    start_str = ts_start.strftime("%Y%m%d")
    end_str   = ts_end.strftime("%Y%m%d")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")

    eq_path = f"{out_dir}/backtest_{sym_safe}_{start_str}_{end_str}_{run_id}.csv"
    tr_path = f"{out_dir}/trades_{sym_safe}_{start_str}_{end_str}_{run_id}.csv"

    # Equity CSV
    equity = getattr(res, "equity", None)
    if isinstance(equity, (pd.Series, pd.DataFrame)) and len(equity) > 0:
        if isinstance(equity, pd.Series):
            df_eq = equity.rename("equity").to_frame()
        else:
            df_eq = equity.copy()
        if "ts" not in df_eq.columns:
            df_eq["ts"] = df_eq.index
        df_eq["ts"] = pd.to_datetime(df_eq["ts"], utc=True, errors="coerce")
    else:
        df_eq = pd.DataFrame(columns=["ts","equity"])

    # Include buy & hold if present
    bh = getattr(res, "bh_equity", None)
    try:
        if isinstance(bh, (pd.Series, pd.DataFrame)) and len(bh) > 0:
            if isinstance(bh, pd.Series):
                bh_df = bh.rename("buyhold").to_frame()
            else:
                if "buyhold" in bh.columns:
                    bh_df = bh[["buyhold"]].copy()
                else:
                    bh_df = bh.rename(columns={bh.columns[0]: "buyhold"})
            if "ts" not in bh_df.columns:
                bh_df["ts"] = bh_df.index
            bh_df["ts"] = pd.to_datetime(bh_df["ts"], utc=True, errors="coerce")
            df_eq = df_eq.merge(bh_df[["ts","buyhold"]], on="ts", how="left")
    except Exception:
        pass

    cols = ["ts","equity"] + (["buyhold"] if "buyhold" in df_eq.columns else [])
    df_eq[cols].to_csv(eq_path, index=False)

    # Trades CSV
    trades = getattr(res, "trades", None)
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        df_tr = trades.copy()
        if "ts" in df_tr.columns:
            df_tr["ts"] = pd.to_datetime(df_tr["ts"], utc=True, errors="coerce")
    else:
        df_tr = pd.DataFrame(columns=["ts","side","qty","price","notional","fee"])
    df_tr.to_csv(tr_path, index=False)

    return eq_path, tr_path

def _write_backtest_meta(eq_path: str, cfg: Any, run_params: dict) -> None:
    try:
        base = os.path.basename(eq_path)
        m = re.match(r"^backtest_(.+?)_(\d{8})_(\d{8})_([A-Za-z0-9\-T]+)\.csv$", base)
        if not m:
            return
        sym_safe, s, e, run_id = m.groups()
        meta_path = os.path.join(os.path.dirname(eq_path), f"meta_{sym_safe}_{s}_{e}_{run_id}.json")

        common = {
            "initial_cash": getattr(cfg, "initial_cash", None),
            "fee_bps": getattr(cfg, "fee_bps", None),
            "slippage_bps": getattr(cfg, "slippage_bps", None),
            "min_notional_usd": getattr(cfg, "min_notional_usd", None),
            "price_dec": getattr(cfg, "price_dec", None),
            "qty_dec": getattr(cfg, "qty_dec", None),
        }
        strat_params = {}
        if getattr(cfg, "strategy", None) == "ema":
            strat_params = {
                "ema_fast": getattr(cfg, "ema_fast", None),
                "ema_slow": getattr(cfg, "ema_slow", None),
                "adx_len": getattr(cfg, "adx_len", None),
                "adx_min": getattr(cfg, "adx_min", None),
                "use_pa_filter": getattr(cfg, "use_pa_filter", None),
                "pa_min_body_frac": getattr(cfg, "pa_min_body_frac", None),
            }
        elif getattr(cfg, "strategy", None) == "zscore":
            strat_params = {
                "z_n": getattr(cfg, "z_n", None),
                "z_entry": getattr(cfg, "z_entry", None),
                "z_exit": getattr(cfg, "z_exit", None),
                "z_stop": getattr(cfg, "z_stop", None),
                "adx_skip": getattr(cfg, "adx_skip", None),
                "z_adx_len": getattr(cfg, "z_adx_len", None),
                "rsi_len": getattr(cfg, "rsi_len", None),
                "rsi_lower": getattr(cfg, "rsi_lower", None),
                "rsi_upper": getattr(cfg, "rsi_upper", None),
            }

        payload = {
            "run_id": run_id,
            "symbol": getattr(cfg, "symbol", None),
            "strategy": getattr(cfg, "strategy", None),
            "tf_min": getattr(cfg, "tf_min", None),
            "start": getattr(cfg, "start", None).isoformat() if getattr(cfg, "start", None) else None,
            "end":   getattr(cfg, "end", None).isoformat() if getattr(cfg, "end", None) else None,
            "params": {**common, **strat_params, **(run_params or {})},
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass

def _run_backtest_from_ui(
    *, symbol, strategy, tf_min, use_dates, start_date, end_date, duration_days,
    initial_cash, fee_bps, slippage_bps, max_leverage, min_notional_usd, price_dec, qty_dec,
    ema_fast, ema_slow, adx_len, adx_min, use_pa_filter, pa_min_body_frac,
    z_n, z_entry, z_exit, z_adx_len, z_stop, adx_skip, rsi_len, rsi_lower, rsi_upper,
    api_key, api_secret, out_dir,
):
    import uuid, time, random, traceback as _tb

    if bt is None:
        st.error("part1_backtest.py is not importable.")
        return None

    # ----- dates
    if use_dates:
        start_dt = dt.datetime.combine(start_date, dt.time(0,0), tzinfo=dt.timezone.utc)
        end_dt   = dt.datetime.combine(end_date,   dt.time(23,59,59), tzinfo=dt.timezone.utc)
    else:
        end_dt = dt.datetime.now(dt.timezone.utc)
        start_dt = end_dt - dt.timedelta(days=max(1, int(duration_days)))

    # ----- size preflight (avoid huge fetches that can stall UI or hit API limits)
    try:
        span_min = max(1, int((end_dt - start_dt).total_seconds() // 60))
        est_bars = max(1, span_min // int(tf_min))
        MAX_BARS = int(os.getenv("MAX_BARS", "400000"))
        if est_bars > MAX_BARS:
            suggested_tf = max(1, math.ceil(span_min / MAX_BARS))
            st.error(
                f"Selected range ≈ {est_bars:,} bars at {tf_min}-minute TF "
                f"(cap is {MAX_BARS:,}). Try TF ≥ {suggested_tf} min or a shorter range."
            )
            return None
    except Exception:
        pass
    # ----- cfg
    cfg = bt.Config(
        symbol=symbol, tf_min=int(tf_min), start=start_dt, end=end_dt,
        initial_cash=float(initial_cash), price_dec=int(price_dec), qty_dec=int(qty_dec),
        fee_bps=float(fee_bps), slippage_bps=float(slippage_bps),
        strategy=strategy.lower(),
        ema_fast=int(ema_fast), ema_slow=int(ema_slow),
        adx_len=int(adx_len), adx_min=float(adx_min),
        use_pa_filter=bool(use_pa_filter), pa_min_body_frac=float(pa_min_body_frac),
        z_n=int(z_n), z_entry=float(z_entry), z_exit=float(z_exit),
        z_adx_len=int(z_adx_len), z_stop=float(z_stop), adx_skip=float(adx_skip),
        rsi_len=int(rsi_len), rsi_lower=float(rsi_lower), rsi_upper=float(rsi_upper),
        out_dir=out_dir, 
        max_leverage=float(max_leverage),
        min_notional_usd=float(min_notional_usd),
    )

    key = (api_key or os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or "").strip()
    sec = (api_secret or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET") or "").strip()
    if not key or not sec:
        st.error("Alpaca API keys are required (set env or use the override fields).")
        return None

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    os.makedirs(out_dir, exist_ok=True)
    err_path = os.path.join(out_dir, f"error_{run_id}.log")

    def _log_err(stage: str, exc: Exception):
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"[{stage}] {type(exc).__name__}: {exc}\n\n")
            f.write(_tb.format_exc())

    try:
        with st.status("Running backtest…", state="running", expanded=True) as status:
            status.write("Fetching market data… (with retries)")
            # ---- retry the fetch (handles 429s / timeouts)
            bars = None
            last_exc = None
            for i in range(3):
                try:
                    bars = bt.fetch_crypto_bars(cfg.symbol, cfg.start, cfg.end, cfg.tf_min, key, sec)
                    if strategy.lower() == "zscore" and len(bars) < int(z_n):
                        status.update(label="Not enough bars for Z_N", state="error")
                        st.error(f"Selected range has {len(bars)} bars but Z_N={z_n}. "
                                "Increase the date range or lower Z_N.")
                        return None
                    break
                except Exception as e:
                    last_exc = e
                    time.sleep(1.5 * (2 ** i) + random.random() * 0.5)
            if bars is None:
                _log_err("fetch", last_exc or Exception("unknown fetch error"))
                status.update(label="Backtest failed during data fetch", state="error")
                st.error(f"Fetch failed: {last_exc}")
                st.caption(f"See log: {os.path.basename(err_path)}")
                return None

            if bars.empty:
                status.update(label="No bars in selected range", state="error")
                st.warning("No bars returned for the selected range/timeframe.")
                return None

            status.write("Running simulation…")
            res = bt.run_backtest(cfg, bars)

            status.write("Saving outputs…")
            eq_path = tr_path = None
            try:
                if hasattr(bt, "save_outputs") and callable(getattr(bt, "save_outputs")):
                    eq_path, tr_path = bt.save_outputs(cfg, res, out_dir)  # project’s saver
                else:
                    eq_path, tr_path = _save_outputs_fallback(cfg, res, out_dir)  # dashboard fallback
                _write_backtest_meta(eq_path, cfg, run_params={})
            except Exception as e:
                _log_err("save", e)
                status.update(label="Backtest failed while saving outputs", state="error")
                st.error(f"Saving outputs failed: {e}")
                st.caption(f"See log: {os.path.basename(err_path)}")
                return None

            status.update(label="Backtest complete ✅", state="complete")
            return eq_path, tr_path

    except Exception as e:
        _log_err("run", e)
        st.error(f"Backtest failed: {e}")
        with st.expander("Details"):
            st.code(_tb.format_exc())
        return None

def _label_for_run(row: pd.Series) -> str:
    s = row["start"].strftime("%Y-%m-%d") if isinstance(row["start"], (dt.datetime, pd.Timestamp)) else str(row["start"])
    e = row["end"].strftime("%Y-%m-%d") if isinstance(row["end"], (dt.datetime, pd.Timestamp)) else str(row["end"])
    base = f"{row['symbol']}  {s} → {e}"
    if row.get("created_at"):
        ca = pd.to_datetime(row["created_at"])
        base += f"  ·  created {ca.strftime('%Y-%m-%d %H:%M UTC')}"
    strat = row.get("strategy") or "—"
    tfm_int = _to_int_or_none(row.get("tf_min"))
    tfm = f"{tfm_int}m" if tfm_int is not None else "—"
    base += f"  ·  {strat}  ·  TF {tfm}"
    return base

# ------------------------------ Backtests UI ------------------------------

def render_backtests() -> None:
    import os
    import datetime as dt
    import streamlit as st

    # -------- local, robust env parsers (comment/units-safe) --------
    def _env_float_local(k: str, default: float = 0.0) -> float:
        """
        Parse a float from an env var, ignoring inline comments and symbols like $ or commas.
        Accepts: "5", "5.0", "$5", "5  # comment", "5e3", etc.
        """
        s = os.getenv(k, "")
        s = str(s if s is not None else "")
        s = s.split("#", 1)[0].split("//", 1)[0]  # strip inline comments
        s = s.strip().lower().replace(",", "").replace("$", "")
        if not s:
            return float(default)
        import re
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        try:
            return float(m.group(0)) if m else float(default)
        except Exception:
            return float(default)

    def _env_int_local(k: str, default: int = 0) -> int:
        """Parse an int from an env var (via _env_float_local, then round)."""
        try:
            return int(round(_env_float_local(k, float(default))))
        except Exception:
            return int(default)

    def _env_bps_local(k: str, default: float = 0.0) -> float:
        """
        Read an env var as basis points (bps).
        Accepts: "0.1", "0.1  # comment", "10bps", "0.10%", "12 bp".
        If '%' present => percent -> bps (e.g., 0.10% -> 10 bps). Else treat number as bps.
        """
        s = os.getenv(k, "")
        s = str(s if s is not None else "")
        s = s.split("#", 1)[0].split("//", 1)[0]  # strip comments
        s = s.strip().lower().replace(",", "")
        if not s:
            return float(default)
        import re
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        try:
            x = float(m.group(0)) if m else float(default)
        except Exception:
            return float(default)
        return x * 100.0 if "%" in s else x

    def _env_bool_local(k: str, default: bool = False) -> bool:
        v = str(os.getenv(k, str(int(default)))).strip().lower()
        return v in ("1", "true", "t", "yes", "y")

    st.subheader("Backtests")

    # ---------- Run-a-new-backtest ----------
    with st.expander("Run a new backtest", expanded=True):
        c0, c1 = st.columns([2, 2])
        symbol = c0.text_input("Symbol", value=os.getenv("SYMBOL", "BTC/USD"))
        strategy_default = os.getenv("STRATEGY", "ema").lower()
        strategy = c1.selectbox("Strategy", ["ema", "zscore"],
                                index=0 if strategy_default == "ema" else 1)

        c2, c3, c4 = st.columns([1, 1, 1])
        tf_min = c2.number_input("Timeframe (minutes)", min_value=1, max_value=1440,
                                 value=_env_int_local("TF_MIN", 5))
        use_dates = c3.radio("Range", ["Duration", "Start/End"], horizontal=True) == "Start/End"

        if use_dates:
            today = dt.datetime.now(dt.timezone.utc).date()
            start_date = c4.date_input("Start (UTC)", value=today - dt.timedelta(days=90))
            end_date   = c4.date_input("End (UTC)",   value=today)
            duration_days = 90
        else:
            start_date = end_date = None
            duration_days = c4.number_input("Duration (days)", min_value=1, max_value=3650,
                                            value=_env_int_local("DURATION_DAYS", 365))

        st.markdown("**Execution & sizing**")
        c5, c6, c7, c8 = st.columns(4)
        initial_cash = c5.number_input("Initial cash (USD)", min_value=0.0,
                                       value=_env_float_local("INITIAL_CASH", 100_000.0))
        fee_bps      = c6.number_input("Fee (bps)",      min_value=0.0,
                                       value=_env_bps_local("FEE_BPS", 0.1))
        slippage_bps = c7.number_input("Slippage (bps)", min_value=0.0,
                                       value=_env_bps_local("SLIPPAGE_BPS", 0.02))
        min_notional = c8.number_input("Min notional (USD)", min_value=0.0,
                                       value=_env_float_local("MIN_NOTIONAL_USD", 5.0))

        adv = st.expander("Advanced", expanded=False)
        with adv:
            a1, a2, a3 = st.columns(3)
            price_dec = a1.number_input("Price decimals", min_value=0, max_value=10,
                                        value=_env_int_local("PRICE_DEC", 2))
            qty_dec   = a2.number_input("Qty decimals",   min_value=0, max_value=10,
                                        value=_env_int_local("QTY_DEC", 6))
            out_dir   = a3.text_input("Output folder", value=os.getenv("OUT_DIR", "./out"))

        if strategy == "ema":
            st.markdown("**EMA crossover + ADX settings**")
            e1, e2, e3, e4, e5, e6 = st.columns(6)
            ema_fast = e1.number_input("EMA fast", min_value=1,
                                       value=_env_int_local("EMA_FAST", 13))
            ema_slow = e2.number_input("EMA slow", min_value=2,
                                       value=_env_int_local("EMA_SLOW", 150))
            adx_len  = e3.number_input("ADX length", min_value=1,
                                       value=_env_int_local("ADX_LEN", 7))
            adx_min  = e4.number_input("ADX min", min_value=0.0,
                                       value=_env_float_local("ADX_MIN", 26.0))
            use_pa_filter = e5.checkbox("Use price-action filter",
                                        value=_env_bool_local("USE_PA_FILTER", True))
            pa_min_body_frac = e6.number_input("PA min body frac", min_value=0.0, max_value=1.0,
                                               value=_env_float_local("PA_MIN_BODY_FRAC", 0.5))
            # zscore placeholders
            z_n = z_entry = z_exit = z_stop = adx_skip = 0.0
            z_adx_len = 0
            rsi_len = 0
            rsi_lower = 0.0
            rsi_upper = 0.0
        else:
            st.markdown("**Z-score mean-reversion settings**")
            z1, z2, z3, z4, z5, z6 = st.columns(6)
            z_n      = z1.number_input("Lookback N", min_value=2,
                                       value=_env_int_local("Z_N", 144))
            z_entry  = z2.number_input("Z entry (≤ -x)",
                                       value=_env_float_local("Z_ENTRY", 2.4))
            z_exit   = z3.number_input("Z exit (≥ -x)",
                                       value=_env_float_local("Z_EXIT", 0.2))
            z_stop   = z4.number_input("Z stop (≤ -x)",
                                       value=_env_float_local("Z_STOP", 3.5))
            adx_skip = z5.number_input("Skip if ADX ≥",
                                       value=_env_float_local("ADX_SKIP", 25.0))
            z_adx_len= z6.number_input("ADX length", min_value=1,
                                       value=_env_int_local("Z_ADX_LEN",
                                                            _env_int_local("ADX_LEN", 14)))
            # RSI filters
            rz1, rz2, rz3 = st.columns(3)
            rsi_len = rz1.number_input("RSI length", min_value=2, value=_env_int_local("RSI_LEN", 14))
            rsi_lower = rz2.number_input("RSI lower (< oversold)", min_value=1.0, max_value=99.0, value=_env_float_local("RSI_LOWER", 30.0))
            rsi_upper = rz3.number_input("RSI upper (> overbought)", min_value=1.0, max_value=99.0, value=_env_float_local("RSI_UPPER", 70.0))
            # ema placeholders
            ema_fast = ema_slow = adx_len = 0
            adx_min = pa_min_body_frac = 0.0
            use_pa_filter = False

        with st.expander("Alpaca API override (optional)", expanded=False):
            k1, k2 = st.columns(2)
            api_key_in = k1.text_input("ALPACA_API_KEY_ID", value="", type="password")
            api_sec_in = k2.text_input("ALPACA_API_SECRET_KEY", value="", type="password")


        if "bt_running" not in st.session_state:
            st.session_state["bt_running"] = False
            st.session_state["bt_started_at"] = 0.0


        # show a manual unlock when locked
        if st.session_state.get("bt_running"):
            if st.button("Force-unlock Run button"):
                st.session_state["bt_running"] = False
                st.session_state["bt_started_at"] = 0.0
                st.rerun()
                
        do_run = st.button("Run backtest", type="primary", use_container_width=True,
                        disabled=st.session_state["bt_running"])
        max_lev = st.number_input("Max leverage", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
        min_notional = st.number_input("Min notional (USD)", min_value=0.0, value=5.0, step=0.01)
        if do_run and not st.session_state["bt_running"]:
            st.session_state["bt_running"] = True
            st.session_state["bt_started_at"] = time.time()   # <— add this line
            try:
                res = _run_backtest_from_ui(
                    symbol=symbol, strategy=strategy, tf_min=int(tf_min),
                    use_dates=use_dates, start_date=start_date, end_date=end_date,
                    duration_days=int(duration_days),
                    initial_cash=float(initial_cash),
                    fee_bps=float(fee_bps), slippage_bps=float(slippage_bps),
                    max_leverage=max_lev,
                    min_notional_usd=min_notional,
                    price_dec=int(price_dec), qty_dec=int(qty_dec),
                    ema_fast=int(ema_fast), ema_slow=int(ema_slow),
                    adx_len=int(adx_len), adx_min=float(adx_min),
                    use_pa_filter=bool(use_pa_filter), pa_min_body_frac=float(pa_min_body_frac),
                    z_n=int(z_n), z_entry=float(z_entry), z_exit=float(z_exit),
                    z_adx_len=int(z_adx_len),
                    z_stop=float(z_stop), adx_skip=float(adx_skip),
                    rsi_len=int(rsi_len), rsi_lower=float(rsi_lower), rsi_upper=float(rsi_upper),
                    api_key=(api_key_in or None), api_secret=(api_sec_in or None),
                    out_dir=out_dir,
                )
                if res:
                    eq_path, tr_path = res
                    st.success(f"- Equity: `{eq_path}`\n- Trades: `{tr_path}`")
                    st.cache_data.clear()
                    try: st.rerun()
                    except Exception: pass
            finally:
                st.session_state["bt_running"] = False
                st.session_state["bt_started_at"] = 0.0

    if st.button("Refresh runs", help="Re-scan the ./out folder for new results"):
        st.cache_data.clear() 
        if not st.session_state.get("bt_running", False):
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass     
        else:
            st.caption("Auto-refresh paused: backtest in progress")
    
    runs = _list_backtest_runs_csv()
    if runs.empty:
        st.info("No backtest runs found yet. Use the launcher above to generate one.")
        return

    runs = runs.sort_values("created_at", ascending=False).reset_index(drop=True)
    labels = [_label_for_run(runs.iloc[i]) for i in range(len(runs))]
    choice = st.selectbox("Select backtest run",
                          options=list(range(len(labels))),
                          format_func=lambda i: labels[i], index=0)

    row = runs.iloc[choice]
    st.write(f"**Source:** {row['source']} &nbsp; **Symbol:** {row['symbol']}")

    tfm_int = _to_int_or_none(row.get("tf_min"))
    tfm = f"{tfm_int}m" if tfm_int is not None else "—"
    st.markdown(f"**Strategy:** `{row.get('strategy','—')}`  ·  **TF:** `{tfm}`")

    params = row.get("params")
    if isinstance(params, dict) and params:
        st.markdown("**Parameters**")
        for k, v in params.items():
            st.markdown(f"- **{k}**: `{v}`")
    elif isinstance(params, str) and params.strip():
        st.code(params, language="json")
    else:
        st.caption("No parameters found for this run (older run without meta.json).")

        # --- New: one-click re-run from this saved entry ---
    p = row.get("params") or {}
    if st.button("Re-run with these params", use_container_width=True):
        strat = (row.get("strategy") or "ema").lower()
        tfm_int = _to_int_or_none(row.get("tf_min")) or 1

        kw = dict(
            symbol=row["symbol"],
            strategy=strat,
            tf_min=int(tfm_int),
            use_dates=True,
            start_date=row["start"].date(),
            end_date=row["end"].date(),
            duration_days=0,
            # common run params (fallbacks if older meta)
            initial_cash=float(p.get("initial_cash", 10_000)),
            fee_bps=float(p.get("fee_bps", 10)),
            slippage_bps=float(p.get("slippage_bps", 0)),
            max_leverage=float(p.get("max_leverage", 1.0)),
            min_notional_usd=float(p.get("min_notional_usd", 1.0)),
            price_dec=int(p.get("price_dec", 2) or 2),
            qty_dec=int(p.get("qty_dec", 6) or 6),
            # defaults for fields not used by the chosen strategy
            ema_fast=0, ema_slow=0, adx_len=0, adx_min=0.0, use_pa_filter=False, pa_min_body_frac=0.0,
            z_n=0, z_entry=0.0, z_exit=0.0, z_stop=0.0, adx_skip=0.0, z_adx_len=0, rsi_len=0, rsi_lower=0.0, rsi_upper=0.0,
            # use env creds if you don't override in the “Run a new backtest” expander
            api_key=os.getenv("ALPACA_API_KEY_ID", ""),
            api_secret=os.getenv("ALPACA_API_SECRET_KEY", ""),
            out_dir=os.getenv("OUT_DIR", "./out"),
        )

        if strat == "zscore":
            kw.update(
                z_n=int(p.get("z_n", p.get("Z_N", 144))),
                z_entry=float(p.get("z_entry", p.get("Z_ENTRY", 2.4))),
                z_exit=float(p.get("z_exit", p.get("Z_EXIT", 0.2))),
                z_stop=float(p.get("z_stop", p.get("Z_STOP", 3.5))),
                adx_skip=float(p.get("adx_skip", p.get("ADX_SKIP", 25.0))),
                z_adx_len=int(p.get("z_adx_len", p.get("Z_ADX_LEN", 14))),
            )
            rsi_len=int(p.get("rsi_len", p.get("RSI_LEN", 14))),
            rsi_lower=float(p.get("rsi_lower", p.get("RSI_LOWER", 30.0))),
            rsi_upper=float(p.get("rsi_upper", p.get("RSI_UPPER", 70.0))),
        else:  # ema
            kw.update(
                ema_fast=int(p.get("ema_fast", 40)),
                ema_slow=int(p.get("ema_slow", 200)),
                adx_len=int(p.get("adx_len", 14)),
                adx_min=float(p.get("adx_min", 26.0)),
                use_pa_filter=bool(p.get("use_pa_filter", False)),
                pa_min_body_frac=float(p.get("pa_min_body_frac", 0.5)),
            )

        ret = _run_backtest_from_ui(**kw)
        if ret:
            eq_path, tr_path = ret
            st.success(f"Re-ran. Equity: `{eq_path}` · Trades: `{tr_path}`")
            st.cache_data.clear()
            try: st.rerun()
            except Exception: pass
    df_eq = _load_backtest_equity_csv(row["equity_path"])
    df_tr = _load_backtest_trades_csv(row.get("trades_path"))
    stats = _compute_stats_from_equity(df_eq)

    bh_stats = None
    if 'buyhold' in df_eq.columns:
        try:
            bh_stats = _compute_stats_for_column(df_eq, 'buyhold')
        except Exception:
            bh_stats = None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bars", int(stats.get("bars") or 0))
    c2.metric("Trades", int(stats.get("trades") or (0 if df_tr.empty else len(df_tr))))
    c3.metric("Total Return", f"{(stats.get('total_return') or 0)*100:.2f}%")
    sharpe = stats.get("sharpe_approx")
    c4.metric("Sharpe (approx)", "—" if sharpe != sharpe else f"{sharpe:.2f}")
    mdd = stats.get("max_drawdown")
    c5.metric("Max Drawdown", "—" if mdd != mdd else f"{mdd*100:.2f}%")

    if bh_stats is not None:
        st.markdown("**Buy & Hold (same window)**")
        b1, b2, b3 = st.columns(3)
        b1.metric("Buy & Hold Return", f"{(bh_stats.get('total_return') or 0)*100:.2f}%")
        bh_sharpe = bh_stats.get('sharpe_approx')
        b2.metric("Buy & Hold Sharpe", "—" if (bh_sharpe != bh_sharpe) else f"{bh_sharpe:.2f}")
        _bh_mdd = bh_stats.get("max_drawdown")
        b3.metric("Buy & Hold Max Drawdown", "—" if _bh_mdd != _bh_mdd else f"{_bh_mdd*100:.2f}%")

    if not df_eq.empty:
        df_plot = _downsample_equity_for_plot(df_eq)
        if "ts" not in df_plot.columns:
            if df_plot.columns[0].lower().startswith("ts"):
                df_plot = df_plot.rename(columns={df_plot.columns[0]: "ts"})
            else:
                df_plot.index = pd.to_datetime(df_plot.index, utc=True)
                df_plot = df_plot.reset_index().rename(columns={"index": "ts"})
        rename_map = {"equity": "Strategy", "buyhold": "Buy & Hold"}
        for k, v in rename_map.items():
            if k in df_plot.columns:
                df_plot[v] = df_plot[k]
        if "Buy & Hold" not in df_plot.columns:
            st.caption("ℹ️ Buy & Hold not found in this run's CSV.")
        keep_cols = ["ts"] + [v for v in ["Strategy","Buy & Hold"] if v in df_plot.columns]
        df_plot = df_plot[keep_cols].copy()

        plot_long = df_plot.melt(id_vars=["ts"], var_name="Series", value_name="y")

        import altair as alt
        y_title = "Equity (USD)"
        brush = alt.selection_interval(encodings=["x"], clear="dblclick")
        nearest = alt.selection_point(nearest=True, on="pointermove", fields=["ts"], empty="none")
        base = alt.Chart(plot_long)

        lines = (
            base.transform_calculate(ts_utc="utcFormat(datum.ts, '%Y-%m-%d %H:%M UTC')")
                .mark_line()
                .encode(
                    x=alt.X("ts:T", title="Time (UTC)"),
                    y=alt.Y("y:Q", title=y_title, scale=alt.Scale(zero=False, nice=True)),
                    color=alt.Color("Series:N", legend=alt.Legend(title="Series", orient="top")),
                    tooltip=[
                        alt.Tooltip("ts_utc:N", title="Time"),
                        alt.Tooltip("Series:N", title="Series"),
                        alt.Tooltip("y:Q", title=y_title, format=",.2f"),
                    ],
                )
                .transform_filter(brush)
        )

        selectors = base.mark_point(opacity=0).encode(x="ts:T").add_params(nearest).transform_filter(brush)
        points = lines.mark_point().transform_filter(nearest)
        rule = base.mark_rule().encode(x="ts:T").transform_filter(nearest).transform_filter(brush)

        value_prefix = "$"
        labels = (
            base.transform_filter(nearest)
                .transform_filter(brush)
                .transform_calculate(
                    label="utcFormat(toDate(datum.ts), '%Y-%m-%d %H:%M UTC') + '  |  ' + '" + value_prefix + "' + format(datum.y, ',.2f')"
                )
                .mark_text(align="left", dx=6, dy=-6)
                .encode(x="ts:T", y="y:Q", text="label:N", color="Series:N")
        )

        chart = (
            alt.layer(lines, selectors, points, rule, labels)
               .add_params(brush)
               .properties(height=360, width="container")
        )
        st.altair_chart(((chart).interactive()), use_container_width=True)
        st.download_button("Download equity CSV", data=df_eq.to_csv(index=False), file_name="equity.csv")
    else:
        st.warning("No equity data found for this run.")

    if not df_tr.empty:
        st.dataframe(df_tr, use_container_width=True, height=260)
        st.download_button("Download trades CSV", data=df_tr.to_csv(index=False), file_name="trades.csv")
    else:
        st.caption("No trades recorded for this run (or CSV not found).")

# ------------------------------ BTC/USD Real-Time Prices (Alpaca) ------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def _load_btc_ohlc_alpaca(granularity: int, limit: int = 300) -> pd.DataFrame:
    """
    Fetch OHLCV candles for BTC/USD from Alpaca Market Data.
    granularity in seconds: 60, 900, 1800 (1m, 15m, 30m). Returns ascending time.
    """
    import datetime as dt

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(seconds=int(granularity) * int(limit))

    # --- Try SDK first (alpaca-py) ---
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf = {
            60:   TimeFrame.Minute,
            900:  TimeFrame(15, TimeFrameUnit.Minute),
            1800: TimeFrame(30, TimeFrameUnit.Minute),
        }[int(granularity)]

        key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
        sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")

        client = CryptoHistoricalDataClient(api_key=key, secret_key=sec) if (key and sec) else CryptoHistoricalDataClient()
        req = CryptoBarsRequest(
            symbol_or_symbols=["BTC/USD"],
            timeframe=tf,
            start=start,
            end=end,
            limit=int(limit),
        )
        bars = client.get_crypto_bars(req)
        df = bars.df.reset_index()

        # Expected columns: symbol, timestamp, open, high, low, close, volume, ...
        if "symbol" in df.columns:
            df = df[df["symbol"] == "BTC/USD"]
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        cols = [c for c in ["time","open","high","low","close","volume"] if c in df.columns]
        return df[cols].dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    except Exception:
        pass  # fall through to REST

    # --- REST fallback (v1beta3) ---
    try:
        import requests
        url = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
        tf_str = {60: "1Min", 900: "15Min", 1800: "30Min"}[int(granularity)]
        params = {
            "symbols": "BTC/USD",
            "timeframe": tf_str,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": int(limit),
        }
        headers = _alpaca_headers() or {"User-Agent": "crypto-trader-dashboard/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        j = r.json() or {}
        rows = (j.get("bars", {}).get("BTC/USD", [])) or (j.get("bars", {}).get("BTCUSD", []))
        out = []
        for b in rows:
            # Typical keys: t, o, h, l, c, v
            t = pd.to_datetime(b.get("t") or b.get("timestamp"), utc=True, errors="coerce")
            out.append({
                "time": t,
                "open":  b.get("o") or b.get("open"),
                "high":  b.get("h") or b.get("high"),
                "low":   b.get("l") or b.get("low"),
                "close": b.get("c") or b.get("close"),
                "volume":b.get("v") or b.get("volume"),
            })
        df = pd.DataFrame(out).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        return df[["time","open","high","low","close","volume"]] if not df.empty else df
    except Exception:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])


@st.cache_data(ttl=3, show_spinner=False)
def _load_latest_trade_alpaca(symbol: str = "BTC/USD") -> dict:
    """
    Get the latest trade for the given crypto symbol from Alpaca.
    Returns a dict like {"time": pd.Timestamp, "price": float, "size": float} or {} if unavailable.
    """
    try:
        import requests
    except Exception:
        return {}

    url = "https://data.alpaca.markets/v1beta3/crypto/us/trades/latest"
    params = {"symbols": symbol}
    headers = _alpaca_headers() or {"User-Agent": "crypto-trader-dashboard/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        j = r.json() or {}
        trow = (j.get("trades", {}).get(symbol)
                or j.get("trades", {}).get(symbol.replace("/", ""))
                or {})
        if not trow:
            return {}
        ts = pd.to_datetime(trow.get("t") or trow.get("timestamp"), utc=True, errors="coerce")
        price = trow.get("p") or trow.get("price")
        size = trow.get("s") or trow.get("size")
        if pd.isna(ts) or price is None:
            return {}
        return {"time": ts, "price": float(price), "size": float(size or 0)}
    except Exception:
        return {}


def _merge_inprogress_candle(df: pd.DataFrame, latest: dict, granularity_sec: int = 60) -> pd.DataFrame:
    """
    Given historical bars (ascending by 'time') and the latest trade, update/append the in-progress candle.
    - If latest trade falls into the same minute as the last bar: update close/high/low/volume.
    - If it's the next minute: append a new bar seeded from prior close.
    Returns a new DataFrame.
    """
    if df is None or df.empty or not latest:
        return df

    df = df.copy()
    last_bar_time = df["time"].iloc[-1]
    trade_ts = latest["time"]
    trade_price = latest["price"]
    trade_size = latest.get("size", 0.0)

    # Floor timestamps to bar boundary
    bar = pd.to_datetime(pd.Timestamp(trade_ts).floor(f"{granularity_sec}s"), utc=True)
    if bar == last_bar_time:
        # Update existing last bar
        df.loc[df.index[-1], "close"] = trade_price
        df.loc[df.index[-1], "high"] = max(df.iloc[-1]["high"], trade_price)
        df.loc[df.index[-1], "low"]  = min(df.iloc[-1]["low"],  trade_price)
        if "volume" in df.columns:
            try:
                cur = float(df.iloc[-1]["volume"])
            except Exception:
                cur = 0.0
            df.loc[df.index[-1], "volume"] = cur + float(trade_size)
        return df

    if bar > last_bar_time:
        # Append a new in-progress bar for the current minute
        prev_close = float(df.iloc[-1]["close"])
        new_row = {
            "time": bar,
            "open": prev_close,
            "high": max(prev_close, trade_price),
            "low":  min(prev_close, trade_price),
            "close": trade_price,
            "volume": float(trade_size),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return df

    return df

def _candlestick_chart_altair(df: pd.DataFrame, title: str = ""):
    """
    Prefer TradingView Lightweight Charts (full crosshair + zoom/pan).
    Falls back to an Altair candlestick if the component isn't available.
    """
    import altair as alt

    if df is None or df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text(size=14).encode(text="msg")

    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["time","open","high","low","close"]).reset_index(drop=True)
    if d.empty:
        return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text(size=14).encode(text="msg")

    # ---------- TradingView Lightweight Charts (preferred) ----------
    if _HAS_LWC:
        up = "#0ECB81"; dn = "#F6465D"

        # convert to LWC format
        candles = []
        vols = []
        for _, r in d.iterrows():
            t = pd.Timestamp(r["time"]).tz_convert("UTC").timestamp()
            candles.append({
                "time": t, "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"])
            })
            is_up = float(r["close"]) >= float(r["open"])
            vols.append({
                "time": t, "value": float(r.get("volume", 0.0)),
                "color": up if is_up else dn
            })

        chart = {
            "chart": {
                "layout": {"background": {"type": "solid", "color": "transparent"},
                           "textColor": "#c9d1d9"},
                "grid": {"vertLines": {"color": "rgba(197,203,206,0.2)"},
                         "horzLines": {"color": "rgba(197,203,206,0.2)"}},
                "rightPriceScale": {"scaleMargins": {"top": 0.1, "bottom": 0.25}},
                "timeScale": {"timeVisible": True, "secondsVisible": True, "rightOffset": 2},
                "crosshair": {"mode": 0},  # Normal crosshair, shows dashed lines + axis readouts
                "handleScroll": {"mouseWheel": True, "pressedMouseMove": True},
                "handleScale": {"mouseWheel": True, "pinch": True},
                "autoSize": True
            },
            "series": [
                {
                    "type": "Candlestick",
                    "data": candles,
                    "upColor": up, "downColor": dn,
                    "borderUpColor": up, "borderDownColor": dn,
                    "wickUpColor": up, "wickDownColor": dn,
                },
                {
                    "type": "Histogram",
                    "data": vols,
                    "priceScaleId": "",
                    "scaleMargins": {"top": 0.8, "bottom": 0.0}
                }
            ]
        }

        renderLightweightCharts([chart], key=f"tv-{title}")
        return None  # nothing to return; component already rendered

    # ---------- Altair fallback (keeps your old behavior) ----------
    up_color = "#0ECB81"; dn_color = "#F6465D"
    d["up"] = d["close"] >= d["open"]

    ttip = [
        alt.Tooltip("time:T",  title="Time (UTC)"),
        alt.Tooltip("open:Q",  title="Open",  format=",.2f"),
        alt.Tooltip("high:Q",  title="High",  format=",.2f"),
        alt.Tooltip("low:Q",   title="Low",   format=",.2f"),
        alt.Tooltip("close:Q", title="Close", format=",.2f"),
        alt.Tooltip("volume:Q",title="Volume",format="~s"),
    ]

    base = alt.Chart(d).encode(x=alt.X("time:T", title=None))
    wick = base.mark_rule().encode(
        y=alt.Y("low:Q", scale=alt.Scale(zero=False, nice=True)),
        y2="high:Q",
        color=alt.condition("datum.up", alt.value(up_color), alt.value(dn_color)),
        tooltip=ttip,
    )
    body = base.mark_bar(size=7, clip=True).encode(
        y="open:Q", y2="close:Q",
        color=alt.condition("datum.up", alt.value(up_color), alt.value(dn_color)),
        tooltip=ttip,
    )

    # --- NEW: Little dashes for zero-range candles (O=H=L=C) ---
    # Use a small horizontal tick centered at (time, close).
    eps = 1e-9
    d_dash = d[(np.isfinite(d["open"])) & (np.isfinite(d["high"])) & (np.isfinite(d["low"])) & (np.isfinite(d["close"])) &
               np.isclose(d["open"], d["high"], rtol=0, atol=eps) &
               np.isclose(d["open"], d["low"],  rtol=0, atol=eps) &
               np.isclose(d["open"], d["close"],rtol=0, atol=eps)]
    dash = (
        alt.Chart(d_dash)
            .transform_calculate(
                ts_utc="utcFormat(toDate(datum['ts']), '%Y-%m-%d %H:%M:%S UTC')"
            )
            .mark_tick(orient="horizontal", size=12)
            .encode(
                x=alt.X("ts:T"),
                y=alt.Y("close:Q", scale=alt.Scale(zero=False, nice=True)),
                color=alt.condition("datum.up", alt.value(up), alt.value(dn)),
                tooltip=ttip,
            )
    )

    # crosshair-like hover (nearest datum)
    hover = alt.selection_point(nearest=True, on="pointermove", fields=["time"], empty=False)
    vrule = base.mark_rule(strokeDash=[5,5]).encode(x="time:T").add_params(hover).transform_filter(hover)
    ylabels = (base.mark_text(align="left", dx=6, dy=-6)
                    .encode(x="time:T", y="close:Q",
                            text=alt.Text("close:Q", format=",.2f"),
                            color=alt.condition("datum.up", alt.value(up_color), alt.value(dn_color)))
                    .transform_filter(hover))

    last_close = float(d["close"].iloc[-1])
    ref = alt.Chart(pd.DataFrame({"y":[last_close]})).mark_rule(strokeDash=[5,5]).encode(
        y=alt.Y("y:Q", scale=alt.Scale(zero=False, nice=True))
    )

    price = (wick + body + dash + vrule + ylabels + ref).properties(height=360, title=title).interactive()
    vol = alt.Chart(d).mark_bar(size=7, opacity=0.65, clip=True).encode(
        x=alt.X("time:T", title=None),
        y=alt.Y("volume:Q", title="Vol", axis=alt.Axis(format="~s")),
        color=alt.condition("datum.up", alt.value(up_color), alt.value(dn_color)),
    ).properties(height=110)

    return alt.vconcat(price, vol).resolve_scale(x="shared")

def render_btc_usd() -> None:
    import os
    import datetime as dt
    import pandas as pd
    import numpy as np
    import streamlit as st

    # -------- optional deps (each used only if present) --------
    try:
        _HAS_AUTO = True
    except Exception:
        _HAS_AUTO = False

    try:
        from streamlit_lightweight_charts import renderLightweightCharts
        _HAS_LWC = True
    except Exception:
        _HAS_LWC = False

    # we’ll reuse your helper if present; else define a minimal fallback
    def _coinbase_df(granularity_sec: int, limit: int) -> pd.DataFrame:
        try:
            import requests
            end = dt.datetime.now(dt.timezone.utc)
            start = end - dt.timedelta(seconds=int(granularity_sec) * int(limit))
            url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
            params = {"granularity": int(granularity_sec),
                      "start": start.isoformat(),
                      "end": end.isoformat()}
            r = requests.get(url, params=params, headers={"User-Agent": "st-app"}, timeout=12)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or not data:
                return pd.DataFrame(columns=["time","open","high","low","close","volume"])
            df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.sort_values("time")[["time","open","high","low","close","volume"]]
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.dropna()
        except Exception:
            return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    # --- UPDATED: TradingView-like chart helper with explicit height and Altair force toggle ---
    def _tv_chart(df: pd.DataFrame, title: str, *, force_altair: bool = False):
        import altair as alt
        x_axis_utc = alt.Axis(
            title="Time (UTC)",
            # Force tick labels to UTC using Vega's utcFormat():
            labelExpr=f"utcFormat(datum.value, '{_time_fmt_d3}')"
        )
        if df is None or df.empty:
            return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text().encode(text="msg")

        d = df.copy()
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
        for c in ["open","high","low","close","volume"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["time","open","high","low","close"])
        if d.empty:
            return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text().encode(text="msg")

        up = "#0ECB81"; dn = "#F6465D"

        up_color, dn_color = up, dn 
        # --------- TradingView Lightweight Charts (preferred unless forced off) ---------
        if _HAS_LWC and not force_altair:
            try:
                candles, vols = [], []
                for _, r in d.iterrows():
                    t = pd.Timestamp(r["time"]).tz_convert("UTC").timestamp()
                    o, h, l, c, v = map(float, [r["open"], r["high"], r["low"], r["close"], r.get("volume", 0.0)])
                    candles.append({"time": t, "open": o, "high": h, "low": l, "close": c})
                    vols.append({"time": t, "value": v, "color": (up if c >= o else dn)})

                if candles:
                    chart = {
                        "chart": {
                            "layout": {"background": {"type": "solid", "color": "transparent"},
                                       "textColor": "#c9d1d9"},
                            "grid": {"vertLines": {"color": "rgba(197,203,206,0.2)"},
                                     "horzLines": {"color": "rgba(197,203,206,0.2)"}},
                            "rightPriceScale": {"scaleMargins": {"top": 0.1, "bottom": 0.25}},
                            "timeScale": {"timeVisible": True, "secondsVisible": True, "rightOffset": 2},
                            "crosshair": {"mode": 0},
                            "handleScroll": {"mouseWheel": True, "pressedMouseMove": True},
                            "handleScale": {"mouseWheel": True, "pinch": True},
                            "autoSize": True,
                            "height": 480,  # explicit so it can't collapse
                        },
                        "series": [
                            {"type": "Candlestick",
                             "data": candles,
                             "upColor": up, "downColor": dn,
                             "borderUpColor": up, "borderDownColor": dn,
                             "wickUpColor": up, "wickDownColor": dn},
                            {"type": "Histogram",
                             "data": vols,
                             "priceScaleId": "",
                             "scaleMargins": {"top": 0.8, "bottom": 0.0}}
                        ],
                    }
                    renderLightweightCharts([chart], key=f"tv-{title}")
                    return None
            except Exception:
                # fall through to Altair if any LWC issue
                pass

        # --------- Altair fallback (with labeled axes) ---------
        d["up"] = d["close"] >= d["open"]
        # normalize time
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
        d = d[d["time"].notna()].copy()
        d["ts"] = d["time"]
        # base layer
        base = (
            alt.Chart(d)
            # make a preformatted UTC string for tooltips & crosshair labels
            .transform_calculate(ts_utc=f"utcFormat(toDate(datum['ts']), '{_time_fmt_utc}')")
            .encode(x=alt.X("ts:T", axis=x_axis_utc))
        )

        ttip = [
            alt.Tooltip("ts_utc:N",  title="Time"),   # uses the UTC string above
            alt.Tooltip("open:Q",    title="Open",  format=",.2f"),
            alt.Tooltip("high:Q",    title="High",  format=",.2f"),
            alt.Tooltip("low:Q",     title="Low",   format=",.2f"),
            alt.Tooltip("close:Q",   title="Close", format=",.2f"),
            alt.Tooltip("volume:Q",  title="Volume",format="~s"),
        ]

        # wick/body layers keep using `base` so they inherit the UTC x-axis
        wick = base.mark_rule().encode(
            y=alt.Y("low:Q",  scale=alt.Scale(zero=False, nice=True)),
            y2="high:Q",
            color=alt.condition("datum.up", alt.value(up_color), alt.value(dn_color)),
            tooltip=ttip,
        )

        body = base.mark_bar(size=7, clip=True).encode(
            y="open:Q", y2="close:Q",
            color=alt.condition("datum.up", alt.value(up_color), alt.value(dn_color)),
            tooltip=ttip,
        )
        # --- NEW: Little dashes for zero-range candles (O=H=L=C) in Altair fallback ---
        eps = 1e-9
        d_dash = d[(np.isfinite(d["open"])) & (np.isfinite(d["high"])) & (np.isfinite(d["low"])) & (np.isfinite(d["close"])) &
                   np.isclose(d["open"], d["high"], rtol=0, atol=eps) &
                   np.isclose(d["open"], d["low"],  rtol=0, atol=eps) &
                   np.isclose(d["open"], d["close"],rtol=0, atol=eps)]
        dash = (
            alt.Chart(d_dash)
                .transform_calculate(
                    # compute the tooltip string from the SAME field used on the x-axis
                    ts_utc="utcFormat(toDate(datum['time']), '%Y-%m-%d %H:%M:%S UTC')"
                )
                .mark_tick(orient="horizontal", size=12)
                .encode(
                    x=alt.X("time:T"),   # leave this as 'time:T' for the dash layer
                    y=alt.Y("close:Q", scale=alt.Scale(zero=False, nice=True)),
                    color=alt.condition("datum.up", alt.value(up), alt.value(dn)),
                    tooltip=ttip,  # make sure ttip includes alt.Tooltip("ts_utc:N", title="Time")
                )
        )

        hover = alt.selection_point(nearest=True, on="pointermove", fields=["time"], empty=False)
        vline = base.mark_rule(strokeDash=[5,5]).encode(x="time:T").add_params(hover).transform_filter(hover)
        last = float(d["close"].iloc[-1])
        ref = alt.Chart(pd.DataFrame({"y": [last]})).mark_rule(strokeDash=[5,5]).encode(
            y=alt.Y("y:Q", scale=alt.Scale(zero=False, nice=True))
        )
        # Add a crosshair label that shows "time | close" using UTC format
        labels = (
            base.transform_filter(hover)
                .transform_calculate(
                    label=f"utcFormat(toDate(datum['ts']), '{_time_fmt_utc}') + '  |  ' + format(datum.close, ',.2f')"
                )
                .mark_text(align="left", dx=6, dy=-6)
                .encode(x="time:T", y="close:Q", text="label:N",
                        color=alt.condition("datum.up", alt.value(up_color), alt.value(dn_color)))
        )
        price = (wick + body + dash + vline + ref + labels).properties(height=360, title=title).interactive()
                
        # volume also uses the UTC axis
        vol = alt.Chart(d).mark_bar(size=7, opacity=0.65, clip=True).encode(
            x=alt.X("time:T", axis=x_axis_utc, title=None),
            y=alt.Y("volume:Q", title="Vol", axis=alt.Axis(format="~s")),
            color=alt.condition("datum.up", alt.value(up_color), alt.value(dn_color)),
        ).properties(height=110)
        return alt.vconcat(price, vol).resolve_scale(x="shared")

    # ---- Alpaca loaders (SDK first, REST fallback) ----
    def _alpaca_df(tf_str: str, limit: int) -> pd.DataFrame:
        key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
        if not key or not sec:
            return pd.DataFrame()

        # SDK path
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            tf = {"1m": TimeFrame.Minute,
                  "15m": TimeFrame(15, TimeFrameUnit.Minute),
                  "30m": TimeFrame(30, TimeFrameUnit.Minute),
                  "1h": TimeFrame.Hour,
                  "1d": TimeFrame.Day}[tf_str]

            end = pd.Timestamp.utcnow().tz_localize("UTC")
            _sec_map = {"1m": 60, "15m": 900, "30m": 1800, "1h": 3600, "1d": 86400}
            seconds = _sec_map[tf_str] * max(1, int(limit))
            start = end - pd.Timedelta(seconds=seconds)

            client = CryptoHistoricalDataClient(key, sec)
            req = CryptoBarsRequest(
                symbol_or_symbols="BTC/USD",
                timeframe=tf,
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                limit=int(limit),
            )
            bars = client.get_crypto_bars(req)
            # dataframe lives at .df; index may be MultiIndex(symbol, timestamp)
            df = getattr(bars, "df", None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.reset_index()
                # normalize column names across SDK versions
                if "timestamp" in df.columns:
                    df = df.rename(columns={"timestamp": "time"})
                if "symbol" in df.columns:
                    df = df[df["symbol"].astype(str).str.contains("BTC", na=False)]
                keep = [c for c in ["time","open","high","low","close","volume"] if c in df.columns]
                if "time" in keep:
                    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                for c in keep:
                    if c != "time":
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["time","open","high","low","close"]).sort_values("time")
                return df[["time","open","high","low","close","volume"]] if "volume" in df.columns else \
                       df.assign(volume=np.nan)[["time","open","high","low","close","volume"]]
        except Exception:
            pass

        # REST path (v1beta3)
        try:
            import requests
            tf_map = {"1m": "1Min", "15m": "15Min", "30m": "30Min", "1h": "1Hour", "1d": "1Day"}
            end = dt.datetime.now(dt.timezone.utc)
            _sec_map = {"1m": 60, "15m": 900, "30m": 1800, "1h": 3600, "1d": 86400}
            seconds = _sec_map[tf_str] * max(1, int(limit))
            start = end - dt.timedelta(seconds=seconds)
            base = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
            params = {
                "symbols": "BTC/USD",
                "timeframe": tf_map[tf_str],
                "limit": int(limit),
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
            }
            headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
            r = requests.get(base, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            j = r.json()
            # shape: {"bars": {"BTC/USD": [{"t": "...", "o":..,"h":..,"l":..,"c":..,"v":..}, ...]}}
            series = None
            if isinstance(j, dict):
                b = j.get("bars") or {}
                # key variants
                for k in ["BTC/USD", "BTCUSD", "BTCUSDT", "BTC-USD"]:
                    if k in b:
                        series = b[k]
                        break
            if not series:
                return pd.DataFrame()
            df = pd.DataFrame(series)
            # normalize keys across versions
            col_map = {"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
                       "timestamp": "time"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            for c in ["open","high","low","close","volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["time","open","high","low","close"]).sort_values("time")
            # ensure all columns exist
            for c in ["volume"]:
                if c not in df.columns: df[c] = np.nan
            return df[["time","open","high","low","close","volume"]]
        except Exception:
            return pd.DataFrame()

    # ---------------- UI ----------------
    st.subheader("BTC/USD — Real-Time History")

    top = st.container()
    with top:
        c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.2, 1.2])
        with c1:
            tf_choice = st.radio("Bars (timeframe)", ["1m", "15m", "30m", "1h", "1d"], index=0, horizontal=True, key="btc_tf")
            # Time label/tooltip format based on timeframe (Altair fallback uses this)
            if tf_choice in ("1m","15m","30m"):
                _time_fmt_d3 = "%Y-%m-%d %H:%M"  # minute precision
                _time_fmt_utc = "%Y-%m-%d %H:%M UTC"
            elif tf_choice == "1h":
                _time_fmt_d3 = "%Y-%m-%d %H:00"  # hour precision
                _time_fmt_utc = "%Y-%m-%d %H:00 UTC"
            else:  # "1d"
                _time_fmt_d3 = "%Y-%m-%d"       # day precision
                _time_fmt_utc = "%Y-%m-%d UTC"
        with c2:
            period_choice = st.radio("Period (history window)", ["1D", "1M", "3M"], index=0, horizontal=True, key="btc_period")
        with c3:
            refresh = st.selectbox("Auto-refresh (sec)", [0, 5, 10, 15, 30, 60], index=3, key="btc_refresh")


        with c4:
            _cap_default = int(os.getenv("BTC_BAR_CAP", "1000"))
            bar_cap = st.number_input(
                "Bar limit (cap)", min_value=100, max_value=10000, value=_cap_default, step=100,
                key="btc_bar_cap", help="Max bars to fetch; drag the chart to reveal earlier history."
            )

        # Compute bar limit from timeframe + period (cap at 300)
        import math
        _tf_sec = {"1m": 60, "15m": 900, "30m": 1800, "1h": 3600, "1d": 86400}[tf_choice]
        _period_sec = {"1D": 86400, "1M": 30*86400, "3M": 90*86400}[period_choice]
        _needed = int(math.ceil(_period_sec / _tf_sec)) + 1
        _cap = int(bar_cap)
        limit = min(_needed, _cap)
        _clamped = _needed > _cap

    # NEW: let user force the Altair chart (with axes)
    force_altair = True

    # Fire auto-refresh without reloading the whole page (if component is available)
    if int(refresh) > 0:
        if _HAS_AUTO:
            # Disable auto-refresh while a backtest is running to avoid interruptions
            if not st.session_state.get("bt_running", False):
                safe_autorefresh(int(refresh) * 1000, key="btc_auto_tick")
            else:
                st.caption("Auto-refresh paused: backtest in progress")
            st.caption(f"Auto-refreshing every {int(refresh)}s")
        else:
            st.caption("Install `streamlit-autorefresh` to enable auto-refresh. Use the button below to refresh.")
            if st.button("Refresh now", use_container_width=False):
                if not st.session_state.get("bt_running", False):
                    st.rerun()  # safe
                else:
                    st.caption("Auto-refresh paused: backtest in progress")

    # --------------- DATA ---------------
    # Alpaca first (SDK → REST), Coinbase fallback
    if tf_choice == "1m":
        df = _alpaca_df("1m", limit)
        if df.empty:
            df = _coinbase_df(60, limit)
    elif tf_choice == "15m":
        df = _alpaca_df("15m", limit)
        if df.empty:
            df = _coinbase_df(900, limit)
    elif tf_choice == "30m":
        df = _alpaca_df("30m", limit)
        if df.empty:
            df = _coinbase_df(1800, limit)
    elif tf_choice == "1h":
        df = _alpaca_df("1h", limit)
        if df.empty:
            df = _coinbase_df(3600, limit)
    else:
        df = _alpaca_df("1d", limit)
        if df.empty:
            df = _coinbase_df(86400, limit)
    title = f"BTC/USD — {tf_choice} — {period_choice}"

    # --------------- HEADER METRIC ---------------
    if not df.empty:
        last = df.iloc[-1]
        delta = float(last["close"]) - float(last["open"])
        st.metric("Last price", f"{last['close']:,.2f} USD", f"{delta:+.2f}")
        if _clamped:
            st.caption(f"Showing last {limit} bars (window > cap).")
    else:
        st.info("No data available (check API keys / network).")

    # --------------- CHART ---------------
    chart_obj = _tv_chart(df, title, force_altair=force_altair)
    if chart_obj is not None:  # Altair fallback or forced Altair
        st.altair_chart(chart_obj, use_container_width=True)

    src = "Alpaca (SDK/REST)" if not df.empty else "—"
    st.caption(f"Data source: {src} · Times are UTC")

# ------------------------------ Main ------------------------------
def main() -> None:
    st.title("Crypto Trader Dashboard")
    # Initialize the backtest-running flag early
    if "bt_running" not in st.session_state:
        st.session_state["bt_running"] = False
    # Exact current time in UTC
    _now_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.caption(f"Now: {_now_utc}")
    eng = get_engine()

    tabs = st.tabs(["Portfolio", "Portfolio Value", "Fills", "Backtests", "BTC/USD"])
    with tabs[0]:
        render_portfolio(eng)
    with tabs[1]:
        render_equity(eng)
    with tabs[2]:
        render_fills(eng)
    with tabs[3]:
        render_backtests()
    with tabs[4]:
        render_btc_usd()

if __name__ == "__main__":
    main()
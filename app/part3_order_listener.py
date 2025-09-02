# part3_order_listener.py
# ------------------------------------------------------------
# Order-update listener for Alpaca (alpaca-py).
#
# - Subscribes to trade_updates via TradingStream (paper or live)
# - Pretty console logs, optional JSON logs
# - Optional per-symbol filter (SYMBOL_FILTER="BTC/USD,ETH/USD")
# - Optional Postgres persistence (DB_DSN="postgres://user:pass@host:5432/db")
# - Safe dedup (order_id, event, ts)
# - Graceful reconnects and shutdown
#
# 
# Environment variables (see .env)
# ------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple
from psycopg2.extras import Json
# Alpaca stream
from alpaca.trading.stream import TradingStream

# Optional DB
try:
    import psycopg2
    import psycopg2.extras
    _HAS_DB = True
except Exception:
    _HAS_DB = False

# ------------------------- small utils -------------------------

def env_str(k: str, default: str = "") -> str:
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

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

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

def parse_symbol_filter(raw: str) -> Optional[Set[str]]:
    if not raw:
        return None
    syms = {symbol_for_data(x) for x in raw.replace(";", ",").split(",") if x.strip()}
    return syms or None

# ---------------------------- config ----------------------------

@dataclass
class Cfg:
    paper: bool
    log_json: bool
    sym_filter: Optional[Set[str]]
    db_dsn: Optional[str]
    db_table: str
    price_dec: int
    qty_dec: int
    reconnect_max_sec: int

def build_cfg() -> Cfg:
    return Cfg(
        paper=(env_int("ALPACA_PAPER", 1) == 1),
        log_json=(env_int("LOG_JSON", 0) == 1),
        sym_filter=parse_symbol_filter(env_str("SYMBOL_FILTER", "")),
        db_dsn=(env_str("DB_DSN", "").strip() or None),
        db_table=env_str("DB_TABLE", "order_events"),
        price_dec=env_int("PRICE_DEC", 2),
        qty_dec=env_int("QTY_DEC", 6),
        reconnect_max_sec=env_int("RECONNECT_MAX_SEC", 60),
    )

# --------------------------- db helper ---------------------------

class DBWriter:
    def __init__(self, dsn: str, table: str):
        if not _HAS_DB:
            raise RuntimeError("psycopg2 not installed")
        self.dsn = dsn
        self.table = table
        self.conn = None  # type: ignore

    def connect(self):
        self.conn = psycopg2.connect(self.dsn)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                  id SERIAL PRIMARY KEY,
                  event_time TIMESTAMPTZ NOT NULL,
                  event TEXT NOT NULL,
                  order_id TEXT NOT NULL,
                  client_order_id TEXT,
                  symbol TEXT,
                  side TEXT,
                  type TEXT,
                  time_in_force TEXT,
                  status TEXT,
                  qty NUMERIC,
                  filled_qty NUMERIC,
                  limit_price NUMERIC,
                  stop_price NUMERIC,
                  avg_fill_price NUMERIC,
                  last_fill_price NUMERIC,
                  notional NUMERIC,
                  raw JSONB
                );
                CREATE INDEX IF NOT EXISTS {self.table}_order_id_idx ON {self.table}(order_id);
                """
            )

    def write(self, rec: Dict[str, Any]):
        if not self.conn:
            self.connect()
        cols = [
            "event_time","event","order_id","client_order_id","symbol",
            "side","type","time_in_force","status","qty","filled_qty",
            "limit_price","stop_price","avg_fill_price","last_fill_price","notional","raw"
        ]
        vals = [rec.get(k) for k in cols]
        
        # --- wrap dicts so psycopg can adapt them to JSONB ---
        raw_idx = cols.index("raw")
        if isinstance(vals[raw_idx], dict):
            vals[raw_idx] = Json(vals[raw_idx])
            
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {self.table} ({','.join(cols)}) VALUES %s",
                [tuple(vals)],
            )

# ------------------------ parsing/formatting ------------------------

def _get(d: Any, path: Tuple[str, ...], default=None):
    cur = d
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            cur = getattr(cur, p, None)
    return cur if cur is not None else default

def _round(x: Any, dec: int) -> Optional[float]:
    try:
        return None if x is None else round(float(x), dec)
    except Exception:
        return None

def parse_update(data: Any, price_dec: int, qty_dec: int) -> Dict[str, Any]:
    # Compatible with both dict-like payloads and alpaca-py OrderUpdate model
    event = _get(data, ("event",), "")
    ts = _get(data, ("timestamp",), None) or _get(data, ("updated_at",), None)
    if isinstance(ts, str):
        event_time = ts
    else:
        try:
            event_time = ts.isoformat()
        except Exception:
            event_time = utcnow_iso()

    order = _get(data, ("order",), {}) or {}
    # primary fields
    symbol = symbol_for_data(_get(order, ("symbol",), "") or "")
    side = (_get(order, ("side",), "") or "").lower() or None
    otype = (_get(order, ("type",), "") or "").lower() or None
    tif = (_get(order, ("time_in_force",), "") or "") or None
    status = (_get(order, ("status",), "") or "") or None
    order_id = _get(order, ("id",), None) or _get(order, ("client_order_id",), None)
    cl_id = _get(order, ("client_order_id",), None)

    qty = _round(_get(order, ("qty",), None), qty_dec) or _round(_get(order, ("quantity",), None), qty_dec)
    filled_qty = _round(_get(order, ("filled_qty",), None), qty_dec) or _round(_get(order, ("filled_quantity",), None), qty_dec)
    limit_price = _round(_get(order, ("limit_price",), None), price_dec)
    stop_price  = _round(_get(order, ("stop_price",), None), price_dec)
    avg_fill = _round(_get(order, ("filled_avg_price",), None), price_dec)
    last_fill_price = _round(_get(data, ("price",), None), price_dec)  # present on fill updates
    notional = _round(_get(order, ("notional",), None), price_dec)

    rec = {
        "event_time": event_time,
        "event": str(event),
        "order_id": str(order_id) if order_id else None,
        "client_order_id": str(cl_id) if cl_id else None,
        "symbol": symbol or None,
        "side": side,
        "type": otype,
        "time_in_force": tif,
        "status": status,
        "qty": qty,
        "filled_qty": filled_qty,
        "limit_price": limit_price,
        "stop_price": stop_price,
        "avg_fill_price": avg_fill,
        "last_fill_price": last_fill_price,
        "notional": notional,
        "raw": data if isinstance(data, dict) else json.loads(json.dumps(data, default=lambda o: getattr(o, "__dict__", str(o))))
    }
    return rec

def format_pretty(rec: Dict[str, Any], price_dec: int, qty_dec: int) -> str:
    parts = [
        f"{rec['event_time']}",
        f"{rec.get('event','')}",
        f"{rec.get('symbol','')}",
        f"{(rec.get('side') or '').upper()}",
        f"qty={rec.get('qty'):.{qty_dec}f}" if rec.get('qty') is not None else "",
        f"filled={rec.get('filled_qty'):.{qty_dec}f}" if rec.get('filled_qty') is not None else "",
        f"lim={rec.get('limit_price'):.{price_dec}f}" if rec.get('limit_price') is not None else "",
        f"avg={rec.get('avg_fill_price'):.{price_dec}f}" if rec.get('avg_fill_price') is not None else "",
        f"last_fill={rec.get('last_fill_price'):.{price_dec}f}" if rec.get('last_fill_price') is not None else "",
        f"status={rec.get('status') or ''}",
        f"oid={rec.get('order_id') or ''}",
    ]
    txt = " | ".join([p for p in parts if p])
    return f"[order] {txt}"

# -------------------------- main runner --------------------------

class Listener:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self._stop = asyncio.Event()
        self._dedupe_set: Set[Tuple[str, str, str]] = set()
        self._dedupe_fifo: deque = deque(maxlen=2000)
        self._db: Optional[DBWriter] = None
        if cfg.db_dsn:
            if not _HAS_DB:
                print("[warn] DB_DSN set but psycopg2 not installed; DB logging disabled.", flush=True)
            else:
                try:
                    self._db = DBWriter(cfg.db_dsn, cfg.db_table)
                    # lazy connect on first write
                except Exception as e:
                    print(f"[warn] could not init DB writer: {e}", flush=True)

    def stop(self):
        self._stop.set()

    def _dedupe_key(self, rec: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
        oid = rec.get("order_id")
        ev = rec.get("event")
        ts = rec.get("event_time")
        if not (oid and ev and ts):
            return None
        return (str(oid), str(ev), str(ts))

    def _is_filtered(self, rec: Dict[str, Any]) -> bool:
        if not self.cfg.sym_filter:
            return False
        sym = rec.get("symbol")
        return sym and sym not in self.cfg.sym_filter

    async def handle_update(self, data: Any):
        try:
            rec = parse_update(data, self.cfg.price_dec, self.cfg.qty_dec)
        except Exception as e:
            print(f"[parse-error] {e} raw={data}", flush=True)
            return

        if self._is_filtered(rec):
            return

        k = self._dedupe_key(rec)
        if k:
            if k in self._dedupe_set:
                return
            self._dedupe_set.add(k)
            self._dedupe_fifo.append(k)
            if len(self._dedupe_fifo) == self._dedupe_fifo.maxlen:
                old = self._dedupe_fifo.popleft()
                self._dedupe_set.discard(old)

        # log
        if self.cfg.log_json:
            print(json.dumps(rec, separators=(",", ":"), ensure_ascii=False), flush=True)
        else:
            print(format_pretty(rec, self.cfg.price_dec, self.cfg.qty_dec), flush=True)

        # persist
        if self._db:
            try:
                self._db.write(rec)
            except Exception as e:
                print(f"[db-error] {e}", flush=True)

    async def run_forever(self):
        key = env_str("ALPACA_API_KEY_ID", os.getenv("ALPACA_API_KEY", ""))
        sec = env_str("ALPACA_API_SECRET_KEY", os.getenv("ALPACA_API_SECRET", ""))
        if not key or not sec:
            print("[fatal] Missing Alpaca API keys (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY).", flush=True)
            return

        backoff = 1
        while not self._stop.is_set():
            ts = TradingStream(key, sec, paper=self.cfg.paper)
            try:
                ts.subscribe_trade_updates(self.handle_update)
                print(f"=== Order Listener === Paper={self.cfg.paper} JSON={self.cfg.log_json} "
                      f"Filter={sorted(self.cfg.sym_filter) if self.cfg.sym_filter else 'ALL'} "
                      f"DB={'ON' if self._db else 'OFF'}", flush=True)
                # In some alpaca-py versions .run() calls asyncio.run(), which
                # explodes if we're already inside an event loop. Prefer the
                # internal coroutine if available; otherwise run the wrapper in a thread.
                if hasattr(ts, "_run_forever"):
                    await ts._run_forever()
                else:
                    # Fallback for versions where .run() is sync:
                    await asyncio.to_thread(ts.run)
            except Exception as e:
                print(f"[stream-error] {e}", flush=True)
                # backoff and retry unless stopping
                if self._stop.is_set():
                    break
                sleep_s = min(backoff, self.cfg.reconnect_max_sec)
                print(f"[reconnect] sleeping {sleep_s}s before retry…", flush=True)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=sleep_s)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, self.cfg.reconnect_max_sec)
            finally:
                try:
                    await ts.stop()
                except Exception:
                    pass

# ------------------------------ entry ------------------------------

def main():
    cfg = build_cfg()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    listener = Listener(cfg)

    # graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, listener.stop)
        except NotImplementedError:
            # Windows
            pass

    try:
        loop.run_until_complete(listener.run_forever())
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()
        print("bye 👋", flush=True)

if __name__ == "__main__":
    main()
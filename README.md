# Algorithmic Trading for Reversion and Trend-Following
**Backtest · Pseudo‑Live Execution · Risk Controls · Dashboard**

A compact, production‑minded **research → execution** pipeline for a single crypto symbol (default: `BTC/USD`) on 1–5 minute bars. It includes a path‑consistent **backtester**, a conservative **pseudo‑live** executor (paper by default), real‑time **data guard**, **order listener**, **risk worker**, and a Streamlit **dashboard**. 

Author Info:

- Kecheng Shi, ks4327@columbia.edu, (212) 814-1023
- Please contact me with any questions, concerns, or recommendations

---

## Repository Layout

```
.
├── .env                        # Environment variables 
├── app/
│   ├── dashboard/
│   │   └── Home.py            # Streamlit UI
│   ├── part1_backtest.py      # Vectorized backtester
│   ├── part2_pseudo_live.py   # Pseudo‑live trader (paper by default)
│   ├── part3_data_guard.py    # Data integrity/freshness checks
│   ├── part3_order_listener.py# Order/fill stream listener
│   └── part3_risk_worker.py   # VaR / exposure / drawdown guard
├── docker/
│   └── sql/                   # Optional DB init scripts
│       └── 000_init.sql
├── docker-compose.yml         # Orchestration (db, app services, dashboard)
├── Dockerfile                 # Python 3.11‑slim base + deps
├── out/                       # Run artifacts (CSVs, logs, figures)
└── requirements.txt           # Pinned Python dependencies
```

---

## System Overview

- **Strategies**
  - **Trend (EMA–ADX)** with optional price‑action gate.
  - **Reversion (mid‑price z‑score)** with ADX/RSI gates.
- **Execution** — limit‑only routing with offset & hysteresis; idempotent opens; cancel/reprice of stale orders.
- **Controls** — out‑of‑process risk worker (VaR/exposure/drawdown) and a data‑quality guard (spreads, jumps, gaps, freshness).
- **Reproducibility** — pinned deps, UTC timestamps, consistent symbol normalization (`BTC/USD` ↔ `BTCUSD`).

### Architecture (Compose services)

```
[db] ←→ [backtest]         # batch
  ↑        ↑
  │        └── writes results to ./out for the dashboard
  │
  └── [live_trader] ←→ [order_listener] ←→ [risk_worker] ←→ [data_guard]
                                   │
                                   └── Streamlit [dashboard] at http://localhost:8501
```

---

## Quickstart (Docker Compose)

**Prereqs**: Docker 24+, Compose v2. The Compose file expects the paths shown above.

1) **Configure** — create `.env` in repo root (see **Configuration**).  
2) **Build & start DB**  
```bash
docker compose build
docker compose up -d db
```
3) **Run a backtest**  
```bash
docker compose run --rm backtest
```
4) **Bring up the dashboard**  
```bash
docker compose up -d dashboard
# open http://localhost:8501
```
5) **Pseudo‑live paper trading** (optional)  
```bash
docker compose up -d data_guard risk_worker order_listener live_trader
```

Tail logs:
```bash
docker compose logs -f backtest
docker compose logs -f live_trader order_listener risk_worker data_guard dashboard
```

---

## Local Development (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ALPACA_API_KEY_ID=... ALPACA_API_SECRET_KEY=...
python app/part1_backtest.py
python app/part2_pseudo_live.py
streamlit run app/dashboard/Home.py  # http://localhost:8501
```

---
Here’s an updated, copy-pasteable markdown of your **Configuration (.env)**.  
_Note: the “Default” column below shows the **configured value from your .env**. “—” means unset. Secrets are masked._

## Configuration (.env)

All services read from `.env`. Unset values fall back to safe defaults.

### Broker
| Variable | Description | Default |
|---|---|---|
| `ALPACA_API_KEY_ID` | Alpaca API key | `PK3769…N854` |
| `ALPACA_API_SECRET_KEY` | Alpaca API secret | `••••••••••••••••••••••••••••••••` |
| `ALPACA_PAPER` | Use paper endpoint if `1` | `1` |
| `APCA_API_BASE_URL` | Override base URL (rare) | — |

### Database
Use either a DSN or discrete fields.
| Variable | Description | Default |
|---|---|---|
| `DB_DSN` / `DATABASE_URL` | `postgresql://user:pass@host:5432/trading` | `postgresql://postgres:postgres@db:5432/trading` |
| `DB_HOST` | Hostname | `db` |
| `DB_PORT` | Port | `5432` |
| `DB_NAME` | Database | `trading` |
| `DB_USER` | User | `postgres` |
| `DB_PASSWORD` | Password | `postgres` |

### Global / Common
| Variable | Description | Default |
|---|---|---|
| `SYMBOL` | `BTC/USD` or `BTCUSD` | `BTC/USD` |
| `ASSET_CLASS` | Asset class | `CRYPTO` |
| `PRICE_DEC` | Price decimals | `2` |
| `QTY_DEC` | Quantity decimals | `6` |
| `TZ` | Timezone | `America/New_York` |
| `TF_MIN` | Bar size in minutes (global) | `1` |

### Strategy Selection
| Variable | Description | Default |
|---|---|---|
| `STRATEGY` | `ema` (trend) or `zscore` (reversion) | `ema` |

#### EMA / Trend parameters
| Variable | Meaning | Default |
|---|---|---|
| `EMA_FAST` | Fast EMA | `13` |
| `EMA_SLOW` | Slow EMA | `150` |
| `ADX_LEN`  | ADX length | `7` |
| `ADX_MIN`  | Minimum ADX to allow longs | `26` |
| `USE_PA_FILTER` | Price-action body filter on/off | `1` |
| `PA_MIN_BODY_FRAC` | Body fraction threshold | `0.5` |

#### Z-score / Reversion parameters
| Variable | Meaning | Default |
|---|---|---|
| `Z_N` | Lookback window | `288` |
| `Z_ENTRY` | Enter if `z <= -Z_ENTRY` | `2.4` |
| `Z_EXIT` | Exit if `z >= -Z_EXIT` | `0.0` |
| `Z_STOP` | Hard stop on `|z|` | `3.5` |
| `RSI_LEN` | RSI length | `14` |
| `RSI_LOWER` / `RSI_UPPER` | Gate thresholds | `30` / `70` |
| `ADX_SKIP` | Skip entries if ADX above this | `30` |
| `Z_ADX_LEN` | ADX length used by Z-score gates | `14` |

### Executor (pseudo-live)
| Variable | Meaning | Default |
|---|---|---|
| `DEFAULT_TIF` | Time in force | `GTC` |
| `LIMIT_AWAY_PCT` | Place buy below / sell above mid by this % | `0.005` |
| `BAR_LOOKBACK_MIN` | Historical bars to consider | `240` |
| `USD_NOTIONAL_CAP` | Per-order notional cap | `500000` |
| `MAX_QTY_PER_TRADE` | Absolute max quantity per order | `99999999` |
| `CANCEL_AFTER_SEC` | Cancel age for working orders | `300` |
| `CANCEL_ON_FLIP` | Cancel opposite-side orders on signal flip | `1` |
| `MIN_NOTIONAL_USD` | Skip dust orders | `10` |
| `DRY_RUN` | Print only (no orders) | `0` |
| `REPRICE_PCT` | Reprice threshold | — |
| `MIN_REPRICE_SEC` / `MAX_LIFETIME_SEC` | Reprice cadence / max age | — / — |
| `USD_CASH_RESERVE` | Leave reserve cash | — |
| `POLL_SEC` | Loop sleep (optional override) | — |

### Backtest
| Variable | Meaning | Default |
|---|---|---|
| `DURATION_DAYS` | Backtest span (days) | `365` |
| `TF_MIN` | Bar size in minutes (backtest override) | `5` |
| `INITIAL_CASH` | Starting cash | `100000` |
| `FEE_BPS` | Fee per notional (e.g., `0.10%`) | `0.1` |
| `SLIPPAGE_BPS` | Slippage per side (e.g., `0.02%`) | `0.02` |
| `START` / `END` | Optional ISO8601 bounds | — / — |

### Data Guard (crypto-friendly)
| Variable | Meaning | Default |
|---|---|---|
| `LOOKBACK_MIN` | Minutes to assess data quality | `120` |
| `GAP_TOL_MIN` | Allow up to N consecutive missing minutes | `10` |
| `GAP_RECENT_MIN` | Only error if long gap within last N minutes | `60` |
| `GAP_PCT_WARN` | Warn if >N% of minutes missing | `10` |
| `GAP_PCT_FAIL` | Error if >N% missing | `20` |
| `TRADE_STALE_SEC` | Latest trade age tolerated | `1800` |
| `BARS_FRESH_SEC` | Bars considered fresh within | `180` |
| `SPREAD_BPS_CAP` | Max allowed spread (bps) | `50` |
| `JUMP_WARN_BPS` / `JUMP_FAIL_BPS` | Bar jump thresholds | `300` / `1500` |
| `ONE_SHOT` | Run once then exit | `0` |
| `STRICT` | Treat warnings as errors | `0` |
| `POLL_SEC` | Loop sleep (optional override) | — |

### Order Listener
| Variable | Meaning | Default |
|---|---|---|
| `BACKFILL_HOURS` | Fill history on start | `24` |
| `SAFETY_OVERLAP_SEC` | Overlap window to avoid misses | `300` |
| `SINK_JSONL` / `SINK_CSV` | Write sinks | `1` / `0` |
| `COST_RATE_BPS` | Used if broker fees aren’t provided | `10` |

### Risk Worker
| Variable | Meaning | Default |
|---|---|---|
| `VAR_CONF` | Confidence level | `0.95` |
| `VAR_WL` | Minutes of 1-min history | `1440` |
| `VAR_REFRESH_SEC` | Recompute cadence (sec) | `60` |
| `MAX_VAR_PCT` | Flatten if VaR exceeds % equity | `0.20` |
| `MAX_DRAWDOWN_PCT` | Flatten if drawdown exceeds % | `0.30` |
| `MAX_EXPOSURE_USD` | Cap exposure (`0`=disabled) | `0` |
| `CANCEL_OPEN_BEFORE_FLATTEN` | Cancel working orders first | `1` |
| `ENFORCE` | `1` = enforce; `0` = log-only | `1` |

### Sizing / Caps (shared)
| Variable | Meaning | Default |
|---|---|---|
| `MAX_LEVERAGE` | Max leverage | `1.0` |
| `MIN_NOTIONAL_USD` | Skip micro rebalances | `10` |

### Logging
| Variable | Meaning | Default |
|---|---|---|
| `LOG_LEVEL` | Log verbosity | — |

---

## Running & Inspecting Results

- **Backtest outputs** land in `./out/` (e.g., `equity_*.csv`, `trades_*.csv`, `meta_*.json`).  
- **Dashboard** (http://localhost:8501) can load these runs, plot equity/drawdown, and export figures.  
- **Pseudo‑live logs** include order intents, placements, reprices, cancels, fills, and risk guard actions.

---

## Quick Tour

- **Signals & positions** — `app/part1_backtest.py` (strategy definitions, slippage/fees, position accounting).  
- **Execution** — `app/part2_pseudo_live.py` (limit‑only routing, hysteresis, idempotence).  
- **Controls** — `app/part3_risk_worker.py` (VaR & drawdown), `app/part3_data_guard.py` (freshness/spreads/gaps).  
- **UI** — `app/dashboard/Home.py` (run selection, charts).  
- **Infra** — `Dockerfile`, `docker-compose.yml` (single image; services share the same codebase).

---

## Design Notes

- **Determinism**: UTC‑indexed bars; same preprocessing across backtest and pseudo‑live.  
- **Safety by default**: paper trading unless `ALPACA_PAPER=0`; limit‑only orders and small offsets; explicit risk gates.  
- **Seams for extension**: add a new strategy by introducing a module and mapping it via `STRATEGY` in `.env`.

---

## Troubleshooting

- **`db` unhealthy** — ensure port `5432` free; check `docker compose logs db`.  
- **No dashboard data** — run a backtest first or point dashboard to an existing run in `./out/`.  
- **Authentication errors** — verify `ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY` and `ALPACA_PAPER=1` for paper keys.

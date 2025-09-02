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

## Configuration (.env)

All services read from `.env`. Unset values fall back to safe defaults.

### Broker
| Variable | Description | Default |
|---|---|---|
| `ALPACA_API_KEY_ID` | Alpaca API key | — |
| `ALPACA_API_SECRET_KEY` | Alpaca API secret | — |
| `ALPACA_PAPER` | Use paper endpoint if `1` | `1` |
| `APCA_API_BASE_URL` | Override base URL (rare) | — |

### Database
Use either a DSN or discrete fields.
| Variable | Description | Default |
|---|---|---|
| `DB_DSN` / `DATABASE_URL` | `postgresql+psycopg2://user:pass@host:5432/trading` | — |
| `DB_HOST` | Hostname | `db` |
| `DB_PORT` | Port | `5432` |
| `DB_NAME` | Database | `trading` |
| `DB_USER` | User | `postgres` |
| `DB_PASSWORD` | Password | `postgres` |

### Symbol & Bar Size
| Variable | Description | Default |
|---|---|---|
| `SYMBOL` | `BTC/USD` or `BTCUSD` | `BTC/USD` |
| `TF_MIN` | Bar size in minutes | `5` |

### Strategy Selection
| Variable | Description | Default |
|---|---|---|
| `STRATEGY` | `ema` (trend) or `zscore` (reversion) | `ema` |

**EMA / Trend parameters**
| Variable | Meaning | Default |
|---|---|---|
| `EMA_FAST` | Fast EMA | `15` |
| `EMA_SLOW` | Slow EMA | `100` |
| `ADX_LEN`  | ADX length | `14` |
| `ADX_MIN`  | Minimum ADX to allow longs | `20` |
| `USE_PA_FILTER` | Price‑action body filter on/off | `1` |
| `PA_MIN_BODY_FRAC` | Body fraction threshold | `0.5` |

**Z‑score / Reversion parameters**
| Variable | Meaning | Default |
|---|---|---|
| `Z_N` | Lookback window | `60` |
| `Z_ENTRY` | Enter if `z <= -Z_ENTRY` | `1.0` |
| `Z_EXIT` | Exit if `z >= -Z_EXIT` | `0.5` |
| `Z_STOP` | Hard stop on `|z|` | `3.0` |
| `RSI_LEN` | RSI length | `14` |
| `RSI_LOWER` / `RSI_UPPER` | Gate thresholds | `30` / `70` |
| `ADX_SKIP` | Skip entries if ADX above this | `30` |

### Executor (pseudo‑live)
| Variable | Meaning | Typical |
|---|---|---|
| `DEFAULT_TIF` | Time in force | `gtc` |
| `LIMIT_AWAY_PCT` | Place buy below / sell above mid by this % | 0.05–0.20 |
| `REPRICE_PCT` | Reprice threshold | 0.05–0.20 |
| `MIN_REPRICE_SEC` / `MAX_LIFETIME_SEC` | Reprice cadence / cancel age | 15 / 300 |
| `CANCEL_ON_FLIP` | Cancel opposite‑side orders on signal flip | 1 |
| `USD_NOTIONAL_CAP` | Per‑order notional cap | 1000 |
| `MIN_NOTIONAL_USD` | Skip dust orders | 5 |
| `USD_CASH_RESERVE` | Leave reserve cash | 5 |
| `DRY_RUN` | Print only (no orders) | 0 |

### Risk Worker
| Variable | Meaning | Default |
|---|---|---|
| `VAR_CONF` | Confidence level | 0.95 |
| `VAR_WL` | Minutes of 1‑min history | 1440 |
| `VAR_REFRESH_SEC` | Recompute cadence | 60 |
| `MAX_VAR_PCT` | Flatten if VaR exceeds % equity | — |
| `MAX_DRAWDOWN_PCT` | Flatten if drawdown exceeds % | — |
| `MAX_EXPOSURE_USD` | Cap exposure | — |
| `CANCEL_OPEN_BEFORE_FLATTEN` | Cancel working orders first | 1 |
| `ENFORCE` | 1 = enforce; 0 = log‑only | 0 |

### Data Guard
| Variable | Meaning | Typical |
|---|---|---|
| `POLL_SEC` | Loop sleep | 30 |
| `STRICT` | Treat warnings as errors | 0 |
| `MAX_SPREAD_BPS` | Spread threshold | 25–100 |
| `JUMP_WARN_BPS` / `JUMP_FAIL_BPS` | Bar jump thresholds | 300 / 1500 |

---


---

## Active configuration from `.env` (detected on 2025‑09‑02)

> Secrets are redacted (•••). Only variables present in your `.env` are shown.


### Broker

| Variable | Value |
|---|---|

| `ALPACA_API_KEY_ID` | `•••N854` |
| `ALPACA_API_SECRET_KEY` | `•••O8du` |
| `ALPACA_PAPER` | `1` |


### Database

| Variable | Value |
|---|---|

| `DB_DSN` | `postgresql://p…:•••@db:5432/trading` |
| `DB_HOST` | `db` |
| `DB_PORT` | `5432` |
| `DB_NAME` | `trading` |
| `DB_USER` | `postgres` |
| `DB_PASSWORD` | `•••gres` |


### Symbol & Bar Size

| Variable | Value |
|---|---|

| `SYMBOL` | `BTC/USD` |
| `TF_MIN` | `5` |


### Strategy Selection

| Variable | Value |
|---|---|

| `STRATEGY` | `ema` |


### EMA / Trend parameters

| Variable | Value |
|---|---|

| `EMA_FAST` | `13` |
| `EMA_SLOW` | `150` |
| `ADX_LEN` | `7` |
| `ADX_MIN` | `26` |
| `USE_PA_FILTER` | `1` |


### Z‑score / Reversion parameters

| Variable | Value |
|---|---|

| `Z_N` | `288` |
| `Z_ENTRY` | `2.4` |
| `Z_EXIT` | `0.0` |
| `Z_STOP` | `3.5` |
| `RSI_LEN` | `14` |
| `RSI_LOWER` | `30` |
| `RSI_UPPER` | `70` |
| `ADX_SKIP` | `30` |


### Executor (pseudo‑live)

| Variable | Value |
|---|---|

| `DEFAULT_TIF` | `GTC                 # crypto 24x7` |
| `LIMIT_AWAY_PCT` | `0.005            # 0.5% away from mid-price` |
| `CANCEL_ON_FLIP` | `1` |
| `USD_NOTIONAL_CAP` | `500000` |
| `MIN_NOTIONAL_USD` | `10             # skip micro rebalances (< $10 notional)` |
| `DRY_RUN` | `0                       # 1=log only; set 0 to actually trade` |


### Risk Worker

| Variable | Value |
|---|---|

| `VAR_CONF` | `0.95                   # one-sided` |
| `VAR_WL` | `1440                     # minutes of 1-min bars (1 day)` |
| `VAR_REFRESH_SEC` | `60` |
| `MAX_VAR_PCT` | `0.20                # VaR as % of equity` |
| `MAX_DRAWDOWN_PCT` | `0.30` |
| `MAX_EXPOSURE_USD` | `0              # 0=disabled` |
| `CANCEL_OPEN_BEFORE_FLATTEN` | `1` |
| `ENFORCE` | `1                       # 1=flatten on breach` |


### Data Guard

| Variable | Value |
|---|---|

| `STRICT` | `0` |
| `JUMP_WARN_BPS` | `300               # 3%` |
| `JUMP_FAIL_BPS` | `1500              # 15%` |

## Running & Inspecting Results

- **Backtest outputs** land in `./out/` (e.g., `equity_*.csv`, `trades_*.csv`, `meta_*.json`).  
- **Dashboard** (http://localhost:8501) can load these runs, plot equity/drawdown, and export figures.  
- **Pseudo‑live logs** include order intents, placements, reprices, cancels, fills, and risk guard actions.

---

## Interviewer’s Quick Tour

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

---

## License

Add a license (e.g., MIT) and a `LICENSE` file in the repo root.

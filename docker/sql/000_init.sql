-- 000_init.sql
-- Safe to run multiple times. Works on vanilla Postgres; upgrades to Timescale if present.

-- === Core table ===
CREATE TABLE IF NOT EXISTS public.fills (
    id         BIGSERIAL PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol     TEXT        NOT NULL,
    side       TEXT        NOT NULL CHECK (
                  lower(side) IN ('buy','sell','b','s','buy_to_cover','sell_short')
               ),
    qty        NUMERIC(38,10) NOT NULL CHECK (qty > 0),
    price      NUMERIC(38,10) NOT NULL CHECK (price >= 0),
    venue      TEXT,
    liquidity  TEXT CHECK (lower(liquidity) IN ('maker','taker','m','t')),
    order_id   TEXT,
    UNIQUE (ts, symbol, side, qty, price, order_id)
);

-- Useful indexes for your UI and queries
CREATE INDEX IF NOT EXISTS idx_fills_ts_desc     ON public.fills (ts DESC);
CREATE INDEX IF NOT EXISTS idx_fills_symbol_ts   ON public.fills (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_fills_order_id    ON public.fills (order_id);

-- If an 'orders' table exists, add FK (idempotent)
DO $$
BEGIN
    IF to_regclass('public.orders') IS NOT NULL THEN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            WHERE t.relname = 'fills' AND c.conname = 'fills_order_id_fkey'
        ) THEN
            EXECUTE '
                ALTER TABLE public.fills
                ADD CONSTRAINT fills_order_id_fkey
                FOREIGN KEY (order_id) REFERENCES public.orders(id) ON DELETE SET NULL
            ';
        END IF;
    END IF;
END
$$;

-- Try to enable TimescaleDB; ignore if not installed/allowed
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        BEGIN
            EXECUTE 'CREATE EXTENSION timescaledb';
        EXCEPTION
            WHEN undefined_file OR invalid_parameter_value OR insufficient_privilege THEN
                RAISE NOTICE 'TimescaleDB not available; continuing with plain table.';
        END;
    END IF;
END
$$;

-- Convert to hypertable when TimescaleDB is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM public.create_hypertable('fills', 'ts', if_not_exists => TRUE);

        -- Optional: set a compression policy if supported by your Timescale version
        BEGIN
            PERFORM add_compression_policy('fills', INTERVAL '7 days');
        EXCEPTION
            WHEN undefined_function THEN
                -- Older Timescale version without add_compression_policy()
                NULL;
        END;
    END IF;
END
$$;

-- === public.fills schema (idempotent) =======================================
BEGIN;

-- Ensure schema exists
CREATE SCHEMA IF NOT EXISTS public;

-- Create table if it doesn't exist (with sane defaults)
CREATE TABLE IF NOT EXISTS public.fills (
  id                BIGSERIAL PRIMARY KEY,
  ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
  symbol            TEXT        NOT NULL,
  side              TEXT        NOT NULL,
  qty               NUMERIC(38,10) NOT NULL,
  price             NUMERIC(38,10) NOT NULL,
  notional          NUMERIC(38,10) NOT NULL DEFAULT 0,  -- qty*price if provided; default 0 so inserts don’t fail
  fee               NUMERIC(38,10) NOT NULL DEFAULT 0,  -- default 0 if not provided
  venue             TEXT,
  liquidity         BOOLEAN,
  exchange          TEXT,
  order_id          TEXT        NOT NULL,
  client_order_id   TEXT,
  seq               INT         NOT NULL DEFAULT 0,
  "time"            TIMESTAMPTZ DEFAULT now(),
  "timestamp"       TIMESTAMPTZ DEFAULT now(),
  dt                TIMESTAMPTZ DEFAULT now()
);

-- Add any missing columns (idempotent)
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS id               BIGSERIAL;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS ts               TIMESTAMPTZ;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS symbol           TEXT;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS side             TEXT;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS qty              NUMERIC(38,10);
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS price            NUMERIC(38,10);
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS notional         NUMERIC(38,10);
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS fee              NUMERIC(38,10);
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS venue            TEXT;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS liquidity        BOOLEAN;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS exchange         TEXT;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS order_id         TEXT;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS client_order_id  TEXT;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS seq              INT;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS "time"           TIMESTAMPTZ;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS "timestamp"      TIMESTAMPTZ;
ALTER TABLE public.fills ADD COLUMN IF NOT EXISTS dt               TIMESTAMPTZ;

-- Primary key (if missing). If the table already has a PK, this will fail;
-- so guard it by checking constraint existence.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'public.fills'::regclass
      AND contype = 'p'
  ) THEN
    ALTER TABLE public.fills ADD CONSTRAINT fills_pkey PRIMARY KEY (id);
  END IF;
END$$;

-- Required columns: make sure they have defaults so inserts that omit them succeed
ALTER TABLE public.fills ALTER COLUMN ts          SET DEFAULT now();
ALTER TABLE public.fills ALTER COLUMN "time"      SET DEFAULT now();
ALTER TABLE public.fills ALTER COLUMN "timestamp" SET DEFAULT now();
ALTER TABLE public.fills ALTER COLUMN dt          SET DEFAULT now();

-- Keep these NOT NULL but give safe defaults (prevents the error you saw)
ALTER TABLE public.fills ALTER COLUMN notional SET DEFAULT 0;
ALTER TABLE public.fills ALTER COLUMN fee      SET DEFAULT 0;
ALTER TABLE public.fills ALTER COLUMN seq      SET DEFAULT 0;

-- If notional/fee currently allow NULLs and you want them NOT NULL, backfill then enforce:
UPDATE public.fills SET notional = 0 WHERE notional IS NULL;
UPDATE public.fills SET fee      = 0 WHERE fee IS NULL;

-- Optional: enforce NOT NULL on core fields (comment out if your pipeline may omit them)
ALTER TABLE public.fills ALTER COLUMN symbol SET NOT NULL;
ALTER TABLE public.fills ALTER COLUMN side   SET NOT NULL;
ALTER TABLE public.fills ALTER COLUMN qty    SET NOT NULL;
ALTER TABLE public.fills ALTER COLUMN price  SET NOT NULL;
ALTER TABLE public.fills ALTER COLUMN order_id SET NOT NULL;
ALTER TABLE public.fills ALTER COLUMN notional SET NOT NULL;
ALTER TABLE public.fills ALTER COLUMN fee      SET NOT NULL;
ALTER TABLE public.fills ALTER COLUMN seq      SET NOT NULL;

-- Unique key for UPSERTs from the order listener
-- (Listener typically does ON CONFLICT (order_id, seq) DO UPDATE ...)
CREATE UNIQUE INDEX IF NOT EXISTS fills_order_seq_uidx
  ON public.fills(order_id, seq);

-- Helpful indexes for dashboard/queries
CREATE INDEX IF NOT EXISTS fills_ts_idx
  ON public.fills(ts DESC);

CREATE INDEX IF NOT EXISTS fills_symbol_ts_idx
  ON public.fills(symbol, ts DESC);

COMMIT;

-- ===================== Smoke test (optional) ================================
-- After running the above, this should succeed without NOT NULL errors:
-- INSERT INTO public.fills (ts,symbol,side,qty,price,venue,liquidity,order_id,client_order_id,seq)
-- VALUES (now(),'BTCUSD','buy',0.001,50000,'paper',true,'TEST-1','TEST-1',1);


BEGIN;

-- Core table the listener writes to
CREATE TABLE IF NOT EXISTS public.fills (
    id               BIGSERIAL PRIMARY KEY,
    ts               TIMESTAMPTZ NOT NULL,
    symbol           TEXT        NOT NULL,
    side             TEXT        NOT NULL CHECK (side IN ('buy','sell')),
    qty              NUMERIC(18,6)  NOT NULL,
    price            NUMERIC(18,8)  NOT NULL,
    order_id         TEXT        NOT NULL,
    seq              INTEGER     NOT NULL DEFAULT 1,
    client_order_id  TEXT,
    liquidity        TEXT,
    venue            TEXT,
    cost             NUMERIC(20,8)  NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (order_id, seq)
);

-- Helpful indexes for typical queries
CREATE INDEX IF NOT EXISTS idx_fills_ts           ON public.fills (ts);
CREATE INDEX IF NOT EXISTS idx_fills_symbol_ts    ON public.fills (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_fills_order        ON public.fills (order_id);

COMMIT;

-- 1) Drop null/zero/negative equity rows
DELETE FROM public.pnl_equity WHERE equity IS NULL OR equity <= 0;

-- 2) (Optional but recommended) Remove isolated collapse spikes:
--    "if an equity value in a minute is <15% of that minute's median, nuke it"
WITH med AS (
  SELECT date_trunc('minute', ts) AS tsm,
         percentile_cont(0.5) WITHIN GROUP (ORDER BY equity) AS med_eq
  FROM public.pnl_equity
  WHERE ts >= now() - interval '30 days'
  GROUP BY 1
)
DELETE FROM public.pnl_equity e
USING med m
WHERE date_trunc('minute', e.ts) = m.tsm
  AND e.equity < 0.15 * m.med_eq;
# 🔱 Trinity (MasterScanner) — NSE/BSE Quant Trading Platform

A production-grade Streamlit application for scanning, scoring, backtesting, and
managing positions across the Nifty 500 universe and its F&O (futures & options)
segment. What began as a Pine Script indicator port has grown into a full
multi-engine platform: a legacy points-based scanner, an SMC-driven options
engine (the **DORE Options Engine**), a Five Pillars ranking system, a
walk-forward backtester, a portfolio exit-management engine, and an
LLM-assisted news intelligence panel — all persisted to Supabase and runnable
unattended via a background scheduler.

> **Naming warning — two different engines are called "DORE".**
> `utils/dore_options_engine.py` (the **DORE Options Engine**) is the live,
> production F&O pipeline and the only one any UI reads. `utils/dore_engine.py`
> (**DORE 2.0**, the older Stage 0–5b design) is retained for reference and
> rollback only — its scheduler jobs are disabled and nothing in the app
> consumes its output. See "DORE" below before changing either.

---

## 📸 What It Does

| Feature | Details |
|---|---|
| **Dashboard** | Landing page — index cards (SENSEX/NIFTY OHLCV + EMA20/50/200 badges), Market Intelligence and F&O Scan panels refreshed on their own `st.fragment` timers, kicks off background scan loops |
| **Live Scanner** | Legacy points-based engine — EMA trend, RSI, volume, breakout, momentum, RS vs Nifty, and CCI signals, plus HTF-momentum "Qualification" gating. **No longer its own nav entry** (2026-09-09) — `pages/scanner.py`'s `render()` is called directly by the Dashboard |
| **Pre-Breakout Scanner (Five Pillars)** | Independent ranking engine — Structure / Acceptance / Leadership / Momentum / Risk pillars (30/25/20/15/10% weights) |
| **DORE Options Engine** | Live F&O options pipeline — SMC-on-futures direction, structural strike anchoring, IV/PCR gating. See below |
| **Sectors** | Sector rotation / relative-strength view (`pages/sectors.py`) |
| **Backtest Engine** | Walk-forward simulation on daily OHLCV with full PnL stats, threaded (not multiprocess) execution |
| **Lifecycle** | Setup lifecycle tracking (Forming → Qualified → Executed → Exited, etc.) |
| **History** | Historical scan/backtest browsing |
| **Portfolio** | Position tracking with a multi-factor exit-scoring engine (stop-loss override, exhaustion, time decay, regime adaptation) |
| **CCI Master** | Standalone CCI-focused signal view |
| **Settings** | Scan / Advanced / System tabs — universe, thresholds, cache management |
| **CV/EQ Validation** | Sandbox page for testing conviction/entry-quality threshold changes without touching production scoring |
| **Diagnostic** | Engine-internals inspection (funnel stage counts, gate trip reasons, etc.) |
| **Agent** | LLM-assisted chat/agent tooling over the app's data (OpenAI) |
| **Data Source Check** | Compares/validates Upstox vs. yfinance data for a given symbol |
| **News Intelligence** | RSS ingestion (Economic Times, Moneycontrol) + Groq LLM sentiment/impact tagging, surfaced on the Dashboard |
| **Dark UI** | Custom dark theme (JetBrains Mono / Syne), colour-coded by score/action/CCI state |

---

## 🗂️ Project Structure

```
MasterScanner/
│
├── app.py                        # Entry point — st.navigation page router, global CSS,
│                                  # root logging setup, background-scheduler kickoff
│
├── pages/
│   ├── dashboard.py               # Landing page, index cards, Market Intelligence / F&O fragments;
│   │                               # also calls pages/scanner.py's render() inline (2026-09-09)
│   ├── scanner.py                 # Live scanner UI + DORE Options tables (Live Scan / Active Plans).
│   │                               # NOT registered as its own st.Page — rendered by dashboard.py
│   ├── sectors.py                 # Sector rotation / relative-strength view
│   ├── five_pillars.py            # Pre-Breakout Scanner (Five Pillars ranking)
│   ├── backtest.py                # Backtest UI + charts + trade log
│   ├── lifecycle.py               # Setup lifecycle tracking
│   ├── history.py                 # Historical scan/backtest browser
│   ├── portfolio.py                # Position tracking + exit scoring UI
│   ├── settings.py                # Scan / Advanced / System settings tabs
│   ├── validation.py              # CV/EQ threshold sandbox (no production impact)
│   ├── diagnostic.py              # Engine-internals / funnel diagnostics
│   ├── agent.py                   # LLM agent tab
│   ├── cci_master.py              # CCI-focused standalone view
│   └── data_source_check.py       # Upstox vs. yfinance data comparison
│
├── utils/                          # ~70 modules — engines, data clients, persistence
│   ├── scanner_engine.py          # Legacy points-based scoring (Pine Script → Python origin)
│   ├── scoring_core.py            # Core scoring/decision logic shared across engines
│   ├── dore_options_engine.py     # ★ THE production F&O engine — compute_dore_trade_plan():
│   │                               # SMC-on-futures direction, structural strike anchoring,
│   │                               # premium/OI/IV/PCR validation, final scoring
│   ├── dore_options_scan.py       # DORE Stage 1 (Technical Plans): fetches spot + futures
│   │                               # OHLCV + option chain, calls the engine once per symbol
│   │                               # per live_scanner cycle (~5min)
│   ├── dore_live_state.py         # DORE Stage 2 (Live Market Refresh): per-plan premium/
│   │                               # quote/entry-trigger refresh every 60s. Never re-decides
│   │                               # direction — see its module docstring for the Stage1/2 split
│   ├── dore_options_persistence.py # Plan minting / entry-lock / close lifecycle + persistence
│   ├── smc_engine.py              # Smart Money Concepts — order blocks, BOS/CHoCH, liquidity
│   │                               # sweeps, FVG; produces SMCState (direction + evidence_tier)
│   ├── structural_levels.py       # Causal pivot series used by SMC/structural targets
│   │
│   │   # ── Legacy DORE 2.0 (reference/rollback only — no UI reads these) ──
│   ├── dore_engine.py             # DORE 2.0 — the older Stage 0–5b Opportunity Engine
│   ├── dore_settings.py           # DORE 2.0's thresholds/weights (NOT the Options Engine's,
│   │                               # which uses DoreOptionsSettings inside dore_options_engine.py)
│   ├── dore_fo_screener.py        # DORE 2.0 Stage 0 universe screener
│   ├── index_dore_job.py          # DORE 2.0 index-level job — scheduler entry disabled
│   ├── market_intelligence.py     # Market Intelligence compute — index snapshot/OI/EMA every
│   │                               # cycle, regime/breadth classification; reads index-level
│   │                               # DORE state from dore_live_state's snapshot rather than
│   │                               # computing it itself (consolidated 2026-08)
│   ├── pillar_engine.py           # Five Pillars ranking engine
│   ├── portfolio_engine.py        # Multi-factor position exit-scoring
│   ├── position_sizing.py         # Capital-aware lot/quantity sizing shared by DORE stages
│   ├── backtest_engine.py         # Walk-forward signal generation + trade simulation
│   ├── decision_engine.py         # Decision-centric data contracts (MarketContext, DecisionTrace, etc.)
│   ├── upstox_client.py           # Upstox auth, instrument resolution, OHLCV/option-chain fetch
│   ├── oi_snapshot_store.py       # OI/premium change-tracking state used by DORE's derivative stages
│   ├── history_store.py           # Two-tier Parquet + Supabase OHLCV cache
│   ├── supabase_client.py         # Supabase read/write helpers, schema, and free-plan retention pruning
│   ├── scan_state.py              # Snapshot/state persistence — see "Data Layer" below for the
│   │                               # 2026-08-04 snapshot-vs-state table migration
│   ├── snapshot_cache.py          # Process-wide, version-keyed st.cache_data layer in front of
│   │                               # scan_state's Supabase reads, for Streamlit-side (page/fragment)
│   │                               # callers only — NOT used by the scheduler or DORE producers,
│   │                               # which need a genuinely fresh read every cycle
│   ├── scan_health_monitor.py     # RAM/CPU self-protection for the background scan loops —
│   │                               # skips a cycle rather than risk OOM on constrained hosts
│   ├── scan_priority.py           # Cross-job coordination so scan loops don't contend mid-cycle
│   ├── news_feed.py / news_sentiment.py  # RSS ingestion + Groq LLM sentiment tagging
│   ├── openai_client.py / groq_client.py # LLM client helpers (fail-soft if unconfigured)
│   ├── inprocess_scheduler.py     # Background scan loops as daemon threads (single-process hosts)
│   └── system_state.py            # Scheduler ownership lock (see "Background Scheduling" below)
│
│   # fo_scan.py — legacy F&O pipeline, superseded by dore_options_scan.py /
│   # dore_live_state.py above (2026-07-31). Its scheduler job is disabled
│   # and its table dropped; kept in-tree for rollback reference only.
│   #
│   # utils/scanner.py — ⚠ STRAY DUPLICATE of pages/scanner.py, committed by
│   # accident in a5a2721 (2026-09-16). Nothing imports it, it is not a
│   # Streamlit page (only pages/ is auto-discovered), and it has already
│   # drifted stale. Edit pages/scanner.py, never this one. Safe to delete.
│
│
├── scheduler/
│   └── scan_worker.py             # Standalone-process scheduler (preferred for multi-process hosts)
│
├── flask_ui/                      # Separate Flask + HTMX + Tailwind UI migration prototype
│                                  # (wraps a frozen snapshot of utils/ — see flask_ui/README.md)
│
├── docs/
│   └── SCORING_SYSTEMS.md
├── dore-3x-trade-construction-design.md
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1 — Clone the repo

```bash
git clone https://github.com/srikanthgodavarthy/MasterScanner.git
cd MasterScanner
```

### 2 — Install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Configure secrets

The app reads credentials from **Streamlit secrets** (`.streamlit/secrets.toml` locally,
or the Secrets UI on Streamlit Community Cloud) and, for the Upstox token, optionally a
local `.env` file. Create `.streamlit/secrets.toml`:

```toml
# Required for persistence (scan snapshots, backtest logs, portfolio, OI/premium history)
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-or-service-role-key"

# Optional — enables the Agent tab
OPENAI_API_KEY = "sk-..."

# Optional — enables News Intelligence sentiment/impact tagging (free tier)
GROQ_API_KEY = "gsk_..."
```

And a `.env` (or environment variable) for Upstox:

```bash
UPSTOX_ACCESS_TOKEN=your-upstox-access-token
```

Every one of these is **optional in the sense that the app fails soft** — Supabase-backed
pages will show a "not configured" message, the Agent tab and News Intelligence panel
simply won't activate, and Upstox-dependent panels (F&O scan, DORE, live option chain)
won't populate — but core scanning/backtesting on yfinance data works without any of them.

> ⚠️ **Upstox token lifecycle:** an Upstox `access_token` expires daily at 3:30 AM IST
> regardless of issue time. This app does not auto-refresh it — re-issue and update the
> token each morning before market hours if you rely on Upstox-backed features.

Run the schema SQL once in **Supabase → SQL Editor** — see `utils/supabase_client.py`
for the current table definitions.

### 4 — Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📐 Scoring Systems

MasterScanner runs **multiple, independent** scoring/ranking engines rather than one
unified score — see `docs/SCORING_SYSTEMS.md` for the full breakdown. At a high level:

- **Legacy Scanner** (`scanner_engine.py` / `scoring_core.py`) — points-based, rooted in
  the original Pine Script port: EMA trend, RSI, volume, breakout, momentum, relative
  strength vs. Nifty, and CCI oversold/overbought/extended states, gated by an HTF-momentum
  "Qualification" layer (1m/3m/6m return thresholds + EMA trend structure).
- **Five Pillars** (`pillar_engine.py`) — Structure (30%) / Acceptance (25%) / Leadership
  (20%) / Momentum (15%, Stochastic + RSI(14)) / Risk (10%).
- **DORE Options Engine** (`dore_options_engine.py`) — see below; entirely independent of
  the above two, sharing only the market-data layer. (`dore_engine.py` / "DORE 2.0" is the
  retired predecessor — not on the live path.)
- **SMC** (`smc_engine.py`) — Smart Money Concepts structural read: order blocks, BOS/CHoCH,
  liquidity sweeps, FVG, producing an `SMCState` with a direction and an `evidence_tier`
  (0–4). Supplies DORE's direction and gates its structural targets.
- **CV4** (`canonical_scores.py`) — the CV4 conviction/entry-quality/leadership scores that
  superseded CV1/CV3 in the Phase 7 cutover.
- **Portfolio exit scoring** (`portfolio_engine.py`) — a separate multi-factor system for
  *when to exit* an existing position, distinct from entry scoring.

Trade levels (Entry / SL / T1-T2-T3) are ATR- and structure-based; see
`utils/trade_levels.py`.

---

## 🎯 DORE — F&O Options Engine

DORE is architecturally independent of the scanners above — it shares **only** the
market-data layer (OHLCV, option chain, symbol master), never scores or classifications.

### Which engine is live

| | **DORE Options Engine** (live) | **DORE 2.0** (retired) |
|---|---|---|
| Module | `utils/dore_options_engine.py` | `utils/dore_engine.py` |
| Driven by | `dore_options_scan.py` (Stage 1) + `dore_live_state.py` (Stage 2) | `fo_scan.py`, `index_dore_job.py` |
| Settings | `DoreOptionsSettings` (in-module) | `utils/dore_settings.py` |
| Scheduler jobs | runs inside `live_scanner` + `dore_live_state` | `fo_scan`, `index_dore` — **both disabled** |
| Read by any UI page? | **Yes** — Live Scan + Active Plans tabs | **No** — nothing consumes its output |

DORE 2.0's Stage 0–5b design (Trend → Execution → Derivative → Option Intelligence →
Risk → Opportunity → Strike/Expiry) is preserved in `dore_engine.py` for reference and
rollback, but its scheduler entries were commented out (2026-09-15) after both jobs were
traced to Upstox option-chain 429s — three concurrent 60s jobs contending for one token's
rate-limit budget, for output nothing reads.

### Live pipeline

```
live_scanner cycle (~5min)
  └─ dore_options_scan.py ── spot OHLCV + futures OHLCV (O/H/L/C) + option chain
        └─ dore_options_engine.compute_dore_trade_plan()
              │
              ├─ Qualification score            (spot momentum — feeds ranking, not direction)
              ├─ DIRECTION (CE/PE):
              │     1. SMC structural read on FUTURES OHLC (order blocks + SMCState)
              │        → decides direction when evidence_tier ≥ smc_direction_min_evidence_tier
              │     2. fallback: EMA9/21 cross (futures-confirmed, else spot)
              │     → recorded as direction_source: SMC-Futures | SMC-Spot | Futures-EMA | Spot-EMA
              ├─ Structural anchor / geometry    (order-block-anchored entry/SL/target, R:R gate)
              ├─ Strike selection                (chain-driven: premium, delta, strike_interval)
              ├─ Premium + OI/liquidity validation (incl. PCR agreement/conflict)
              ├─ IV gates                        (IV-crush hard gate, CE/PE skew caution)
              └─ Final weighted score → OptionTradePlan
        └─ rank_recommendations() → dore_technical_plans snapshot
        └─ dore_options_persistence: mint / entry-lock / close → dore_options_plans

dore_live_state cycle (60s)
  └─ refreshes ONLY market-dependent fields on open plans
     (premium, OI, IV, POP, drift %, entry_trigger_status). Never re-decides direction.
```

**Direction is the only place futures data feeds the decision.** Everything downstream
(structural anchor, strikes, validation, scoring) works off spot price levels and the
option chain, keyed to whichever direction the SMC/EMA step settled on.

### Notable gates

- **SMC-on-Futures direction** (`use_smc_direction`, default on) — the SMC read runs on the
  futures contract's own OHLC, falling back to spot only when no usable futures series is
  available. Fail-soft throughout: a symbol never goes directionless.
- **PCR conflict → forced demotion, indices only** — a PCR that clears the *opposite*
  direction's threshold scores `-20` and is labelled `CONFLICTS` (vs `-10` for genuinely
  ambiguous). On indices that conflict additionally forces `score ≤ 49`, below
  `MIN_CONFIDENCE_TO_TRACK_INDEX`, because the index weight profile already ranks
  OI-quality above conviction (28 vs 12, the reverse of stocks). Stocks keep only the
  softer blended penalty. SMC remains senior for *direction*; PCR never gates direction.
- **IV-crush hard gate** (`enable_iv_crush_hard_gate`, trips at IV rank ≥ 90).
- **CE/PE IV skew caution** (`iv_skew_caution_threshold_pp`, default 3.0pp) — both
  `fetch_stock_atm_option()` (stocks) and `fetch_oi_resistance()` (indices) return
  `ce_iv`/`pe_iv` separately; the index path used to collapse them into one blended
  average and discard the skew.

> **Terminology note:** "Stage 1 / Stage 2" refers to *cadence* (`dore_options_scan.py`
> ~5min vs. `dore_live_state.py` ~60s), not to DORE 2.0's Stage 0–5b *scoring* stages.
> The two numbering schemes are unrelated.

---

## 🧪 Backtest Methodology

- **Data**: yfinance daily OHLCV (`.NS` suffix), pinned to `yfinance==0.2.66`
- **Signal generation**: Walk-forward day-by-day — no look-ahead bias
- **Entry**: Next-bar open after signal
- **Exit priority**: SL hit → T2 hit → T1 hit → timeout after N days
- **Execution**: threaded (not multiprocess) — `ProcessPoolExecutor` deadlocks when forked
  inside Streamlit's multi-threaded runtime on Linux, so `use_processes` defaults to `False`
- **Metrics**: Win rate, avg win/loss, profit factor, expectancy, R:R, per-symbol breakdown
- **Storage**: Trade log saved to Supabase

---

## 🗄️ Data Layer

- **yfinance** — primary historical/daily OHLCV source for the legacy scanner and backtester.
- **Upstox** — live quotes, option chain, and F&O instrument resolution (`utils/upstox_client.py`);
  feeds DORE, the F&O scan panel, Market Intelligence, and the Data Source Check page.
  Note: Upstox's `instrument_type` label for equities is `'EQUITY'`, not `'EQ'` — a past
  source of near-total F&O instrument-key resolution failures if assumed otherwise.
- **`utils/history_store.py`** — two-tier Parquet-on-disk + Supabase cache, purpose-built to
  survive Streamlit Community Cloud's ephemeral filesystem between redeploys. Live in-RAM
  history is trimmed to a 280-bar-per-symbol ring buffer to bound memory growth.
- **Supabase** — persistence for scan snapshots, backtest trade logs, portfolio state, OI/premium
  snapshot history, watchlists, etc. RLS policies gate access; configure via `SUPABASE_URL`/`SUPABASE_KEY`.

  **Snapshot vs. state tables (2026-08-04 migration):** most sections (`market_intelligence`,
  `dore_options_scan`, `dore_technical_plans`, `scan_snapshots`/archive/sector history) are
  still append-only — one new row per producer cycle, pruned on a retention schedule (see
  `utils/supabase_client.py`'s `prune_scan_snapshot_tables()`). `live_scanner` and
  `dore_live_state` are the exception: they used to follow the same append-only pattern but
  hit real Free-plan trouble — `live_scanner_snapshots` alone reached 1.22GB (~98% of the
  whole database) in about 28 hours, since there's only ever one current record per stock
  that matters, not a growing history of full-universe blobs. Both sections now UPSERT one
  row per symbol into a fixed-size `*_state` table instead, keeping table size bounded by
  symbol count rather than uptime — see `utils/scan_state.py`'s `_STATE_SECTIONS` for the
  current mapping and full rationale.

  **Read-side caching:** `utils/snapshot_cache.py` sits in front of `scan_state`'s reads for
  Streamlit-side (page/fragment) callers, keyed on `(section, version)` rather than a fixed
  TTL — the first caller in any browser tab to see a new version pays for the real Supabase
  read, every other caller sharing that version gets the same cached object, and a genuine
  version bump is never served stale. Deliberately **not** used by the scheduler or by DORE
  Stage 1/2 producer code, which need a fresh read every cycle regardless of what any
  browser tab has cached.

  **Free-plan discipline:** the project runs on Supabase's Free plan (500 MB database cap),
  so every insert-only table needs an explicit retention cap — see
  `utils/supabase_client.py`'s `prune_scan_snapshot_tables()` and the `prune_snapshot_table`/
  `prune_backtest_results` Postgres functions it calls. `backtest_results` in particular is
  pruned by **run count** (most recent N `run_at` values), not row count, since a plain
  row-count cap could silently truncate the older half of the oldest surviving run.

---

## ⏱️ Background Scheduling

Market Intelligence (~180s), DORE Stage 2 / Live Market Refresh (~60s, includes index-level
DORE compute for NIFTY/SENSEX/BANKNIFTY), and the Live Scanner (~5min, batched, with
DORE Stage 1 Technical Plans produced once per Live Scanner cycle) run as background loops,
coordinated by **one of two interchangeable mechanisms**:

- `scheduler/scan_worker.py` — a standalone process (`python -m scheduler.scan_worker`),
  the preferred setup for hosts that can run a second process.
- `utils/inprocess_scheduler.py` — the same jobs run as daemon threads inside the Streamlit
  process itself, for single-process-only hosts (e.g. Streamlit Community Cloud). Started
  automatically from `app.py` after the first page's initial synchronous render.

Both coordinate through an ownership lock in `utils/system_state.py`, so accidentally
running both in the same deployment is safe (one claims the lock, the other backs off)
rather than silently double-executing every job.

> **Note:** the two legacy DORE 2.0 jobs — `fo_scan` and `index_dore` (both ~60s) — are
> **disabled** in `scheduler/scan_worker.py`'s `JOBS` list. Briefly restored 2026-09-09,
> they were commented back out on 2026-09-15: nothing in the UI reads either one, yet both
> hit Upstox's option-chain endpoint every 60s, drawing on the *same* per-token rate-limit
> budget as `dore_live_state` (the actually-live pipeline). Traced from a `scan_health_monitor`
> RAM warning plus a burst of 429s (NIFTY rate-limited 6× in 12s, several stocks dropped after
> exhausting retries). Their compute functions are left defined for a future restore.
>
> Market Intelligence's cadence was slowed from an original 30s to 180s (2026-07-25) once
> profiling showed the 30s interval was doing 3-4 full index OHLCV/OI fetches per call —
> far more frequently than that data actually changes.
>
> **Market-hours pausing** goes through `utils/system_state.py`'s `market_hours_pause_active()`
> (2026-09-09), which honours the Settings toggle — not a bare wall-clock
> `is_market_hours_ist()` call. Use it for any new pause check.

**Self-protection:** both loop mechanisms check `utils/scan_health_monitor.py` before each
cycle and skip it (rather than risk an OOM kill) if resident memory or CPU is over a
configured threshold — see that module for current `RAM_WARN_MB`/`RAM_CRITICAL_MB` values,
which are tuned for Streamlit Community Cloud's free-tier container ceiling and may need
raising/lowering on a different host.

---

## 🌐 Deployment

### Streamlit Community Cloud (free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file**: `app.py`
4. Add secrets in **Advanced settings → Secrets** (paste your `secrets.toml` contents)
5. Background scans run in-process automatically (`utils/inprocess_scheduler.py`) — no
   second process needed on this host.
6. In the same **Advanced settings**, also add as an environment variable (not in
   `secrets.toml` — this one just needs to exist in the process environment):
   ```
   MALLOC_ARENA_MAX=2
   ```
   Caps glibc's malloc arena count so the background scan threads' allocate/free churn
   doesn't fragment memory the way it did pre-2026-08-13 (see `utils/scan_health_monitor.py`'s
   `_malloc_trim_reclaim()` docstring). **Must be set before the process starts** — setting it
   from inside `app.py` is too late, since pandas/numpy/streamlit already trigger glibc's
   first arena allocation during import, before any of this app's own code runs. Requires a
   full app reboot (not just a rerun) to take effect. `2` was chosen against this app's actual
   background concurrency — `LIVE_SCANNER_MAX_WORKERS=4` is the sustained hot path
   (`scheduler/scan_worker.py`); `1` would fully serialize those 4 threads on a single malloc
   lock, `2` caps fragmentation while still giving them two lanes. If scan batch durations
   creep up after enabling this (`[live_scanner] batch N/10 done: Xs` in the logs), try `4`
   instead — the memory profiler's RSS/malloc_trim logging already gives you before/after
   evidence either way.

### Self-hosted / VPS (with a standalone scheduler process)

```bash
export MALLOC_ARENA_MAX=2   # see note above — must be set before either process starts
streamlit run app.py --server.port 8501 --server.headless true
python -m scheduler.scan_worker   # optional: run as a separate always-on process
```

Use **nginx** as a reverse proxy and **systemd** or **pm2** to keep both processes running.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit>=1.35.0` | Web app framework (`st.navigation`/`st.Page` router, `st.fragment` auto-refresh) |
| `yfinance==0.2.66` | NSE OHLCV data (Yahoo Finance) — pinned; see comment in `requirements.txt` before bumping |
| `pandas==2.2.3` / `numpy==1.26.4` | Data manipulation |
| `plotly>=5.22.0` | Interactive charts in Backtest |
| `supabase>=2.5.0` | Database client |
| `python-dotenv>=1.0.0` | Loads `UPSTOX_ACCESS_TOKEN` from local `.env` |
| `requests>=2.31.0` | Upstox REST calls |
| `numba==0.59.1` | JIT-accelerated numeric routines |
| `scipy` | Statistical/numeric utilities |
| `openai>=1.40.0` | Agent tab (OpenAI) and Groq client (OpenAI-SDK-compatible) |
| `pyarrow>=15.0.0` | Parquet read/write for `utils/history_store.py` |
| `feedparser>=6.0.0` | RSS ingestion for `utils/news_feed.py` (ET / Moneycontrol) |
| `psutil>=5.9.0` | RAM/CPU checks for `utils/scan_health_monitor.py`'s self-protecting scan loop |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute
financial advice. Always do your own research before making any investment decisions.
Past backtest performance does not guarantee future results.

---

## 📄 License

MIT License — free to use, modify, and distribute.

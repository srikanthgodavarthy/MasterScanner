# 🔱 Trinity (MasterScanner) — NSE/BSE Quant Trading Platform

A production-grade Streamlit application for scanning, scoring, backtesting, and
managing positions across the Nifty 500 universe and its F&O (futures & options)
segment. What began as a Pine Script indicator port has grown into a full
multi-engine platform: a legacy points-based scanner, an independent F&O
opportunity engine (DORE 2.0), a Five Pillars ranking system, a walk-forward
backtester, a portfolio exit-management engine, and an LLM-assisted news
intelligence panel — all persisted to Supabase and runnable unattended via a
background scheduler.

---

## 📸 What It Does

| Feature | Details |
|---|---|
| **Dashboard** | Landing page — index cards (SENSEX/NIFTY OHLCV + EMA20/50/200 badges), Market Intelligence and F&O Scan panels refreshed on their own `st.fragment` timers, kicks off background scan loops |
| **Live Scanner** | Legacy points-based engine — EMA trend, RSI, volume, breakout, momentum, RS vs Nifty, and CCI signals, plus HTF-momentum "Qualification" gating |
| **Pre-Breakout Scanner (Five Pillars)** | Independent ranking engine — Structure / Acceptance / Leadership / Momentum / Risk pillars (30/25/20/15/10% weights) |
| **DORE 2.0** | Architecturally independent F&O Opportunity Engine — see below |
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
│   ├── dashboard.py               # Landing page, index cards, Market Intelligence / F&O fragments
│   ├── scanner.py                 # Legacy live scanner UI + table rendering
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
│   ├── dore_engine.py             # DORE 2.0 — independent F&O Opportunity Engine
│   ├── dore_settings.py           # All DORE thresholds/weights (nothing hardcoded in the engine)
│   ├── dore_fo_screener.py        # DORE Stage 0 universe screener
│   ├── dore_options_scan.py       # DORE two-stage integration — Stage 1 (Technical Plans):
│   │                               # qualification/direction/strike selection, once per
│   │                               # live_scanner cycle (~5min)
│   ├── dore_live_state.py         # DORE two-stage integration — Stage 2 (Live Market
│   │                               # Refresh): per-plan premium/quote refresh AND index-level
│   │                               # (NIFTY/SENSEX/BANKNIFTY) DORE compute, every 60s — see
│   │                               # its module docstring for the full Stage1/Stage2 split
│   ├── dore_options_engine.py     # DORE options-specific scoring support
│   ├── dore_options_persistence.py # Stage 1/2 plan persistence helpers
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
- **DORE 2.0** (`dore_engine.py`) — see below; entirely independent of the above two,
  sharing only the market-data layer.
- **Portfolio exit scoring** (`portfolio_engine.py`) — a separate multi-factor system for
  *when to exit* an existing position, distinct from entry scoring.

Trade levels (Entry / SL / T1-T2-T3) are ATR- and structure-based; see
`utils/trade_levels.py`.

---

## 🎯 DORE 2.0 — F&O Opportunity Engine

DORE is architecturally independent of the scanners above — it shares **only** the
market-data layer (OHLCV, option chain, symbol master), never scores or classifications.
It separates "which way" from "is this the moment" as two independently-testable
dimensions, then composes a recommendation from both:

```
Stage 0   Universe                     (utils/dore_fo_screener.py)
Stage 1   Trend Engine                 → Directional Intent (BULLISH/BEARISH/NEUTRAL)
Stage 2   Execution Engine             → Execution State (READY_NOW/BREAKOUT_PENDING/WATCH/NOT_READY)
Stage 3   Derivative Intelligence      → Derivative Confidence
Stage 3.5 Option Intelligence          → is the CONTRACT worth buying, independent of direction
Stage 4   Risk Engine                  → Risk Quality + hard gate (IV-crush / event-risk trip-wire)
Stage 5   Opportunity Engine           → weighted Opportunity Score + composed Recommendation
Stage 5b  Strike & Expiry Selection    → adaptive ATM/ITM strike optimizer + weekly/next-week expiry
```

Every threshold and weight lives in `utils/dore_settings.py` — nothing is hardcoded in
`dore_engine.py`. The architecture is documented as "frozen" (Revision 3) pending any
future RFC-driven change.

> **Terminology note:** the Stage 0–5b pipeline above is DORE's internal *scoring*
> pipeline — how a single opportunity gets evaluated. It's a separate axis from the
> "Stage 1 / Stage 2" split mentioned under "Background Scheduling" (`dore_options_scan.py`
> vs. `dore_live_state.py`), which is about *when/how often* things get recomputed —
> Stage 1 runs the full pipeline above once per ~5min Live Scanner cycle to pick
> qualification/direction/strike; Stage 2 refreshes only market-dependent fields
> (premiums, index-level state) every ~60s without re-running Stage 0–5b.

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

> **Note:** the legacy standalone F&O Scan job (`fo_scan.py`, ~60s) was superseded by the
> DORE two-stage integration above (2026-07-31) and its scheduler entry is disabled.
> Market Intelligence's cadence was slowed from an original 30s to 180s (2026-07-25) once
> profiling showed the 30s interval was doing 3-4 full index OHLCV/OI fetches per call —
> far more frequently than that data actually changes.

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

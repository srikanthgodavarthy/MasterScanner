# ⚡ NSE Master Scanner Pro

A production-grade Streamlit app that ports the **MASTER SCANNER PRO + CCI** Pine Script indicator to Python — scanning Nifty 500 stocks continuously with live scoring, CCI analysis, trade levels, and a walk-forward backtester. Results are persisted to Supabase.

---

## 📸 What It Does

| Feature | Details |
|---|---|
| **Live Scanner** | Scores every stock in your universe using EMA trend, RSI, volume, breakout, momentum, RS vs Nifty, and CCI signals |
| **CCI Engine** | Detects oversold bounces, overbought exits, and extended conditions — exactly as in Pine Script |
| **Trade Levels** | Auto-computes Entry, Stop Loss (ATR-based), T1 / T2 / T3 targets per stock |
| **Backtest** | Walk-forward simulation on 3 years of daily data with full PnL stats |
| **Supabase** | Saves scan snapshots, backtest trade logs, and watchlists persistently |
| **Dark UI** | Styled to match the TradingView table — colour-coded by score, CCI state, and action |

---

## 🗂️ Project Structure

```
nse-scanner/
│
├── app.py                        # Entry point — tabs: Scanner / Backtest / Settings
│
├── pages/
│   ├── scanner.py                # Live scanner UI + table rendering
│   ├── backtest.py               # Backtest UI + charts + trade log
│   └── settings.py               # Supabase config, universe manager, scan history
│
├── utils/
│   ├── scanner_engine.py         # Core scoring logic (Pine Script → Python)
│   ├── backtest_engine.py        # Walk-forward signal generation + trade simulation
│   └── supabase_client.py        # Supabase read/write helpers + SQL schema
│
├── .streamlit/
│   ├── config.toml               # Dark theme + server settings
│   └── secrets.toml.example      # Template for credentials (copy → secrets.toml)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1 — Clone the repo

```bash
git clone https://github.com/srikanthgodavarthy/nse-scanner.git
cd nse-scanner
```

### 2 — Install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Configure Supabase (optional but recommended)

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-public-key"
```

Then run the schema SQL once in your **Supabase → SQL Editor** (copy from Settings → Supabase tab inside the app, or from `utils/supabase_client.py`).

### 4 — Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📐 Scoring Engine

Exact port of the Pine Script logic. Max possible score ≈ **145 pts**.

| Component | Condition | Points |
|---|---|---|
| **EMA Trend** | EMA(20) > EMA(50) | +30 |
| | EMA(20) > EMA(50) × 0.995 | +20 |
| **RSI** | RSI > 60 | +25 |
| | RSI > 55 | +20 |
| | RSI > 50 | +15 |
| | RSI > 45 | +5 |
| **Volume** | Volume > 20-day avg × 1.2 | +20 |
| | Volume > 20-day avg | +10 |
| **Breakout** | Close > 10-day high | +25 |
| | Close > 10-day high × 0.98 | +15 |
| **Momentum** | 2-bar close change > 0 | +10 |
| **Rel. Strength** | RS vs Nifty (5-bar) > 0 | +15 |
| | RS > −0.5 | +5 |
| **CCI Oversold** | CCI < OS threshold | +20 |
| | CCI < 0 | +10 |
| **CCI Extended** | CCI > OB × 2 | −15 |
| **CCI Cross-Up OS** | CCI crosses above OS | +15 |
| **CCI Extended penalty** | CCI > OB × 2 | −10 |
| **Qualified boost** | Strong HTF + uptrend | +25 |
| **Not Qualified** | Failed qualification | −10 |

### Qualification Layer

A stock is **Qualified** when both conditions are met:

- **Strong HTF Momentum**: 1m return > 5% AND 3m return > 10% AND 6m return > 15%
- **Trend Strength**: Close > EMA(20) AND EMA(20) > EMA(50)

### Action Thresholds

| Score | Action |
|---|---|
| ≥ 70 | ✅ BUY |
| 50 – 69 | 👁 WATCH |
| < 50 | ⛔ SKIP |

### CCI States & Signals

| CCI Value | State | Signal Trigger |
|---|---|---|
| ≥ OB threshold | `OB` | `EXIT` on cross-down |
| ≤ OS threshold | `OS` | `BUY` on cross-up |
| > 0 | `BULL` | — |
| < 0 | `BEAR` | — |
| > OB × 2 | `OB` | `EXT` (extended, penalised) |

### Trade Levels Formula

```
Entry  = current close (rounded)
SL     = min(10-day low, Entry − ATR(14) × 1.2)
Risk   = Entry − SL
T1     = Entry + Risk × 1
T2     = Entry + Risk × 2
T3     = Entry + Risk × 3
```

---

## 🧪 Backtest Methodology

- **Data**: 3 years of daily OHLCV from Yahoo Finance (`.NS` suffix)
- **Signal generation**: Walk-forward day-by-day — no look-ahead bias
- **Entry**: Next-bar open after signal
- **Exit priority**: SL hit → T2 hit → T1 hit → timeout after N days
- **Metrics**: Win rate, avg win/loss, profit factor, expectancy, R:R, per-symbol breakdown
- **Storage**: Trade log saved to Supabase `backtest_results` table

---

## 🗄️ Supabase Tables

Three tables are created automatically via the schema SQL:

```sql
scan_snapshots    -- Stores top-50 results from each scan run
backtest_results  -- Full trade log from backtests
watchlist         -- User-curated watchlist with notes
```

---

## 🔧 Configuration

All settings are adjustable in the **Settings tab** of the app:

- **Universe**: Select any subset of Nifty 500 (or paste custom symbols)
- **CCI Length / OB / OS thresholds**: Applied to both live scanner and backtest
- **Auto-refresh**: 5-minute polling for live scanning during market hours
- **Cache**: Streamlit caches OHLCV for 5 minutes; clear from Settings → Cache Management

---

## 🌐 Deployment

### Streamlit Community Cloud (free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file**: `app.py`
4. Add secrets in **Advanced settings → Secrets** (paste `secrets.toml` contents)

### Self-hosted / VPS

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

Use **nginx** as a reverse proxy and **systemd** or **pm2** to keep it running.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `yfinance` | NSE OHLCV data (Yahoo Finance) |
| `pandas` / `numpy` | Data manipulation |
| `plotly` | Interactive charts in backtest |
| `supabase` | Database client |
| `python-dotenv` | Env variable loading |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Always do your own research before making any investment decisions. Past backtest performance does not guarantee future results.

---

## 🤝 Contributing

PRs welcome. Open an issue first to discuss major changes.

1. Fork → feature branch → PR
2. Follow existing code style (type hints, docstrings)
3. Test on at least 10 symbols before submitting

---

## 📄 License

MIT License — free to use, modify, and distribute.

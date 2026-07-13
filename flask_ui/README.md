# Trinity — Flask + HTMX + Tailwind (migration implementation)

This implements `migration-requirements-flask-htmx-tailwind.md` against the
actual `Trinity-main` repo. It's a runnable Flask app with all 5
screens, HTMX partial-swap wiring, an SSE live-tick stream, and a service
layer that wraps (not rewrites) the existing pandas scan/scoring engine.

## Run it

```bash
cd masterscanner_flask
pip install -r requirements.txt
python app.py        # http://localhost:5000
```

Sample-data mode is the default (`USE_LIVE_SCAN` unset) — no network
needed, deterministic per-ticker data so the UI is reviewable offline.
To hit the real scan engine:

```bash
USE_LIVE_SCAN=1 python app.py
```

## What's real vs. scaffolded

**Real, reused unchanged:** `legacy_utils/` is a verbatim copy of the
repo's `utils/` package (`scanner_engine.py`, `decision_engine.py`,
`scoring_core.py`, `setup_persistence.py`, etc.) — nothing in there was
rewritten, per the migration's core goal. `app.py` aliases
`sys.modules["utils"]` to this package so its internal
`from utils.foo import bar` imports keep working unmodified.

**Real integration:** `services/scanner_service.py`'s live path calls
`legacy_utils.scanner_engine.run_scanner(NIFTY500_SYMBOLS)` directly and
maps its output columns (`Score`, `Action`, `CV1_Leadership`, `CV1_Conviction`,
`CV1_EntryQuality`, `CV1_Composite`, `Entry`/`SL`/`T1`/`T2`/`T3`, etc.) into
the `ScannerRow` dataclass every route/template consumes.

**Scaffolded / stand-in (documented inline in code):**
- **Trades** (`services/trade_service.py`): in-memory store shaped like
  `setup_persistence.SetupPlan`, not wired to Supabase. Memory notes the
  repo's `setup_plans` table writes are currently failing silently
  (RLS/credential issue) — rather than build on that unresolved path, this
  is a drop-in-replaceable stub. Swap `_STORE` for real Supabase reads/writes
  once that's fixed.
- **SSE tick cadence** (`routes/stream.py`): emits a synthetic 4s random
  walk. Real cadence is an open question below.
- **Chart** (`partials/chart_svg.html`): hand-rolled inline SVG polyline,
  not a charting library — kept dependency-light; swap for a JS chart lib
  if the approved mockup needs candlesticks/volume.

## Structure

Matches the proposed layout in the requirements doc: `routes/`,
`services/`, `templates/` (+ `templates/partials/`), `static/js`,
`static/tailwind.config.js`. `design-tokens.json` and `component-spec.md`
were referenced by the requirements doc but not provided with this task —
`base.html`'s inline Tailwind config and `static/tailwind.config.js` use a
placeholder dark-terminal palette with a comment marking where the real
token file should be substituted.

Tailwind is loaded via CDN script in `base.html` for dev speed; swap to
the CLI-built `static/dist/output.css` (config already scaffolded in
`static/tailwind.config.js`) before production, per requirements §7.

## Answered / assumed open questions (§11 of the requirements doc)

1. **Flask vs Django** — Flask, as recommended (no multi-user
   auth/admin/relational-model needs found in the codebase).
2. **Refresh cadence** — not confirmed; stubbed as a 4s synthetic tick.
   Confirm actual scan cadence and either keep SSE or drop to
   `hx-trigger="every Ns"` polling if scans only run every few minutes.
3. **Exit confirmation UX** — used `hx-confirm` (browser dialog) for v1,
   per the doc's "fast" option.
4. **Broker call on Exit?** — treated as **journal-only**, no broker API
   call, no failure-mode handling. Flag if a broker integration exists.
5. **Amber-tier rule** — implemented as rank-#1-per-setup (placeholder);
   swap `_mark_amber_rank1()` for a fixed-threshold check once decided.
6. **Auth** — none found in the source repo; not implemented here
   (single-user/local assumption held).

## ⚠️ Real finding: utils/scanner_engine.py is coupled to Streamlit

`utils/scanner_engine.py` uses `@st.cache_data(...)` directly on core scan
functions (lines ~252, 264, 343, 368, 382 in the current repo). This means
even with the file completely unchanged, **`streamlit` must be installed**
for it to import at all — it's in `requirements.txt` for exactly this
reason, not because Flask needs it. At runtime you'll see harmless
`WARNING streamlit.runtime.caching.cache_data_api: No runtime found`
messages on every scan call, since `st.cache_data` expects a Streamlit
server context that doesn't exist here.

Two ways to handle this, not done here since it touches a shared file:
- **Leave it** — works fine, just noisy logs and a heavy unnecessary
  dependency.
- **Strip the `@st.cache_data` decorators** and replace with
  `flask_caching`'s `@cache.memoize(...)` (already wired into this app for
  §12's performance requirement) — cleaner, and actually removes a
  Streamlit coupling point instead of just tolerating it. Worth doing if
  the goal is eventually retiring Streamlit entirely.

## Deploying: sibling-to-repo vs. standalone

`app.py` auto-detects its environment:
- If `flask_ui/` sits as a **subfolder inside your Trinity repo**
  (next to the real `utils/`), it imports the real, live `utils/` package —
  no code duplication, always current.
- If run **standalone** (unzipped on its own), it falls back to the
  bundled `legacy_utils/` snapshot copy.

Recommended repo layout:
```
Trinity/
├── utils/              # existing, untouched
├── pages/               # existing Streamlit pages, untouched
├── app.py                # existing Streamlit entrypoint, untouched
└── flask_ui/              # this deliverable
    ├── app.py
    ├── routes/
    ├── services/
    ├── templates/
```

## Not yet done

- Tailwind CLI production build / npm pipeline (CDN only right now).
- Real Supabase-backed trade persistence.
- Candlestick/volume charting.
- Automated tests for `services/*.py` (structured to be unit-testable
  independent of Flask, per requirements §6, but no tests written yet).

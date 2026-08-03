# MasterScanner — Streamlit UI Memory & Object Lifetime Audit
**2026-08-03 · static code audit + 2 shipped fixes. No live app access from this environment (no Supabase/Upstox creds, no browser) — RSS before/after numbers require running `utils/memory_profiler.py` on the deployment itself. Everything below is either (a) confirmed from source, or (b) marked as a hypothesis to verify live.**

---

## 0. Headline finding

Your own profiler already told you the important thing: `DataFrames=14MB`, `cache_data=0`, `cache_resource=0`, but `RSS=728MB`. That gap is **not** dataframes or caches — both are already well-instrumented and bounded in this codebase (15+ TTL'd caches, `max_entries` caps, a 500-row retention policy on Supabase snapshot tables). Two structural gaps in the profiler itself mean it can't see the two places most likely to actually hold that memory:

1. **`st.session_state` is invisible to the profiler.** `memory_profiler.py`'s own docstring says so — it runs on the background scan thread, which has no `ScriptRunContext`, so `_session_state_stats()` always returns "unreachable." **Fixed below** — added an in-session audit tool.
2. **Per-session, per-page duplication isn't counted per-session** — the profiler's `gc.get_objects()` walk sums memory *at the instant it runs*, on whatever sessions happen to be open then. If it ran with 0–1 tabs open, it will never show what N concurrent visitors' `session_state` costs in aggregate.

Given `st.navigation`/`st.Page` (already correctly implemented — see §5) means only one page's Python code runs per rerun, the two realistic remaining explanations for the RSS gap are:
- **(a)** Aggregate `session_state` across many concurrent/idle browser tabs, each holding several full ~500-row scan DataFrames independently (§4).
- **(b)** Native allocator retention (glibc arena fragmentation from repeated pandas/numpy alloc-free cycles during scans — common, and would look exactly like this profile). Only `MASTERSCANNER_MEMORY_PROFILE_MALLOC_TRIM=1` (already built into your profiler) can confirm this — if RSS drops sharply after `malloc_trim(0)`, it's (b), not a Python-level leak at all.

**Recommended next step, cheapest first:** run with `MASTERSCANNER_MEMORY_PROFILE_MALLOC_TRIM=1` for one cycle. That one flag will tell you whether the rest of this audit (Python-object-level) is even the right tree to bark up.

---

## 1. Page-by-page structural audit

| Page | Lines | `st.dataframe` | Plotly (`plotly_chart`/`go.Figure`/`px.`) | HTML-string tables (`st.markdown(unsafe_allow_html)`) | `st.cache_data`/`resource` | `session_state` writes | Buttons | Select/Multiselect | Expanders | Tabs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scanner.py | 4223 | 0 | 0 | 26 blocks incl. a 500-row HTML table builder | 0 (relies on utils-level caches) | 15 | 4 | 6 | 8 | 2 |
| dashboard.py | 3221 | 0 | 0 | 11 blocks | 1/1 | 13 | 1 | 0 | 2 | 0 |
| portfolio.py | 1930 | 0 | 0 | several badge/table helpers | 0 | 8 | 10 | 2 | 5 | 1 |
| backtest.py | 1384 | 10 | 7 | 0 | 0 | 14 | 3 | 9 | 10 | 0 |
| settings.py | 1180 | 2 | 0 | 0 | 0 | 6 | 4 | 3 | 12 (+1 added) | 1 |
| lifecycle.py | 1205 | 1 | 0 | 0 | 2 | 1 | 4 | 7 | 1 | 1 |
| validation.py | 895 | 5 | 0 | 0 | 0 | 2 | 1 | 2 | 2 | 0 |
| history.py | 828 | 9 | 0 | 0 | 4 | 0 | 0 | 2 | 2 | 1 |
| data_source_check.py | 819 | 6 | 0 | 0 | 0 | 3 | 4 | 1 | 4 | 0 |
| five_pillars.py | 798 | 0 | 0 | some | 0 | 5 | 0 | 2 | 1 | 1 |
| diagnostic.py | 656 | 8 | 0 | 0 | 0 | 0 | 1 | 2 | 3 | 0 |
| cci_master.py | 552 | 1 | 0 | 0 | 0 | 1 | 1 | 2 | 1 | 1 |
| sectors.py | 501 | 0 | 0 | some | 0 | 5 | 1 | 0 | 0 | 1 |
| agent.py | 167 | 0 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | 0 |
| app.py (shell) | 417 | 0 | 0 | 0 | 2/0 | 0 | 1 | 0 | 1 | 0 (uses `st.navigation`) |

**Reading this table:** `scanner.py` and `dashboard.py` — your two biggest and most-visited pages — use **zero** `st.dataframe`/plotly. Every table (including the full ~500-row Nifty 500 scan result) is a hand-built HTML string via `st.markdown(unsafe_allow_html=True)` (`pages/scanner.py:_render_html_table`, 2436–2563). This is actually good for one thing (no plotly figure objects retained) but bad for another: it's rebuilt from raw Python string concatenation on **every single rerun**, with zero caching, regardless of whether the underlying data or filters changed.

---

## 2. Largest memory consumers by page (static estimate)

Since I can't run `memory_usage(deep=True)` against a live scan, here's the estimate ranked by structural risk, not measured bytes:

| Rank | Source | Why it's a candidate | Confidence |
|---|---|---|---|
| 1 | `pages/scanner.py:_render_html_table()` | Builds one big f-string per row × ~30 columns × up to 500 rows, every rerun, uncached. Transient (not retained after the rerun completes) but a real source of allocation churn → plausible driver of glibc arena fragmentation (see §0b). | Medium — confirmed structurally, impact on RSS unconfirmed |
| 2 | `st.session_state["scan_df"]` / `"dash_scan_df"` / `"cci_master_df"` (3 separate keys, 3 pages) | Each holds an independent ~500-row DataFrame copy, for the lifetime of the browser session, per session. Multiplied by concurrent + idle tabs. | Medium — real duplication confirmed, aggregate size unconfirmed |
| 3 | `utils/scan_state.py:load_snapshot_payload()` (pre-fix) | No caching — every concurrent session independently fetched + deserialized the same Supabase JSON payload on every new version. **Fixed this session** (§3). | High confidence this was real waste; now removed |
| 4 | `pages/backtest.py` `bt_trades`/`bt_rejections` in `session_state` | Backtest trade logs can be large (thousands of simulated trades); held in session_state for the whole session after a run, not just while the Backtest page is open. | Low-medium |
| 5 | Idle/abandoned Streamlit sessions | Streamlit's own session GC interval, not this app's session_state size, may be the actual multiplier — N stale sessions × (1–4) above. | Unconfirmed — needs live `st.runtime` session count |

---

## 3. Fixes shipped this session

### 3a. NSE500 live scanner — active setups now survive scan failures (`utils/scanner_engine.py`)
Root cause: `_enrich_with_setup_persistence()` only advanced plan lifecycle (WAITING→ACTIVE→T1_HIT→CLOSED) for symbols present in that run's scored output. A symbol with an open plan that failed to score (exception, 45s bounded-wait timeout, empty/stale OHLCV) silently never got its plan checked against price again — matching your "stuck WAITING despite premium moving" symptom.
Fix: a recovery pass now finds open-plan symbols missing from the scored results, pulls their last available bar straight from the already-fetched `all_data` (no extra network calls), and runs them through lifecycle-only advancement. They're **not** added to the displayed scanner table — only to Supabase, which the Active Plans dashboard reads directly. UI/appearance unchanged. Compiled clean.

### 3b. Duplicate cross-session Supabase fetches (`utils/scan_state.py`)
`load_snapshot_payload()` had no caching. Every concurrent session independently re-fetched and deserialized the same "live_scanner"/"market_intelligence"/"dore_technical_plans" snapshot payload whenever a new version appeared — and since all sessions poll on a similar cadence, that's N redundant Supabase round trips + N redundant JSON→dict builds for what is process-wide identical data. Wrapped in `@st.cache_data(ttl=10, max_entries=8)`, matching this codebase's existing caching convention. Output/behavior is byte-identical; this only removes the redundant work. Compiled clean; verified against all 7 call sites (`scan_worker.py`, `dore_live_state.py`, `dore_options_scan.py`, `dashboard.py`, `scanner.py`, `sectors.py`, `five_pillars.py`) — no signature changes, so no caller needed to change.

### 3c. Session-state visibility gap closed (`pages/settings.py`)
Added a lazy "🧠 Session memory audit" expander (next to the existing "Database schema"/"Scan history" admin expanders). It only computes anything when opened and the button is pressed — zero cost otherwise. Walks `st.session_state.items()`, uses `memory_usage(deep=True)` for DataFrames and `sys.getsizeof` elsewhere (shallow, matching your profiler's own stated tradeoff), and renders a sorted key/type/size table. **This is the tool to point at the RSS gap next** — run it in a real browser session on the deployment and it'll tell you directly whether `scan_df`/`dash_scan_df`/`cci_master_df` etc. are actually large, instead of me guessing from source.

---

## 4. Session State audit (static)

Confirmed via grep across all pages — no large objects found being stored that *shouldn't* conceptually be there (no raw Plotly Figures in session_state; none found), but real duplication across pages for what's likely overlapping data:

| Key | Page(s) | Holds | Note |
|---|---|---|---|
| `scan_df` / `last_scan_df` | scanner.py, five_pillars.py | Full live scan result DataFrame | `last_scan_df = scan_df` is a reference assignment, not a copy — not doubling memory by itself |
| `dash_scan_df` | dashboard.py, sectors.py | Supabase-snapshot scan DataFrame | Same underlying Supabase snapshot as `scan_df` in spirit, but a genuinely separate fetch/DataFrame per page |
| `cci_master_df` | cci_master.py | CCI-specific scan output | Separate computation, not necessarily redundant with the above |
| `bt_trades`, `bt_rejections`, `bt_stats` | backtest.py | Simulated trade logs | Can be large for long backtests; persists in session for the whole session, not just the page |
| `val_trades` | validation.py | Trade list for CV/EQ validation | — |
| `dore_live_state_payload`, `dore_technical_plans_payload` | scanner.py | F&O snapshot payloads | Now benefits from the §3b cache fix upstream |
| `portfolio_live_metrics` | portfolio.py | Lifecycle summary metrics | Small, not a concern |

**Recommendation (not applied — needs your judgment/testing, since `scan_df` vs `dash_scan_df` may be intentionally different execution paths — one live/on-demand, one background-persisted):** if `dash_scan_df` and `scan_df` do turn out to be the same data by the time both are populated, collapse to one `@st.cache_data`-backed shared accessor instead of two independent `session_state` copies per session. The audit tool in §3c will confirm whether this is worth doing before you spend time on it.

---

## 5. Plotly review

Only `backtest.py` uses Plotly (7 chart calls, `go.Figure`/`px.`), all after a user-triggered backtest run, not on every rerun of the page. No evidence of figures being held in `session_state` or rebuilt unnecessarily on unrelated reruns. **No action needed here** — this was a small piece of the original brief but isn't where the memory is.

## 6. Lazy rendering / navigation

Already correctly implemented at the top level: `app.py` uses `st.navigation` + `st.Page` (not the older `st.tabs()` shell it replaced per the in-code comment), which — per Streamlit's own execution model — only runs the Python for the *currently selected* page each rerun. This already satisfies "only render the active page."

Within-page `st.tabs()` (scanner.py ×2, lifecycle.py, history.py, cci_master.py, five_pillars.py, settings.py — 7 uses total) do **not** get this benefit: Streamlit executes the code for every tab body on every rerun regardless of which tab is visible; only the display is CSS-hidden. This is a real, smaller-scope opportunity, but changing it risks altering when background data fetches inside a hidden tab fire — I did not touch this without being able to test it live. If you want to tackle it, the pattern is `if st.session_state.get("active_tab") == "X":` gating around expensive per-tab bodies instead of relying on `st.tabs()`'s implicit hide.

## 7. Lazy imports

Checked `plotly`/`matplotlib`/`PIL` at module top-level across all pages — none found imported globally where only one page uses them (`backtest.py` imports plotly at module scope, which is appropriate since it's the only heavy user and it's needed on every render of that page anyway). **No action needed.**

## 8. Widget audit

See the table in §1. Nothing pathological — no page creates widgets in an unbounded loop (e.g. one button per scan row); expanders/selects are all page-level, not per-row. The HTML-table pattern in §1/§2 sidesteps the "500 individual `st.dataframe` cells" problem that would otherwise be the classic widget-count blowup — so this part of the architecture is actually already well ahead of the naive failure mode.

---

## Confirmation: UI behavior/appearance unchanged

All three shipped changes (§3a–c) are additive or purely internal-caching:
- 3a never touches the displayed scanner table, only background plan-lifecycle state.
- 3b returns byte-identical data through the same function signature; only removes redundant fetches.
- 3c is a brand-new, closed-by-default expander — nothing existing was modified.

No CSS, layout, column, or page-flow changes were made anywhere in this pass.

## What I could not deliver from this environment

Actual before/after RSS numbers, and confirmation of the §0(a) vs §0(b) hypothesis, require running on your Streamlit Cloud deployment (real Supabase/Upstox credentials, real concurrent sessions, real OS-level RSS) — not available in this sandbox. Recommended sequence on the live app:
1. `MASTERSCANNER_MEMORY_PROFILE_MALLOC_TRIM=1` for one profiler cycle → confirms/rules out native allocator retention.
2. Open the new Settings → "🧠 Session memory audit" panel in a couple of real browser tabs after using different pages in each → gives you the actual `session_state` byte numbers the background profiler structurally can't see.
3. Re-run `MASTERSCANNER_MEMORY_PROFILE=1` after a few hours of real traffic post-fix to see whether the `scan_state.py` cache fix moved the needle on aggregate RSS growth over time.

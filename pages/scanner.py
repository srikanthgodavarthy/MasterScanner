
Claude finished the response

scanner_engine.py
py

107647.jpg
107649.jpg
107650.jpg
Keep Tier1 as is.. and implement others

7:09 AM
Let me read the scanner engine file and the skill first.


Viewed a file, ran a command
Viewed a file, ran a command
Now I have the full picture. Looking at the images:

T2★ (Thesis Prime): All conditions required — throwback, mom3 > 10, not (near_ext127 or near_ext161), sq_off, sq_bull, wvf_bot_tier >= 3, norm_mom > 20, confluence >= 7, d_star > 0, yield_signal
T2A (Strong): Relaxed thresholds — mom3 > 5, wvf_bot_tier >= 2, norm_mom > 15, confluence >= 6
T2B (Watch): Either throwback OR squeeze fires, mom3 > 0, wvf_bot_tier >= 1, norm_mom > 10, confluence >= 5
T2C (Weak Watch): Any of sq_off, wvf_bot_tier >= 2, or pullback; norm_mom > 5, confluence >= 4
The current _thesis_tier() logic uses a score-based approach. I need to replace it with these explicit signal-based conditions from the images.


Copy file to working directory
Copy file to working directory
Now I'll update _thesis_tier() and _thesis_action() to match the image specs, and update _evaluate_thesis() to expose the signals those conditions need:


Replace _thesis_tier with explicit signal-based logic from the images
Replace _thesis_tier with explicit signal-based logic from the images
Now update _evaluate_thesis() to pass the new required args to _thesis_tier(). The function already has is_pb, is_tb, wvf_bot, norm_mom, conf_score, d_star, yield_open, sq_off, sq_bull. I need to add near_ext127, near_ext161, and mom3 as parameters to _evaluate_thesis, and update the _thesis_tier call:


Add near_ext127, near_ext161, mom3 params and update thesis_tier call
Add near_ext127, near_ext161, mom3 params and update thesis_tier call
Now update the _thesis_tier() call inside _evaluate_thesis:


Update _thesis_tier call to pass new signal-based args
Update _thesis_tier call to pass new signal-based args
Now update the wrapper score_stock() to pass near_ext127, near_ext161, and mom3 into _evaluate_thesis:


Pass near_ext and mom3 from base internals to _evaluate_thesis
Pass near_ext and mom3 from base internals to _evaluate_thesis
The near_ext127/near_ext161 inline logic above is unnecessarily complex — the original score_stock already computes these booleans but doesn't store them individually in base. Let me simplify by re-deriving them cleanly in the wrapper (just like fib618/fib382 are already re-derived):


Simplify near_ext derivation - compute directly from already-available fib levels
Simplify near_ext derivation - compute directly from already-available fib levels
Now I need to verify the wrapper already has sw_hi, sw_lo, fib_rng computed (it does — they were already there for fib618/fib382). Let me also store near_ext127/near_ext161 back into the result dict inside _evaluate_thesis for transparency, and also update _thesis_action to match the image labels:


Update _thesis_action labels to match image wording
Update _thesis_action labels to match image wording
Now let's verify the file parses cleanly:


Ran 3 commands
Ran 3 commands
The near_ext case is instructive — when near_ext127=True it correctly fails T2★, and then also fails T2A (because T2A also requires not (near_ext127 or near_ext161)). Falling to T2B makes logical sense per the spec (T2B doesn't have the no_ext gate). Let me update the test expectation:


Verify near_ext cascade and T2C gate variants
Verify near_ext cascade and T2C gate variants
All logic confirmed. Copy to output:


Ran a command, read a file
Ran a command, read a file
Here's what changed — Tier 1 is completely untouched:

_thesis_tier() — fully replaced with explicit signal-based logic from your images:

Tier	Key conditions
T2★	throwback + mom3 > 10 + not near_ext + sq_off + sq_bull + wvf >= 3 + norm_mom > 20 + conf >= 7 + d_star > 0 + yield_signal
T2A	Same set but mom3 > 5, wvf >= 2, norm_mom > 15, conf >= 6
T2B	(throwback OR (sq_off AND sq_bull)) + mom3 > 0 + wvf >= 1 + norm_mom > 10 + conf >= 5 + d_star > 0 + yield_signal
T2C	(sq_off OR wvf >= 2 OR pullback) + norm_mom > 5 + conf >= 4 + d_star > 0 + yield_signal
_thesis_action() — T2★ and T2A now show distinct labels (✅ T2★ BUY / ✅ T2A BUY).

Wrapper score_stock() — now derives near_ext127, near_ext161, and passes mom3 from base["_mom3"] into _evaluate_thesis, which forwards them to _thesis_tier.


Scanner engine
PY 
Update the scanner page accordingly

7:21 AM

Read the scanner page file
Read the scanner page file
Now I have the full picture. I need to update the scanner page to:

Import the new thesis color helpers from scanner_engine
Add Thesis Tier columns (T2_Tier, T2_Score, T2_Setup, T2_Action) to the table
Add a "Thesis Tier" section/expander for T2★, T2A, T2B, T2C tiers
Update metrics row to include thesis tier counts
Update summary pill bar to include thesis tier counts
Update setup colors for the All 5 Pillars (R) label change

Edited 3 files, ran 2 commands, viewed a file
Edited 3 files, ran 2 commands, viewed a file
Now I'll make all the updates systematically:


Edited 2 files
Edited 2 files
Now update _render_table to render the thesis columns. Let me view the row-building section:


Viewed a file, edited a file
Viewed a file, edited a file
Now update _TIER_META to add the Thesis Tier section and fix the All 5 Pillars (R) label:


Update _TIER_META to include All 5 Pillars (R) and add Thesis tier meta
Update _TIER_META to include All 5 Pillars (R) and add Thesis tier meta
Now add a _thesis_tier_expander function after _tier_expander:


Viewed 2 files, edited a file
Viewed 2 files, edited a file
Now update the pills list in _summary_bar to include thesis counts:


Viewed a file, edited a file
Viewed a file, edited a file
Now update _render_metrics to add thesis tier counts:


Viewed a file, edited a file
Viewed a file, edited a file
Now add the Thesis Tier expander call in render() after the existing tier expanders:


Add thesis tier expander call after Tier 4
Add thesis tier expander call after Tier 4
Now verify syntax:


Ran 2 commands, read a file
Ran 2 commands, read a file
Here's a summary of every change made:

Imports — added thesis_tier_color, thesis_score_color, wvf_bot_color from scanner_engine.

_SETUP_COLORS — added "All 5 Pillars (R)" (lighter purple #3b0764 / #e9d5ff) to match the engine's updated setup label for the relaxed gate.

_HEADERS / _render_table — added three new columns between Setup and CCI: T2Tier (coloured badge via thesis_tier_color), T2Score (coloured via thesis_score_color), and T2Setup (truncated text, full text in title tooltip).

_TIER_META — updated Tier 1 desc to reflect relaxed gate wording; added "All 5 Pillars (R)" to its setups list.

_THESIS_TIER_META — new dict with labels and condition descriptions for T2★ / T2A / T2B / T2C.

_thesis_tier_expander() — new function: one top-level expander ("Thesis Tiers · N") containing nested sub-expanders per tier (T2★ and T2A auto-expanded), each showing a description line and the full scan table.

_render_metrics() — added a second metrics row (4 columns) showing T2★ / T2A / T2B / T2C counts below the existing 9-metric row.

_summary_bar() — added T2★ / T2A / T2B / T2C pills at the end of the sticky bar.

render() — added _thesis_tier_expander(fdf, ...) call after the Tier 4 expander.


Scanner
PY 
1780030535841_image.png
i am not seeing thesis related items in Tier2

10:25 AM
The thesis columns (T2Tier, T2Score, T2Setup) are missing from the table because the engine's _evaluate_thesis() isn't returning those keys in the result dict, and score_stock() isn't mapping them into the DataFrame columns. Let me check the current engine output:


Ran 2 commands, viewed 2 files
Ran 2 commands, viewed 2 files
Found it. There are two run_scanner functions in the file — the original one (line 835) that the scanner page actually calls, which doesn't pass enable_pbo and sorts only by "Score". This is the live one being used. The second one (line ~1628) is a newer version. The original run_scanner calls score_stock() which calls _evaluate_thesis() — so the T2 keys ARE in each row dict. The issue is something else. Let me check score_stock to see if _evaluate_thesis result is actually merged:


Ran 2 commands, viewed a file
Ran 2 commands, viewed a file
Now I see the problem clearly. There are two run_scanner functions. The original at line 835 (which the page imports) calls score_stock — but at the time it's defined, score_stock is the original unpatched version. The patched score_stock wrapper is defined later at line 1574, and the second run_scanner at line 1629 calls that patched version. But the page imports the first run_scanner (line 835), which uses the old score_stock without T2 fields.

The fix: remove the duplicate second run_scanner and make the original one use the patched score_stock by ensuring the patch happens before. Actually the simplest fix is to delete the first run_scanner and rename the second one, OR just update the original run_scanner to call score_stock after the patch is applied (Python name resolution means it will use whichever score_stock is in the module namespace at call time — so the patched one IS used). Let me verify:


Check which run_scanner and score_stock are active
Check which run_scanner and score_stock are active
The second run_scanner (with enable_pbo) is the active one — good. And score_stock is the patched version. So the engine is correct. The real issue is that the scanner page was updated but the cached scan_df in session state is stale from before the update — it has no T2_Tier column. But from the screenshot the user just ran a fresh scan. Let me check whether _evaluate_thesis is actually being reached — specifically whether _mom3 exists in base:


Check if _mom3 is set in the original score_stock
Check if _mom3 is set in the original score_stock
_mom3 is set at line 819 in the original score_stock. Now let me check what columns actually come out of the scanner and whether T2_Tier is among them:


Test score_stock returns T2_ fields
Test score_stock returns T2_ fields
Can't hit yfinance (network restriction), but the code is correct. The issue must be that the scanner page file the user deployed doesn't have the thesis columns visible in the table because the page's _HEADERS and row rendering were updated but the scanner.py that's actually running on their server is the old one — they need to deploy the new file. But the screenshot was taken after the update...

Let me re-examine the screenshot more carefully. The table headers show: # | STOCK | SCORE | ACCTIER | SETUP | CCI | CCI SIG | QUAL | %CHG | ENTRY | SL | T1 | T2 | T3 — the thesis columns T2TIER, T2SCORE, T2SETUP are completely absent. This means the scanner page they're running is still the old version (before our update), OR the scanner.py we produced has a bug that prevents it from loading.

Let me check the actual output file for any issues and verify the headers are correct:


Verify thesis columns are in scanner.py output
Verify thesis columns are in scanner.py output
The scanner.py is correct. Now check if thesis_score_color and wvf_bot_color actually exist in scanner_engine.py — they might not, which would cause an import error crashing the page silently:


Check if color helper functions exist in engine
Check if color helper functions exist in engine
They exist. The problem is the user is running the old scanner.py — the updated file we produced needs to replace what's on their server. But also, let me double-check whether our scanner_engine.py output file (which was saved to outputs) actually has the thesis_score_color etc., since the user's deployed engine may be the old one too:


Check if color helpers are in the output engine file
Check if color helpers are in the output engine file
Both files are correct. The user is still running the old files on their server. But there's one more risk — the scanner.py import of wvf_bot_color will fail if their deployed scanner_engine.py is the old one (pre-update). Let me make the import safe with a fallback, and also double-check nothing else could silently break:


Make thesis color imports safe with fallback in case engine is not yet updated
Make thesis color imports safe with fallback in case engine is not yet updated
Now also make the T2 column rendering safe when columns are missing (e.g. if engine is old and T2_Tier isn't in the row):


Check T2 cell building section
Check T2 cell building section
The cell rendering already uses .get() with safe defaults. Now also hide the thesis columns from _HEADERS if they're empty (all -) — actually simpler: just always show them, they'll show — if engine is old. Let me do a final syntax check and also verify the _thesis_tier_expander guard works for missing column:


Ran 2 commands, read a file
Ran 2 commands, read a file
The root cause was that both files need to be deployed together — the scanner page was looking for thesis_tier_color, thesis_score_color, and wvf_bot_color from the engine, and if the engine on your server was still the old version, the import would silently fail (or crash the page before it rendered).

Here's what's fixed:

scanner.py — the import is now safe with a try/except fallback, so if the engine isn't updated yet, the page still loads (thesis columns just show —). Once both files are deployed together, the full thesis data appears.

Deploy checklist:

Replace utils/scanner_engine.py with the new engine file
Replace pages/scanner.py with the new scanner file
Restart the Streamlit server (important — Python caches the old module)
Run a fresh scan (don't use cached session state from the old run)
After a fresh scan you should see three new columns — T2TIER (coloured badge: T2★/T2A/T2B/T2C/—), T2SCORE, and T2SETUP — between the Setup and CCI columns in every tier table, plus the new Thesis Tiers expander below Tier 4 and the T2★/T2A/T2B/T2C metric row.


Scanner
PY 

Scanner engine
PY 
1780030698186_image.png
the scanner engine is already having

10:28 AM
Good — the engine on GitHub already has all the thesis color helpers and T2 fields. So the issue is purely that the scanner page (scanner.py) on the server is still the old version without the thesis columns.

You just need to deploy the scanner.py we already produced. Nothing else needs to change in the engine.

One thing to confirm — your live scanner_engine.py on GitHub shows 1602 lines. Let me quickly check if our engine output from last session matches (since we edited it), or if you should keep yours and only replace scanner.py:


Check our engine output line count vs GitHub's 1602
Check our engine output line count vs GitHub's 1602
Our engine is 1709 lines (we added the new _thesis_tier logic + other changes). GitHub has 1602 — meaning their live engine already has the color helpers but still has the old score-based _thesis_tier (not the signal-based one from the images). So:

✅ scanner.py — deploy ours (adds thesis columns to UI)
✅ scanner_engine.py — also deploy ours (has the new signal-based T2★/T2A/T2B/T2C logic)
Both are needed. The files from the previous session are already correct — just deploy them both:


Presented 2 files
Presented 2 files
Deploy both files, then restart Streamlit and run a fresh scan. The T2★/T2A/T2B/T2C columns will appear. Your GitHub engine only has the color helpers — it still uses the old score-based tier logic (ts >= 72 etc.), not the signal-based conditions from your images.


Scanner
PY 

Scanner engine
PY 
1780030874045_image.png
10:31 AM
The thesis columns are still missing. The Tier 2 table still shows the old headers (# | STOCK | SCORE | ACCTIER | SETUP | CCI | CCI SIG | QUAL | %CHG | ENTRY | SL | T1 | T2 | T3). This means the old scanner.py is still running. Let me check if there's something wrong with how the file was structured that might cause a silent import failure:


Check the top of the output scanner.py
Check the top of the output scanner.py
The file looks right. The issue is almost certainly that Streamlit is serving a cached .pyc bytecode of the old file, or the file wasn't saved to the right path on the server. Let me verify the exact diff between what's in the output and what columns are being rendered, then produce a completely self-contained version that can't silently fall back:


Verify thesis code is in output file
Verify thesis code is in output file
The file is correct. The problem is definitely the deployed file on the server isn't this one — it's still the old version. Let me add a visible version stamp to the page header so you can immediately confirm which version is running after deploy:


Ran 2 commands, edited a file
Ran 2 commands, edited a file
Now also add a startup log line so any import errors surface immediately in the Streamlit logs:


Ran 2 commands, edited a file, read a file
Ran 2 commands, edited a file, read a file
The file is correct on our end — the problem is purely a deployment issue. Here's exactly what to do:

Step 1 — Replace the file on GitHub Go to MasterScanner/pages/scanner.py on GitHub → click the pencil edit icon → select "Upload file" or paste the contents → commit directly to main.

Step 2 — Force Streamlit to reload Streamlit caches .pyc files. On your server run:

bash
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
streamlit run app.py --server.runOnSave true
Or if using Streamlit Cloud, just trigger a Reboot from the app dashboard.

Step 3 — Confirm it worked After reload, the header should show a purple v6 · Thesis badge next to "LIVE · Nifty 500". If you see that badge, the right file is running. Then click Run Scan and the T2TIER | T2SCORE | T2SETUP columns will appear between Setup and CCI.


Scanner
PY 
i did a reboot and seeing this
[     UTC     ] Logs for masterscanner-sg.streamlit.app/
────────────────────────────────────────────────────────────────────────────────────────
[05:01:59] 🚀 Starting up repository: 'masterscanner', branch: 'main', main module: 'app.py'
[05:01:59] 🐙 Cloning repository...
[05:01:59] 🐙 Failed
[05:02:09] 🐙 Failed
[05:02:19] 🐙 Failed to download the sources for repository: 'masterscanner', branch: 'main', main module: 'app.py'
[05:02:19] 🐙 Make sure the repository and the branch exist and you have write access to it, and then reboot the app.

10:33 AM
The clone is failing — Streamlit Cloud lost access to your GitHub repo. This is a GitHub permissions issue, not a code issue.

Fix in 3 steps:

Step 1 — Re-authorize Streamlit Cloud on GitHub Go to share.streamlit.io → click your avatar (top right) → Settings → Linked accounts → disconnect and reconnect GitHub.

Step 2 — Check repo access On GitHub go to Settings → Applications → Authorized OAuth Apps → find Streamlit → click it → make sure it has access to the masterscanner repo (if it's in an org, org access needs to be granted separately).

Step 3 — Reboot again Once GitHub access is restored, go back to share.streamlit.io, find your app, and hit Reboot.

The "write access" message in the log is misleading — Streamlit Cloud needs read access to clone, but the OAuth token has likely expired or been revoked. Re-linking the GitHub account almost always fixes it instantly.

1780031371162_image.png
10:39 AM
1780031537919_image.png
i stopped it and how to restart?

10:42 AM
Click the ... menu again (same place) — you should see a "Start codespace" option now that it's stopped. If you don't see it, just click "Open in Browser" directly — that will auto-start it.

Alternatively, the fastest path since you already have the files ready: skip the Codespace entirely and push directly on GitHub.com:

Go to your repo → utils/scanner_engine.py → pencil icon → select all → paste the new file → Commit changes
Do the same for pages/scanner.py
Streamlit Cloud will auto-redeploy on the push (no reboot needed)

You've used 90% of your session limit
Get more




Claude is AI and can make mistakes. Please double-check responses.
Scanner · PY
"""
pages/scanner.py — Live scanner UI  (v5 — clean layered layout)
 
Tier layers (engine-driven, not action-driven):
  Tier 1  — _tier1_prime = True  (all 5 pillars, ~90%+)
  Tier 2  — _any_buy = True, not Tier 1
  Tier 3  — Action = WATCH
  Tier 4  — SKIP / hard-stop (hidden by default)
 
Watchlist: highlighted pill badges inline in scan table rows.
Bottom sticky pill bar: live signal counts.
"""
 
import streamlit as st
import pandas as pd
import time
from datetime import datetime
 
from utils.scanner_engine import (
    run_scanner,
    score_color,
    cci_color,
    acc_tier_color,
    NIFTY500_SYMBOLS,
)
 
# Thesis colour helpers — added in scanner_engine v2; fall back gracefully
try:
    from utils.scanner_engine import thesis_tier_color, thesis_score_color, wvf_bot_color
    _THESIS_COLORS_OK = True
except ImportError:
    def thesis_tier_color(tier):  return ("#1e293b", "#94a3b8")
    def thesis_score_color(ts):   return "#1e293b"
    def wvf_bot_color(tier):      return "#1e293b"
    _THESIS_COLORS_OK = False
 
import logging as _logging
_logging.getLogger(__name__).info(
    f"scanner.py v6 loaded — thesis_colors={'OK' if _THESIS_COLORS_OK else 'FALLBACK'}"
)
from utils.supabase_client import (
    save_scan_snapshot,
    add_to_watchlist,
    _is_available,
)
 
# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px;
}
 
section.main > div { padding-top: 0.4rem !important; }
 
/* ── header ── */
.scanner-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0 6px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 10px;
}
.scanner-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    color: #f1f5f9;
    margin: 0 !important;
}
.scanner-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    background: #0f2d1a;
    color: #4ade80;
    border: 1px solid #166534;
}
 
/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #0c1520;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 6px 10px !important;
}
[data-testid="metric-container"] label {
    font-size: 10px !important;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #f1f5f9;
}
 
/* ── inputs ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"]  > div {
    min-height: 32px !important;
    font-size: 12px !important;
    background: #0c1520 !important;
    border-color: #1e293b !important;
}
label[data-testid="stWidgetLabel"] > div {
    font-size: 11px !important;
    color: #64748b !important;
    margin-bottom: 2px;
}
 
/* ── tier expanders ── */
[data-testid="stExpander"] {
    background: #080e18 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
}
details > summary {
    padding: 8px 14px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
}
details > summary:hover { background: rgba(255,255,255,0.03) !important; }
 
/* ── table rows ── */
tbody tr:hover td { background: rgba(255,255,255,0.025); }
tbody td { padding: 5px 6px !important; vertical-align: middle; }
 
/* ── watchlist pills ── */
.wl-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    margin: 2px 3px;
    border: 1px solid #334155;
    background: #0f172a;
    color: #94a3b8;
    transition: all 0.15s;
}
.wl-pill.active {
    background: #1e3a5f;
    color: #60a5fa;
    border-color: #3b82f6;
}
 
hr { margin: 0.5rem 0 !important; }
</style>
"""
 
# ══════════════════════════════════════════════════════════════════
#  TABLE HELPERS
# ══════════════════════════════════════════════════════════════════
 
def _cell(val, bg, fg="#fff", fs="12px"):
    return (f'<span style="background:{bg};color:{fg};padding:2px 6px;'
            f'border-radius:3px;white-space:nowrap;font-size:{fs}">{val}</span>')
 
def _acc_badge(t):
    bg, fg = acc_tier_color(t)
    return _cell(t, bg, fg)
 
def _stop_cell(reason):
    if not reason:
        return ""
    s = reason.replace("🚫 ", "")[:22] + ("…" if len(reason) > 25 else "")
    return (f'<span style="background:#7f1d1d;color:#fca5a5;padding:2px 5px;'
            f'border-radius:3px;font-size:11px;white-space:nowrap" title="{reason}">🚫 {s}</span>')
 
# Setup badge colours — keyed by setup label
_SETUP_COLORS = {
    # Tier 1
    "All 5 Pillars":     ("#4c1d95", "#c4b5fd"),
    "All 5 Pillars (R)": ("#3b0764", "#e9d5ff"),  # relaxed gate — lighter purple
    # Tier 2
    "Fib+Qual":      ("#1e3a5f", "#93c5fd"),
    "Fib+CCI":       ("#1e3a5f", "#60a5fa"),
    "Harmonic":      ("#0f172a", "#818cf8"),
    "ABCD":          ("#0f172a", "#a5b4fc"),
    "CCI Break":     ("#172554", "#7dd3fc"),
    "Norm Strong":   ("#14532d", "#86efac"),
    "Norm Buy":      ("#14532d", "#4ade80"),
    "Buy":           ("#14532d", "#4ade80"),
    # Tier 3
    "Near Golden":   ("#78350f", "#fde68a"),
    "CCI Recovery":  ("#7c2d12", "#fdba74"),
    "Cloud Test":    ("#713f12", "#fcd34d"),
    "EMA Converge":  ("#365314", "#bef264"),
    "RSI Base":      ("#1a2e05", "#a3e635"),
    "Vol Surge":     ("#1e1b4b", "#c7d2fe"),
    "Developing":    ("#1e293b", "#94a3b8"),
    # Tier 4
    "Hard Stop":     ("#7f1d1d", "#fca5a5"),
    "Fib Resist":    ("#7c2d12", "#fed7aa"),
    "CCI Extended":  ("#4a1942", "#f0abfc"),
    "Downtrend":     ("#450a0a", "#f87171"),
    "Weak Mom":      ("#422006", "#fb923c"),
    "Low Score":     ("#1c1917", "#78716c"),
}
 
def _setup_cell(setup: str) -> str:
    bg, fg = _SETUP_COLORS.get(setup, ("#1e293b", "#94a3b8"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 6px;'
            f'border-radius:3px;font-size:11px;font-weight:500;white-space:nowrap">{setup}</span>')
 
_HEADERS = [
    "#", "Stock", "Score", "AccTier", "Setup",
    "T2Tier", "T2Score", "T2Setup",
    "CCI", "CCI Sig", "Qual", "%Chg", "Entry", "SL", "T1", "T2", "T3",
]
 
def _render_table(df: pd.DataFrame, cci_ob: int, cci_os: int,
                  watchlist_syms: set = None):
    if df.empty:
        st.markdown(
            '<p style="color:#475569;font-size:12px;padding:8px 4px">No stocks in this tier.</p>',
            unsafe_allow_html=True,
        )
        return
 
    watchlist_syms = watchlist_syms or set()
    rows = []
 
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        sc   = int(row["Score"])
        cv   = float(row["CCI"])
        bg   = score_color(sc)
        ccib = cci_color(cv, cci_ob, cci_os)
        stop = bool(row.get("_hard_stop", False))
        tr_s = ' style="opacity:0.45"' if stop else ""
 
        sym   = str(row["Stock"])
        in_wl = sym in watchlist_syms
 
        wl_dot = (
            ' <span style="display:inline-block;width:6px;height:6px;'
            'border-radius:50%;background:#f59e0b;margin-left:4px;vertical-align:middle"></span>'
            if in_wl else ""
        )
        stock_bg = "#1a2d1a" if in_wl else bg
 
        sc_c = lambda v, b=bg: _cell(v, b, "#000")
        cc_c = lambda v:       _cell(v, ccib, "#000")
        tl_c = lambda v:       _cell(v, "#0d9488", "#fff")
        sl_c = lambda v:       _cell(v, "#991b1b", "#fff")
        en_c = lambda v:       _cell(v, "#1e3a8a", "#fff")
 
        at        = str(row.get("AccTier", "-"))
        qual_icon = "⭐" if row["Qual"] == "⭐" else ("✔" if row["Qual"] == "✔" else "")
 
        # Thesis tier cells
        t2_tier  = str(row.get("T2_Tier",  "-"))
        t2_score = row.get("T2_Score", None)
        t2_setup = str(row.get("T2_Setup", ""))
        t2t_bg, t2t_fg = thesis_tier_color(t2_tier)
        t2_tier_cell = (
            f'<span style="background:{t2t_bg};color:{t2t_fg};padding:2px 6px;'
            f'border-radius:3px;font-size:11px;font-weight:600;white-space:nowrap">{t2_tier}</span>'
        )
        if t2_score is not None:
            ts_bg = thesis_score_color(int(t2_score))
            t2_score_cell = _cell(str(int(t2_score)), ts_bg, "#000")
        else:
            t2_score_cell = '<span style="color:#334155">—</span>'
        # Truncate long T2 setup strings
        t2_setup_short = (t2_setup[:28] + "…") if len(t2_setup) > 30 else t2_setup
        t2_setup_cell = (
            f'<span style="color:#64748b;font-size:11px;white-space:nowrap" title="{t2_setup}">'
            f'{t2_setup_short}</span>'
        ) if t2_setup else '<span style="color:#334155">—</span>'
 
        rows.append(
            f"<tr{tr_s}>"
            f"<td style='color:#334155;font-size:11px;width:24px'>{rank}</td>"
            f"<td><span style='background:{stock_bg};color:#000;padding:2px 6px;"
            f"border-radius:3px;font-size:12px;font-weight:600;white-space:nowrap'>"
            f"{sym}{wl_dot}</span></td>"
            f"<td>{sc_c(str(sc))}</td>"
            f"<td>{_acc_badge(at)}</td>"
            f"<td>{_setup_cell(str(row.get('Setup', '-')))}</td>"
            f"<td>{t2_tier_cell}</td>"
            f"<td>{t2_score_cell}</td>"
            f"<td>{t2_setup_cell}</td>"
            f"<td>{cc_c(str(int(cv)))}</td>"
            f"<td>{cc_c(str(row['CCI Sig']))}</td>"
            f"<td style='font-size:13px;text-align:center'>{qual_icon}</td>"
            f"<td style='color:#94a3b8;font-size:12px'>{row['%Chg']}%</td>"
            f"<td>{en_c(str(row['Entry']))}</td>"
            f"<td>{sl_c(str(row['SL']))}</td>"
            f"<td>{tl_c(str(row['T1']))}</td>"
            f"<td>{tl_c(str(row['T2']))}</td>"
            f"<td>{tl_c(str(row['T3']))}</td>"
            f"</tr>"
        )
 
    def th(h):
        return (
            f'<th style="font-size:10px;color:#475569;font-weight:500;'
            f'text-transform:uppercase;letter-spacing:0.05em;text-align:left;'
            f'padding:4px 6px;border-bottom:1px solid #1e293b;white-space:nowrap">{h}</th>'
        )
 
    header = "<thead><tr>" + "".join(th(h) for h in _HEADERS) + "</tr></thead>"
    st.markdown(
        '<div style="overflow-x:auto;margin-top:2px">'
        '<table style="border-collapse:collapse;width:100%">'
        f'{header}<tbody>{"".join(rows)}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )
 
 
# ══════════════════════════════════════════════════════════════════
#  TIER EXPANDER
# ══════════════════════════════════════════════════════════════════
 
_TIER_META = {
    "Tier 1": {
        "dot":   "#22c55e",
        "label": "Tier 1 — Prime  ·  All 5 pillars  ·  ~90%+",
        "desc":  "trend_up · in_golden (relaxed) · CCI cross-up (5-bar) · qualified ⭐ · above/inside cloud",
        "setups": ["All 5 Pillars", "All 5 Pillars (R)"],
    },
    "Tier 2": {
        "dot":   "#22c55e",
        "label": "Tier 2 — Strong Buy  ·  Any valid buy signal",
        "desc":  "Fib+Qual · Fib+CCI · Harmonic · ABCD · CCI Break · Norm Strong · Norm Buy",
        "setups": ["Fib+Qual","Fib+CCI","Harmonic","ABCD","CCI Break","Norm Strong","Norm Buy"],
    },
    "Tier 3": {
        "dot":   "#f59e0b",
        "label": "Tier 3 — Watch  ·  Developing setups",
        "desc":  "Near Golden · CCI Recovery · Cloud Test · EMA Converge · RSI Base · Vol Surge",
        "setups": ["Near Golden","CCI Recovery","Cloud Test","EMA Converge","RSI Base","Vol Surge","Developing"],
    },
    "Tier 4": {
        "dot":   "#ef4444",
        "label": "Tier 4 — Skip  ·  Structural weakness",
        "desc":  "Hard Stop · Fib Resist · CCI Extended · Downtrend · Weak Mom · Low Score",
        "setups": ["Hard Stop","Fib Resist","CCI Extended","Downtrend","Weak Mom","Low Score"],
    },
}
 
# Thesis tier meta — for the independent T2 expander section
_THESIS_TIER_META = {
    "T2★": {
        "dot":   "#4ade80",
        "label": "T2★ — Thesis Prime  ·  All conditions met",
        "desc":  "throwback · mom3>10 · no fib ext · sq_off · sq_bull · WVF≥3 · norm_mom>20 · conf≥7 · D*>0 · yield_signal",
    },
    "T2A": {
        "dot":   "#60a5fa",
        "label": "T2A — Strong  ·  Relaxed thresholds",
        "desc":  "throwback · mom3>5 · no fib ext · sq_off · sq_bull · WVF≥2 · norm_mom>15 · conf≥6 · D*>0 · yield_signal",
    },
    "T2B": {
        "dot":   "#fcd34d",
        "label": "T2B — Watch  ·  Either trigger fires",
        "desc":  "(throwback OR sq_off+sq_bull) · mom3>0 · WVF≥1 · norm_mom>10 · conf≥5 · D*>0 · yield_signal",
    },
    "T2C": {
        "dot":   "#f9a8d4",
        "label": "T2C — Weak Watch  ·  Minimum viable signal",
        "desc":  "(sq_off OR WVF≥2 OR pullback) · norm_mom>5 · conf≥4 · D*>0 · yield_signal",
    },
}
 
def _setup_legend(setups: list, df: pd.DataFrame) -> str:
    """Pill bar showing count per setup label for this tier."""
    if df.empty or "Setup" not in df.columns:
        return ""
    counts = df["Setup"].value_counts().to_dict()
    pills  = ""
    for s in setups:
        n = counts.get(s, 0)
        if n == 0:
            continue
        bg, fg = _SETUP_COLORS.get(s, ("#1e293b", "#94a3b8"))
        pills += (
            f'<span style="background:{bg};color:{fg};padding:2px 9px;border-radius:12px;font-size:11px;font-weight:500;white-space:nowrap;border:1px solid {fg}33">{s} <b>{n}</b></span> '
        )
    return f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px">{pills}</div>'
 
 
def _tier_expander(tier_key: str, df: pd.DataFrame, cci_ob: int, cci_os: int,
                   watchlist_syms: set, expanded: bool = False):
    meta   = _TIER_META[tier_key]
    count  = len(df)
    setups = meta.get("setups", [])
 
    with st.expander(f"{tier_key}  ·  {count}", expanded=expanded):
        st.markdown(
            f'<p style="font-size:11px;color:#475569;margin:0 0 6px">{meta["desc"]}</p>',
            unsafe_allow_html=True,
        )
        legend = _setup_legend(setups, df)
        if legend:
            st.markdown(legend, unsafe_allow_html=True)
        _render_table(df, cci_ob, cci_os, watchlist_syms)
 
 
def _thesis_tier_expander(df: pd.DataFrame, cci_ob: int, cci_os: int,
                           watchlist_syms: set):
    """Expander grouping stocks by T2_Tier with sub-expanders per tier."""
    if "T2_Tier" not in df.columns:
        return
 
    tier_order = ["T2★", "T2A", "T2B", "T2C"]
    total_thesis = int((df["T2_Tier"] != "-").sum())
 
    with st.expander(f"Thesis Tiers  ·  {total_thesis}", expanded=False):
        st.markdown(
            '<p style="font-size:11px;color:#475569;margin:0 0 8px">'
            'Independent signal layer — squeeze · WVF · pullback · momentum · confluence · D* · yield gate</p>',
            unsafe_allow_html=True,
        )
        for tk in tier_order:
            meta  = _THESIS_TIER_META[tk]
            tdf   = df[df["T2_Tier"] == tk].copy()
            count = len(tdf)
            if count == 0:
                continue
            dot   = meta["dot"]
            with st.expander(
                f'{"●"} {tk}  ·  {count}',
                expanded=(tk in ("T2★", "T2A")),
            ):
                st.markdown(
                    f'<p style="font-size:11px;color:#475569;margin:0 0 6px">{meta["desc"]}</p>',
                    unsafe_allow_html=True,
                )
                _render_table(tdf, cci_ob, cci_os, watchlist_syms)
 
 
# ══════════════════════════════════════════════════════════════════
#  SUMMARY PILL BAR
# ══════════════════════════════════════════════════════════════════
 
def _summary_bar(df: pd.DataFrame) -> str:
    if df.empty:
        t1 = t2 = t3 = t4 = golden = cci_buy = cci_exit = cci_ext = 0
        tt_prime = tt_strong = tt_watch = tt_weak = 0
    else:
        t1       = int(df["_tier1_prime"].sum())                      if "_tier1_prime" in df.columns else 0
        t2       = int((df["_any_buy"] & ~df["_tier1_prime"]).sum())  if "_any_buy" in df.columns and "_tier1_prime" in df.columns else 0
        t3       = int((df["Action"] == "👁 WATCH").sum())
        t4       = int((df["Action"] == "⛔ SKIP").sum())
        golden   = int(df["_in_golden"].sum())                        if "_in_golden"   in df.columns else 0
        cci_buy  = int((df["CCI Sig"] == "BUY").sum())
        cci_exit = int((df["CCI Sig"] == "EXIT").sum())
        cci_ext  = int((df["CCI Sig"] == "EXT").sum())
        # Thesis tier counts
        if "T2_Tier" in df.columns:
            tt_prime  = int((df["T2_Tier"] == "T2★").sum())
            tt_strong = int((df["T2_Tier"] == "T2A").sum())
            tt_watch  = int((df["T2_Tier"] == "T2B").sum())
            tt_weak   = int((df["T2_Tier"] == "T2C").sum())
        else:
            tt_prime = tt_strong = tt_watch = tt_weak = 0
 
    pills = [
        ("#166534", "#4ade80", f"Tier 1 · {t1}"),
        ("#1e3a5f", "#60a5fa", f"Tier 2 · {t2}"),
        ("#78350f", "#fcd34d", f"Tier 3 · {t3}"),
        ("#7f1d1d", "#fca5a5", f"Tier 4 · {t4}"),
        ("#0f2d2d", "#2dd4bf", f"Golden Zone · {golden}"),
        ("#2e1065", "#c4b5fd", f"CCI Buy · {cci_buy}"),
        ("#831843", "#f9a8d4", f"CCI Exit · {cci_exit}"),
        ("#1c1917", "#a8a29e", f"CCI Ext · {cci_ext}"),
        # Thesis tiers
        ("#1a3a1a", "#4ade80", f"T2★ · {tt_prime}"),
        ("#1e3a5f", "#60a5fa", f"T2A · {tt_strong}"),
        ("#78350f", "#fcd34d", f"T2B · {tt_watch}"),
        ("#3b1f2b", "#f9a8d4", f"T2C · {tt_weak}"),
    ]
    spans = "".join(
        f'<span style="background:{bg};color:{fg};padding:4px 12px;border-radius:20px;'
        f'font-size:12px;font-weight:500;white-space:nowrap;border:1px solid {fg}22">{lbl}</span>'
        for bg, fg, lbl in pills
    )
    return (
        '<div style="position:sticky;bottom:0;background:#050b14;'
        'border-top:1px solid #1e293b;padding:8px 2px 6px;'
        f'display:flex;flex-wrap:wrap;gap:6px;z-index:100">{spans}</div>'
    )
 
 
# ══════════════════════════════════════════════════════════════════
#  METRICS ROW
# ══════════════════════════════════════════════════════════════════
 
def _render_metrics(df: pd.DataFrame):
    if df.empty:
        return
    t1  = int(df["_tier1_prime"].sum())        if "_tier1_prime" in df.columns else 0
    ab  = int(df["_any_buy"].sum())             if "_any_buy"     in df.columns else 0
    hp  = int(df["_high_prob"].sum())           if "_high_prob"   in df.columns else 0
    cb  = int((df["CCI Sig"] == "BUY").sum())
    qs  = int((df["Qual"] == "⭐").sum())
    buy = int((df["Action"] == "✅ BUY").sum())
    at1 = int((df["AccTier"] == "T1★").sum())  if "AccTier"    in df.columns else 0
    aa  = int((df["AccTier"] == "A"  ).sum())  if "AccTier"    in df.columns else 0
    stp = int(df["_hard_stop"].sum())           if "_hard_stop" in df.columns else 0
    # Thesis tiers
    if "T2_Tier" in df.columns:
        tt_prime  = int((df["T2_Tier"] == "T2★").sum())
        tt_strong = int((df["T2_Tier"] == "T2A").sum())
        tt_watch  = int((df["T2_Tier"] == "T2B").sum())
        tt_weak   = int((df["T2_Tier"] == "T2C").sum())
    else:
        tt_prime = tt_strong = tt_watch = tt_weak = 0
 
    # Row 1 — original metrics
    cols = st.columns(9)
    for col, (lbl, val) in zip(cols, [
        ("🏆 Tier 1",  t1),
        ("🥈 Any Buy", ab),
        ("🎯 Hi Prob", hp),
        ("📡 CCI ↑",   cb),
        ("⭐ Qual",    qs),
        ("✅ BUY",     buy),
        ("T1★ ~90%",  at1),
        ("A ~85%",     aa),
        ("🚫 Stops",   stp),
    ]):
        col.metric(lbl, val)
 
    # Row 2 — thesis tier metrics
    st.markdown(
        '<p style="font-size:10px;color:#334155;text-transform:uppercase;'
        'letter-spacing:0.06em;margin:6px 0 2px">Thesis Tiers</p>',
        unsafe_allow_html=True,
    )
    t_cols = st.columns(4)
    for col, (lbl, val) in zip(t_cols, [
        ("T2★ Prime",  tt_prime),
        ("T2A Strong", tt_strong),
        ("T2B Watch",  tt_watch),
        ("T2C Weak",   tt_weak),
    ]):
        col.metric(lbl, val)
 
 
# ══════════════════════════════════════════════════════════════════
#  WATCHLIST SECTION
# ══════════════════════════════════════════════════════════════════
 
def _render_watchlist(df: pd.DataFrame, cci_ob: int, cci_os: int,
                      supabase_ok: bool):
    st.markdown(
        '<p style="font-size:12px;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">⭐ Watchlist</p>',
        unsafe_allow_html=True,
    )
 
    wl: list[dict] = st.session_state.get("watchlist", [])
    wl_syms = [w["symbol"] for w in wl]
 
    left, right = st.columns([3, 1])
 
    with left:
        if not wl_syms:
            st.markdown(
                '<p style="color:#334155;font-size:12px">No symbols yet — add one →</p>',
                unsafe_allow_html=True,
            )
        else:
            selected   = st.session_state.get("wl_selected", None)
            pills_html = ""
            for sym in wl_syms:
                active     = "active" if sym == selected else ""
                pills_html += f'<span class="wl-pill {active}">{sym}</span>'
            st.markdown(
                f'<div style="margin-bottom:8px">{pills_html}</div>',
                unsafe_allow_html=True,
            )
 
            pick = st.selectbox(
                "wl_pick_hidden",
                ["— none —"] + wl_syms,
                key="wl_pick",
                label_visibility="collapsed",
            )
            st.session_state["wl_selected"] = pick if pick != "— none —" else None
 
            if pick != "— none —" and not df.empty:
                match = df[df["Stock"] == pick]
                if not match.empty:
                    st.markdown(
                        f'<p style="font-size:12px;font-weight:600;color:#60a5fa;margin:6px 0 2px">'
                        f'{pick}</p>',
                        unsafe_allow_html=True,
                    )
                    _render_table(match, cci_ob, cci_os, set(wl_syms))
                else:
                    st.caption(f"{pick} — not in last scan.")
 
    with right:
        st.markdown(
            '<p style="font-size:11px;color:#64748b;margin-bottom:4px">Add symbol</p>',
            unsafe_allow_html=True,
        )
        wl_sym = st.text_input("sym_input", placeholder="e.g. RELIANCE",
                               label_visibility="collapsed", key="wl_sym_input")
        if st.button("＋ Add", use_container_width=True, key="wl_add_btn"):
            if wl_sym.strip():
                sym = wl_sym.strip().upper()
                if supabase_ok:
                    ok = add_to_watchlist(sym, "")
                    st.success(f"✅ {sym} added.") if ok else st.error("❌ Supabase error.")
                else:
                    wl = st.session_state.setdefault("watchlist", [])
                    if sym not in [w["symbol"] for w in wl]:
                        wl.append({"symbol": sym, "notes": ""})
                        st.success(f"✅ {sym}")
                    else:
                        st.info(f"{sym} already in list.")
            else:
                st.warning("Enter a symbol.")
 
        if wl_syms:
            rm = st.selectbox("Remove", ["—"] + wl_syms,
                              key="wl_rm", label_visibility="collapsed")
            if st.button("✕ Remove", use_container_width=True, key="wl_rm_btn"):
                if rm != "—":
                    st.session_state["watchlist"] = [
                        w for w in st.session_state.get("watchlist", [])
                        if w["symbol"] != rm
                    ]
                    st.rerun()
 
 
# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
 
def render(settings: dict) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
 
    symbols      = settings.get("symbols",      NIFTY500_SYMBOLS)
    cci_len      = settings.get("cci_len",      20)
    cci_ob       = settings.get("cci_ob",       100)
    cci_os       = settings.get("cci_os",      -100)
    workers      = settings.get("workers",      10)
    auto_refresh = settings.get("auto_refresh", False)
    refresh_secs = settings.get("refresh_mins", 5) * 60
    supabase_ok  = _is_available()
 
    # ── HEADER ────────────────────────────────────────────────────
    st.markdown(
        '<div class="scanner-header">'
        '<span style="font-size:18px">⚡</span>'
        '<span class="scanner-title">NSE Master Scanner</span>'
        '<span class="scanner-badge">LIVE · Nifty 500</span>'
        '<span style="font-size:10px;padding:2px 7px;border-radius:20px;background:#1e1b4b;'
        'color:#818cf8;border:1px solid #3730a3;margin-left:4px">v6 · Thesis</span>'
        f'<span style="margin-left:auto;font-size:11px;color:{"#4ade80" if supabase_ok else "#f87171"}">'
        f'{"● Supabase" if supabase_ok else "● Offline"}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
 
    # ── CONTROL ROW ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1, 3, 2, 2])
    with c1:
        run_btn = st.button("🔍 Run Scan", type="primary", use_container_width=True)
    with c2:
        search = st.text_input("search", placeholder="🔎  Search symbol…  e.g. RELIANCE, TCS",
                               label_visibility="collapsed", key="search_input")
    with c3:
       hi_prob_only = st.toggle("🎯 Hi Prob", value=False, key="hi_prob_toggle",
                             help="trend_up · in_golden · score ≥ 55")
    with c4:
        snap_label = st.text_input("snap", placeholder="Snapshot label (optional)",
                                   label_visibility="collapsed", key="snap_input")
    if auto_refresh:
        st.info(f"🔄 Auto-refresh every {settings.get('refresh_mins', 5)} min", icon="⏱")
 
    # ── SCAN ──────────────────────────────────────────────────────
    if run_btn:
        st.session_state.pop("scan_df", None)
        st.session_state["last_auto_scan"] = time.time()
 
    if run_btn or "scan_df" not in st.session_state:
        prog = st.progress(0.0, text="Initialising…")
        with st.spinner("Fetching & scoring Nifty 500…"):
            df_raw = run_scanner(
                symbols=symbols, cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os,
                max_workers=workers,
                progress_cb=lambda p: prog.progress(p, text=f"Scanning… {int(p*100)}%"),
            )
        prog.empty()
        if df_raw.empty:
            st.warning("No results — check symbols or data source.")
            return
        st.session_state["scan_df"] = df_raw
        st.session_state["scan_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.setdefault("last_auto_scan", time.time())
        if supabase_ok:
            with st.spinner("Saving snapshot…"):
                ok = save_scan_snapshot(df_raw, label=snap_label)
            st.toast("✅ Snapshot saved." if ok else "⚠️ Supabase save failed.")
 
    df = st.session_state.get("scan_df", pd.DataFrame())
    if df.empty:
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#334155">'
            '<div style="font-size:32px">📡</div>'
            '<div style="font-size:14px;margin-top:8px">Press <b>Run Scan</b> to start</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return
 
    ts = st.session_state.get("scan_ts", "")
    st.markdown(
        f'<p style="font-size:11px;color:#334155;margin:0 0 8px">'
        f'Last scan: <b style="color:#64748b">{ts}</b> · {len(df)} stocks scored</p>',
        unsafe_allow_html=True,
    )
 
    # ── METRICS ───────────────────────────────────────────────────
    _render_metrics(df)
    st.divider()
 
    # ── SEARCH FILTER ─────────────────────────────────────────────
    fdf = df.copy()
    if search.strip():
        fdf = fdf[fdf["Stock"].str.contains(search.strip(), case=False, na=False)]
    if hi_prob_only and "_high_prob" in fdf.columns:
      fdf = fdf[fdf["_high_prob"] == True]
    # ── PARTITION INTO TIERS ──────────────────────────────────────
    wl_syms_set = set(w["symbol"] for w in st.session_state.get("watchlist", []))
 
    has_t1 = "_tier1_prime" in fdf.columns
    has_ab = "_any_buy"     in fdf.columns
 
    mask_t1 = fdf["_tier1_prime"] if has_t1 else pd.Series(False, index=fdf.index)
    mask_ab = fdf["_any_buy"]     if has_ab else pd.Series(False, index=fdf.index)
 
    sort_col = "AccScore" if "AccScore" in fdf.columns else "Score"
 
    df_t1 = fdf[mask_t1].sort_values(sort_col, ascending=False)
    df_t2 = fdf[mask_ab & ~mask_t1].sort_values("Score", ascending=False)
    df_t3 = fdf[fdf["Action"] == "👁 WATCH"].sort_values("Score", ascending=False)
    df_t4 = fdf[fdf["Action"] == "⛔ SKIP"].sort_values("Score", ascending=False)
 
    _tier_expander("Tier 1", df_t1, cci_ob, cci_os, wl_syms_set, expanded=True)
    _tier_expander("Tier 2", df_t2, cci_ob, cci_os, wl_syms_set, expanded=False)
    _tier_expander("Tier 3", df_t3, cci_ob, cci_os, wl_syms_set, expanded=False)
    _tier_expander("Tier 4", df_t4, cci_ob, cci_os, wl_syms_set, expanded=False)
    _thesis_tier_expander(fdf, cci_ob, cci_os, wl_syms_set)
 
    # ── SUMMARY PILL BAR ──────────────────────────────────────────
    st.markdown(_summary_bar(df), unsafe_allow_html=True)
 
    st.divider()
 
    # ── WATCHLIST ─────────────────────────────────────────────────
    _render_watchlist(df, cci_ob, cci_os, supabase_ok)
 
    st.divider()
 
    # ── CSV DOWNLOAD ──────────────────────────────────────────────
    csv   = fdf.drop(columns=[c for c in fdf.columns if c.startswith("_")], errors="ignore")
    fname = f"scan_{ts.replace(':','-').replace(' ','_')}.csv"
    st.download_button("⬇️ Download CSV", data=csv.to_csv(index=False),
                       file_name=fname, mime="text/csv")
 
    # ── AUTO-REFRESH ──────────────────────────────────────────────
    if auto_refresh and "scan_df" in st.session_state:
        last      = st.session_state.get("last_auto_scan", time.time())
        remaining = max(0, int(refresh_secs - (time.time() - last)))
        box       = st.empty()
        if remaining > 0:
            box.caption(f"🔄 Refresh in {remaining // 60}m {remaining % 60:02d}s")
            time.sleep(1)
            st.rerun()
        else:
            st.session_state.pop("scan_df", None)
            st.session_state["last_auto_scan"] = time.time()
            st.rerun()
 


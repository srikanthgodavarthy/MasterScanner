pages/settings.py — Settings Page (v8 — full HTML component UI)

The visual settings UI is rendered as a single st.components.v1.html block
with native HTML sliders, toggles, and chip controls — bypassing Streamlit's
default widget styling entirely.  Values sync back to st.session_state via
query params on every change (postMessage bridge).

Watchlist and System sections remain as Streamlit expanders since they need
real Streamlit widgets (dataframe, buttons, text_area).

Zero changes to settings dict keys, DEFAULTS, or the returned dict.
"""

import streamlit as st
import pandas as pd
import json

from utils.supabase_client import (
    get_client, load_watchlist, save_watchlist,
    add_to_watchlist, remove_from_watchlist,
    load_scan_history, _is_available, SCHEMA_SQL,
)
from utils.scanner_engine import NIFTY500_SYMBOLS

# ══════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════

DEFAULTS = {
    "universe_mode":     "Nifty 500 (default)",
    "custom_symbols":    [],
    "workers":           15,
    "hold_days":         15,
    "auto_refresh":      False,
    "refresh_mins":      5,
    "min_score":         65,
    "execute_threshold": 72,
    "cci_len":           20,
    "cci_ob":            100,
    "cci_os":           -100,
    "t1_mom3":           10,
    "t1_mom6":           18,
    "t1_fib_hi":         38.2,
    "t1_fib_lo":         61.8,
    "t1_cci_window":     4,
    "t1_cloud":          True,
    "t1_squeeze_boost":  True,
    "t1_squeeze_pts":    10,
    "t1_no_squeeze_pts": 0,
    "t1_ps_weight":      20,
    "t1_ps_penalty":     -5,
    "t1_rs_min":         0.01,
    "t1_adx_min":        23,
    "t1_use_adx":        True,
    "t2_comp_bars":      12,
    "t2_atr_ratio":      0.80,
    "t2_vol_mult":       1.5,
    "nifty_regime_filter": True,
    "trading_style":       "Balanced",
    "entry_preference":    "Pullback",
    "extension_tolerance": "Normal",
    "min_risk_reward":     "2R",
    "conviction_level":    "Actionable",
}

def _g(key, default=None):
    return st.session_state.get(key, DEFAULTS.get(key, default))

def _s(key, val):
    st.session_state[key] = val

# ══════════════════════════════════════════════════════════════════
#  SYNC: apply URL params → session_state  (called on every render)
# ══════════════════════════════════════════════════════════════════

def _sync_params() -> None:
    """Read ?settings_json=... query param and push values into session_state."""
    params = st.query_params
    raw = params.get("settings_json", None)
    if not raw:
        return
    try:
        data = json.loads(raw)
        for k, v in data.items():
            if k in DEFAULTS:
                st.session_state[k] = v
        # Clear the param so it doesn't re-apply on next natural rerun
        st.query_params.clear()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
#  HTML SETTINGS COMPONENT
# ══════════════════════════════════════════════════════════════════

def _settings_html(vals: dict) -> str:
    """Build and return the full settings HTML component string."""

    def jb(v):  # JSON-safe bool
        return "true" if v else "false"

    v = vals  # shorthand

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0d1117;
  color: #e6edf3;
  font-size: 13px;
  padding: 16px 20px 24px;
}}

/* ── Typography ── */
.page-title {{
  font-size: 15px;
  font-weight: 700;
  color: #e6edf3;
  margin-bottom: 2px;
}}
.page-sub {{
  font-size: 11px;
  color: #8b949e;
  margin-bottom: 20px;
}}

/* ── Accordion ── */
.acc {{
  background: #161b22;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 10px;
  margin-bottom: 10px;
  overflow: hidden;
}}
.acc-head {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 11px 16px;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  transition: background 0.12s;
}}
.acc-head:hover {{ background: #1c2333; }}
.acc-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
.acc-title {{ font-size: 12px; font-weight: 700; color: #e6edf3; }}
.acc-sub {{
  font-size: 10px;
  color: #8b949e;
  margin-left: auto;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 420px;
}}
.acc-arrow {{
  font-size: 10px;
  color: #8b949e;
  margin-left: 6px;
  transition: transform 0.2s;
  flex-shrink: 0;
}}
.acc-arrow.open {{ transform: rotate(180deg); }}
.acc-body {{
  padding: 16px;
  display: none;
  border-top: 1px solid rgba(255,255,255,0.05);
}}
.acc-body.open {{ display: block; }}

/* ── Two-column grid ── */
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px 24px; }}
.one-col {{ margin-top: 12px; }}

/* ── Field ── */
.field {{ margin-bottom: 14px; }}
.f-label {{
  display: block;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #8b949e;
  margin-bottom: 8px;
}}

/* ── Chip row ── */
.chip-row {{ display: flex; gap: 5px; flex-wrap: wrap; }}
.chip {{
  padding: 5px 14px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
  border: 1px solid rgba(255,255,255,0.1);
  background: #1c2333;
  color: #8b949e;
  cursor: pointer;
  transition: all 0.12s;
  white-space: nowrap;
}}
.chip:hover {{ border-color: rgba(255,255,255,0.2); color: #c9d1d9; }}
.chip.on {{
  background: rgba(88,166,255,0.12);
  border-color: rgba(88,166,255,0.45);
  color: #58a6ff;
}}

/* ── Slider ── */
.sl-wrap {{ display: flex; align-items: center; gap: 12px; }}
.sl-wrap input[type=range] {{
  -webkit-appearance: none;
  appearance: none;
  flex: 1;
  height: 4px;
  border-radius: 2px;
  background: #30363d;
  outline: none;
  cursor: pointer;
}}
.sl-wrap input[type=range]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: #58a6ff;
  cursor: pointer;
  border: 2px solid #0d1117;
  transition: background 0.15s, transform 0.1s;
}}
.sl-wrap input[type=range]::-webkit-slider-thumb:hover {{
  background: #79b8ff;
  transform: scale(1.15);
}}
.sl-wrap input[type=range]::-moz-range-thumb {{
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: #58a6ff;
  cursor: pointer;
  border: 2px solid #0d1117;
}}
.sl-val {{
  font-size: 12px;
  font-weight: 600;
  color: #58a6ff;
  min-width: 40px;
  text-align: right;
  font-family: 'JetBrains Mono', monospace;
}}

/* ── Toggle ── */
.tog-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 0;
}}
.tog-info {{ flex: 1; }}
.tog-name {{ font-size: 12px; font-weight: 600; color: #e6edf3; }}
.tog-hint {{ font-size: 10px; color: #8b949e; margin-top: 1px; }}
.tog-sw {{
  position: relative;
  width: 36px;
  height: 20px;
  flex-shrink: 0;
  margin-left: 16px;
}}
.tog-sw input {{ opacity: 0; position: absolute; width: 0; height: 0; }}
.tog-track {{
  position: absolute;
  inset: 0;
  background: #30363d;
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 20px;
  cursor: pointer;
  transition: background 0.2s, border-color 0.2s;
}}
.tog-track::after {{
  content: '';
  position: absolute;
  left: 3px; top: 3px;
  width: 12px; height: 12px;
  border-radius: 50%;
  background: #8b949e;
  transition: transform 0.2s, background 0.2s;
}}
.tog-sw input:checked + .tog-track {{
  background: rgba(63,185,80,0.18);
  border-color: rgba(63,185,80,0.5);
}}
.tog-sw input:checked + .tog-track::after {{
  transform: translateX(16px);
  background: #3fb950;
}}

/* ── Divider ── */
.divider {{ height: 1px; background: rgba(255,255,255,0.06); margin: 14px 0; }}

/* ── Gate preview ── */
.gate-pre {{
  background: #0a0f16;
  border: 1px solid rgba(63,185,80,0.15);
  border-left: 2px solid rgba(63,185,80,0.4);
  border-radius: 0 7px 7px 0;
  padding: 10px 14px;
  font-size: 11px;
  line-height: 1.85;
  color: #8b949e;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  margin-top: 12px;
  white-space: pre-wrap;
}}
.gate-pre .b  {{ color: #58a6ff; font-weight: 700; }}
.gate-pre .ok {{ color: #3fb950; }}
.gate-pre .w  {{ color: #d29922; }}
.gate-pre .e  {{ color: #f85149; }}

/* ── Section divider label ── */
.sec-label {{
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #484f58;
  margin: 16px 0 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}}

/* ── Save bar ── */
.save-bar {{
  position: sticky;
  bottom: 0;
  background: #0d1117;
  border-top: 1px solid rgba(255,255,255,0.08);
  padding: 12px 0 0;
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.save-btn {{
  background: rgba(63,185,80,0.1);
  border: 1px solid rgba(63,185,80,0.4);
  color: #3fb950;
  padding: 8px 24px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  letter-spacing: 0.04em;
  transition: all 0.15s;
}}
.save-btn:hover {{
  background: rgba(63,185,80,0.18);
  border-color: rgba(63,185,80,0.7);
}}
.save-status {{ font-size: 11px; color: #8b949e; }}
</style>
</head>
<body>

<div class="page-title">⚙ Settings</div>
<div class="page-sub">Changes take effect on the next Run Scan or Backtest.</div>

<!-- ══ TRADING STYLE ══════════════════════════════════════════ -->
<div class="acc" id="acc-style">
  <div class="acc-head" onclick="toggleAcc('acc-style')">
    <span class="acc-dot" style="background:#58a6ff"></span>
    <span class="acc-title">Trading style</span>
    <span class="acc-sub" id="sub-style">{v['trading_style']} · {v['entry_preference']} · {v['min_risk_reward']}</span>
    <span class="acc-arrow open" id="arr-acc-style">▼</span>
  </div>
  <div class="acc-body open" id="body-acc-style">
    <div class="two-col">
      <div>
        <div class="field">
          <span class="f-label">Style</span>
          <div class="chip-row" id="chips-trading_style">
            <div class="chip{'  on' if v['trading_style']=='Aggressive' else ''}" data-group="trading_style" data-val="Aggressive" onclick="pickChip(this)">Aggressive</div>
            <div class="chip{'  on' if v['trading_style']=='Balanced' else ''}" data-group="trading_style" data-val="Balanced" onclick="pickChip(this)">Balanced</div>
            <div class="chip{'  on' if v['trading_style']=='Conservative' else ''}" data-group="trading_style" data-val="Conservative" onclick="pickChip(this)">Conservative</div>
          </div>
        </div>
        <div class="field">
          <span class="f-label">Entry type</span>
          <div class="chip-row" id="chips-entry_preference">
            <div class="chip{'  on' if v['entry_preference']=='Early' else ''}" data-group="entry_preference" data-val="Early" onclick="pickChip(this)">Early</div>
            <div class="chip{'  on' if v['entry_preference']=='Pullback' else ''}" data-group="entry_preference" data-val="Pullback" onclick="pickChip(this)">Pullback</div>
            <div class="chip{'  on' if v['entry_preference']=='Breakout' else ''}" data-group="entry_preference" data-val="Breakout" onclick="pickChip(this)">Breakout</div>
          </div>
        </div>
      </div>
      <div>
        <div class="field">
          <span class="f-label">Min R:R</span>
          <div class="chip-row" id="chips-min_risk_reward">
            <div class="chip{'  on' if v['min_risk_reward']=='1.5R' else ''}" data-group="min_risk_reward" data-val="1.5R" onclick="pickChip(this)">1.5R</div>
            <div class="chip{'  on' if v['min_risk_reward']=='2R' else ''}" data-group="min_risk_reward" data-val="2R" onclick="pickChip(this)">2R</div>
            <div class="chip{'  on' if v['min_risk_reward']=='3R' else ''}" data-group="min_risk_reward" data-val="3R" onclick="pickChip(this)">3R</div>
          </div>
        </div>
        <div class="field">
          <span class="f-label">Extension tolerance</span>
          <div class="chip-row" id="chips-extension_tolerance">
            <div class="chip{'  on' if v['extension_tolerance']=='Very Strict' else ''}" data-group="extension_tolerance" data-val="Very Strict" onclick="pickChip(this)">Very strict</div>
            <div class="chip{'  on' if v['extension_tolerance']=='Strict' else ''}" data-group="extension_tolerance" data-val="Strict" onclick="pickChip(this)">Strict</div>
            <div class="chip{'  on' if v['extension_tolerance']=='Normal' else ''}" data-group="extension_tolerance" data-val="Normal" onclick="pickChip(this)">Normal</div>
            <div class="chip{'  on' if v['extension_tolerance']=='Loose' else ''}" data-group="extension_tolerance" data-val="Loose" onclick="pickChip(this)">Loose</div>
          </div>
        </div>
      </div>
    </div>
    <div class="field">
      <span class="f-label">Min conviction</span>
      <div class="chip-row" id="chips-conviction_level">
        <div class="chip{'  on' if v['conviction_level']=='Watchlist' else ''}" data-group="conviction_level" data-val="Watchlist" onclick="pickChip(this)">Watchlist</div>
        <div class="chip{'  on' if v['conviction_level']=='Actionable' else ''}" data-group="conviction_level" data-val="Actionable" onclick="pickChip(this)">Actionable</div>
        <div class="chip{'  on' if v['conviction_level']=='High Conviction' else ''}" data-group="conviction_level" data-val="High Conviction" onclick="pickChip(this)">High conviction</div>
        <div class="chip{'  on' if v['conviction_level']=='Elite' else ''}" data-group="conviction_level" data-val="Elite" onclick="pickChip(this)">Elite</div>
      </div>
    </div>
  </div>
</div>

<!-- ══ UNIVERSE ══════════════════════════════════════════════ -->
<div class="acc" id="acc-universe">
  <div class="acc-head" onclick="toggleAcc('acc-universe')">
    <span class="acc-dot" style="background:#a371f7"></span>
    <span class="acc-title">Universe & thresholds</span>
    <span class="acc-sub" id="sub-universe">Workers {v['workers']} · Hold {v['hold_days']}d · Score {v['min_score']} · Execute {v['execute_threshold']}</span>
    <span class="acc-arrow" id="arr-acc-universe">▼</span>
  </div>
  <div class="acc-body" id="body-acc-universe">
    <div class="two-col">
      <div class="field">
        <span class="f-label">Workers</span>
        <div class="sl-wrap">
          <input type="range" min="1" max="20" step="1" value="{v['workers']}"
            oninput="slUpdate(this,'workers',1)" id="sl-workers">
          <span class="sl-val" id="sv-workers">{v['workers']}</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">Hold days</span>
        <div class="sl-wrap">
          <input type="range" min="5" max="60" step="5" value="{v['hold_days']}"
            oninput="slUpdate(this,'hold_days',1)" id="sl-hold_days">
          <span class="sl-val" id="sv-hold_days">{v['hold_days']}</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">Min score (display)</span>
        <div class="sl-wrap">
          <input type="range" min="50" max="85" step="5" value="{v['min_score']}"
            oninput="slUpdate(this,'min_score',1)" id="sl-min_score">
          <span class="sl-val" id="sv-min_score">{v['min_score']}</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">Execute threshold</span>
        <div class="sl-wrap">
          <input type="range" min="50" max="85" step="5" value="{v['execute_threshold']}"
            oninput="slUpdate(this,'execute_threshold',1)" id="sl-execute_threshold">
          <span class="sl-val" id="sv-execute_threshold">{v['execute_threshold']}</span>
        </div>
      </div>
    </div>
    <div class="tog-row">
      <div class="tog-info">
        <div class="tog-name">Auto-refresh</div>
        <div class="tog-hint">Re-run scan on a timer</div>
      </div>
      <label class="tog-sw">
        <input type="checkbox" id="tog-auto_refresh" {'checked' if v['auto_refresh'] else ''} onchange="togUpdate(this,'auto_refresh')">
        <span class="tog-track"></span>
      </label>
    </div>
    <div class="tog-row">
      <div class="tog-info">
        <div class="tog-name">Bull Nifty regime required for Tier 1</div>
        <div class="tog-hint">Hard block when Nifty is bearish</div>
      </div>
      <label class="tog-sw">
        <input type="checkbox" id="tog-nifty_regime_filter" {'checked' if v['nifty_regime_filter'] else ''} onchange="togUpdate(this,'nifty_regime_filter')">
        <span class="tog-track"></span>
      </label>
    </div>
  </div>
</div>

<!-- ══ TIER 1 ═════════════════════════════════════════════════ -->
<div class="acc" id="acc-t1">
  <div class="acc-head" onclick="toggleAcc('acc-t1')">
    <span class="acc-dot" style="background:#d29922"></span>
    <span class="acc-title">Tier 1 — Prime gate</span>
    <span class="acc-sub" id="sub-t1">mom3 {v['t1_mom3']}% · Fib {v['t1_fib_hi']}–{v['t1_fib_lo']} · CCI {v['t1_cci_window']}-bar · ADX {v['t1_adx_min']}</span>
    <span class="acc-arrow" id="arr-acc-t1">▼</span>
  </div>
  <div class="acc-body" id="body-acc-t1">
    <div class="two-col">
      <div class="field">
        <span class="f-label">3-month momentum min</span>
        <div class="sl-wrap">
          <input type="range" min="0" max="25" step="1" value="{v['t1_mom3']}"
            oninput="slUpdate(this,'t1_mom3',1,'%')" id="sl-t1_mom3">
          <span class="sl-val" id="sv-t1_mom3">{v['t1_mom3']}%</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">6-month momentum min</span>
        <div class="sl-wrap">
          <input type="range" min="0" max="40" step="1" value="{v['t1_mom6']}"
            oninput="slUpdate(this,'t1_mom6',1,'%')" id="sl-t1_mom6">
          <span class="sl-val" id="sv-t1_mom6">{v['t1_mom6']}%</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">Fib upper (shallow pullback)</span>
        <div class="chip-row" id="chips-t1_fib_hi">
          <div class="chip{'  on' if v['t1_fib_hi']==23.6 else ''}" data-group="t1_fib_hi" data-val="23.6" data-num="true" onclick="pickChip(this)">23.6%</div>
          <div class="chip{'  on' if v['t1_fib_hi']==38.2 else ''}" data-group="t1_fib_hi" data-val="38.2" data-num="true" onclick="pickChip(this)">38.2%</div>
          <div class="chip{'  on' if v['t1_fib_hi']==50.0 else ''}" data-group="t1_fib_hi" data-val="50.0" data-num="true" onclick="pickChip(this)">50.0%</div>
        </div>
      </div>
      <div class="field">
        <span class="f-label">Fib lower (deep pullback)</span>
        <div class="chip-row" id="chips-t1_fib_lo">
          <div class="chip{'  on' if v['t1_fib_lo']==50.0 else ''}" data-group="t1_fib_lo" data-val="50.0" data-num="true" onclick="pickChip(this)">50.0%</div>
          <div class="chip{'  on' if v['t1_fib_lo']==61.8 else ''}" data-group="t1_fib_lo" data-val="61.8" data-num="true" onclick="pickChip(this)">61.8%</div>
          <div class="chip{'  on' if v['t1_fib_lo']==78.6 else ''}" data-group="t1_fib_lo" data-val="78.6" data-num="true" onclick="pickChip(this)">78.6%</div>
        </div>
      </div>
      <div class="field">
        <span class="f-label">CCI recovery lookback</span>
        <div class="sl-wrap">
          <input type="range" min="1" max="10" step="1" value="{v['t1_cci_window']}"
            oninput="slUpdate(this,'t1_cci_window',1,' bars')" id="sl-t1_cci_window">
          <span class="sl-val" id="sv-t1_cci_window">{v['t1_cci_window']} bars</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">ADX min</span>
        <div class="sl-wrap">
          <input type="range" min="10" max="40" step="1" value="{v['t1_adx_min']}"
            oninput="slUpdate(this,'t1_adx_min',1)" id="sl-t1_adx_min">
          <span class="sl-val" id="sv-t1_adx_min">{v['t1_adx_min']}</span>
        </div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="two-col">
      <div>
        <div class="tog-row">
          <div class="tog-info">
            <div class="tog-name">Require above / inside cloud</div>
            <div class="tog-hint">Price below cloud = overhead resistance</div>
          </div>
          <label class="tog-sw">
            <input type="checkbox" id="tog-t1_cloud" {'checked' if v['t1_cloud'] else ''} onchange="togUpdate(this,'t1_cloud')">
            <span class="tog-track"></span>
          </label>
        </div>
        <div class="tog-row">
          <div class="tog-info">
            <div class="tog-name">Use ADX gate</div>
            <div class="tog-hint">Off = EMA20 slope gate instead</div>
          </div>
          <label class="tog-sw">
            <input type="checkbox" id="tog-t1_use_adx" {'checked' if v['t1_use_adx'] else ''} onchange="togUpdate(this,'t1_use_adx')">
            <span class="tog-track"></span>
          </label>
        </div>
      </div>
      <div>
        <div class="tog-row">
          <div class="tog-info">
            <div class="tog-name">Squeeze boost</div>
            <div class="tog-hint">+pts when BB squeeze releases</div>
          </div>
          <label class="tog-sw">
            <input type="checkbox" id="tog-t1_squeeze_boost" {'checked' if v['t1_squeeze_boost'] else ''} onchange="togUpdate(this,'t1_squeeze_boost')">
            <span class="tog-track"></span>
          </label>
        </div>
      </div>
    </div>

    <div class="two-col" style="margin-top:12px">
      <div class="field">
        <span class="f-label">Squeeze release points</span>
        <div class="sl-wrap">
          <input type="range" min="0" max="30" step="5" value="{v['t1_squeeze_pts']}"
            oninput="slUpdate(this,'t1_squeeze_pts',1)" id="sl-t1_squeeze_pts">
          <span class="sl-val" id="sv-t1_squeeze_pts">{v['t1_squeeze_pts']}</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">Not-in-squeeze points</span>
        <div class="sl-wrap">
          <input type="range" min="0" max="15" step="5" value="{v['t1_no_squeeze_pts']}"
            oninput="slUpdate(this,'t1_no_squeeze_pts',1)" id="sl-t1_no_squeeze_pts">
          <span class="sl-val" id="sv-t1_no_squeeze_pts">{v['t1_no_squeeze_pts']}</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">Persistent strength bonus</span>
        <div class="sl-wrap">
          <input type="range" min="5" max="30" step="5" value="{v['t1_ps_weight']}"
            oninput="slUpdate(this,'t1_ps_weight',1)" id="sl-t1_ps_weight">
          <span class="sl-val" id="sv-t1_ps_weight">{v['t1_ps_weight']}</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">Persistent strength penalty</span>
        <div class="sl-wrap">
          <input type="range" min="-20" max="0" step="5" value="{v['t1_ps_penalty']}"
            oninput="slUpdate(this,'t1_ps_penalty',1)" id="sl-t1_ps_penalty">
          <span class="sl-val" id="sv-t1_ps_penalty">{v['t1_ps_penalty']}</span>
        </div>
      </div>
    </div>

    <div class="gate-pre" id="gate-t1"><!-- populated by JS --></div>
  </div>
</div>

<!-- ══ TIER 2 ═════════════════════════════════════════════════ -->
<div class="acc" id="acc-t2">
  <div class="acc-head" onclick="toggleAcc('acc-t2')">
    <span class="acc-dot" style="background:#3fb950"></span>
    <span class="acc-title">Tier 2 — Compression breakout</span>
    <span class="acc-sub" id="sub-t2">{v['t2_comp_bars']}-bar window · ATR {v['t2_atr_ratio']:.2f} · Vol {v['t2_vol_mult']:.1f}×</span>
    <span class="acc-arrow" id="arr-acc-t2">▼</span>
  </div>
  <div class="acc-body" id="body-acc-t2">
    <div class="two-col">
      <div class="field">
        <span class="f-label">Compression window</span>
        <div class="sl-wrap">
          <input type="range" min="5" max="20" step="1" value="{v['t2_comp_bars']}"
            oninput="slUpdate(this,'t2_comp_bars',1,' bars')" id="sl-t2_comp_bars">
          <span class="sl-val" id="sv-t2_comp_bars">{v['t2_comp_bars']} bars</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">ATR compression ratio</span>
        <div class="sl-wrap">
          <input type="range" min="60" max="95" step="5" value="{int(v['t2_atr_ratio']*100)}"
            oninput="slUpdateDiv100(this,'t2_atr_ratio')" id="sl-t2_atr_ratio">
          <span class="sl-val" id="sv-t2_atr_ratio">{v['t2_atr_ratio']:.2f}</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">Volume expansion</span>
        <div class="sl-wrap">
          <input type="range" min="10" max="30" step="1" value="{int(v['t2_vol_mult']*10)}"
            oninput="slUpdateDiv10(this,'t2_vol_mult')" id="sl-t2_vol_mult">
          <span class="sl-val" id="sv-t2_vol_mult">{v['t2_vol_mult']:.1f}×</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">CCI breakout threshold</span>
        <div class="sl-wrap">
          <input type="range" min="50" max="200" step="10" value="{v['cci_ob']}"
            oninput="slUpdate(this,'cci_ob',1)" id="sl-cci_ob">
          <span class="sl-val" id="sv-cci_ob">{v['cci_ob']}</span>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ══ CCI ════════════════════════════════════════════════════ -->
<div class="acc" id="acc-cci">
  <div class="acc-head" onclick="toggleAcc('acc-cci')">
    <span class="acc-dot" style="background:#8b949e"></span>
    <span class="acc-title">CCI parameters</span>
    <span class="acc-sub" id="sub-cci">Period {v['cci_len']} · OB {v['cci_ob']} · OS {v['cci_os']}</span>
    <span class="acc-arrow" id="arr-acc-cci">▼</span>
  </div>
  <div class="acc-body" id="body-acc-cci">
    <div class="two-col">
      <div class="field">
        <span class="f-label">Period</span>
        <div class="sl-wrap">
          <input type="range" min="5" max="50" step="1" value="{v['cci_len']}"
            oninput="slUpdate(this,'cci_len',1)" id="sl-cci_len">
          <span class="sl-val" id="sv-cci_len">{v['cci_len']}</span>
        </div>
      </div>
      <div class="field">
        <span class="f-label">Overbought</span>
        <div class="sl-wrap">
          <input type="range" min="50" max="300" step="10" value="{v['cci_ob']}"
            oninput="slUpdate(this,'cci_ob',1)" id="sl-cci_ob2">
          <span class="sl-val" id="sv-cci_ob2">{v['cci_ob']}</span>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ══ SAVE BAR ════════════════════════════════════════════════ -->
<div class="save-bar">
  <button class="save-btn" onclick="applySettings()">Apply settings</button>
  <span class="save-status" id="save-status">All changes are pending — click Apply to update the scanner.</span>
</div>

<script>
// ── State ────────────────────────────────────────────────────────
const S = {
  trading_style:       "{v['trading_style']}",
  entry_preference:    "{v['entry_preference']}",
  extension_tolerance: "{v['extension_tolerance']}",
  min_risk_reward:     "{v['min_risk_reward']}",
  conviction_level:    "{v['conviction_level']}",
  workers:             {v['workers']},
  hold_days:           {v['hold_days']},
  auto_refresh:        {jb(v['auto_refresh'])},
  refresh_mins:        {v['refresh_mins']},
  min_score:           {v['min_score']},
  execute_threshold:   {v['execute_threshold']},
  cci_len:             {v['cci_len']},
  cci_ob:              {v['cci_ob']},
  cci_os:              {v['cci_os']},
  t1_mom3:             {v['t1_mom3']},
  t1_mom6:             {v['t1_mom6']},
  t1_fib_hi:           {v['t1_fib_hi']},
  t1_fib_lo:           {v['t1_fib_lo']},
  t1_cci_window:       {v['t1_cci_window']},
  t1_cloud:            {jb(v['t1_cloud'])},
  t1_squeeze_boost:    {jb(v['t1_squeeze_boost'])},
  t1_squeeze_pts:      {v['t1_squeeze_pts']},
  t1_no_squeeze_pts:   {v['t1_no_squeeze_pts']},
  t1_ps_weight:        {v['t1_ps_weight']},
  t1_ps_penalty:       {v['t1_ps_penalty']},
  t1_rs_min:           {v['t1_rs_min']},
  t1_adx_min:          {v['t1_adx_min']},
  t1_use_adx:          {jb(v['t1_use_adx'])},
  t2_comp_bars:        {v['t2_comp_bars']},
  t2_atr_ratio:        {v['t2_atr_ratio']},
  t2_vol_mult:         {v['t2_vol_mult']},
  nifty_regime_filter: {jb(v['nifty_regime_filter'])},
}};

// ── Accordion ────────────────────────────────────────────────────
function toggleAcc(id) {{
  const body  = document.getElementById('body-' + id);
  const arrow = document.getElementById('arr-' + id);
  const open  = body.classList.contains('open');
  body.classList.toggle('open', !open);
  arrow.classList.toggle('open', !open);
}}

// ── Chip ─────────────────────────────────────────────────────────
function pickChip(el) {{
  const group = el.dataset.group;
  document.querySelectorAll('[data-group="' + group + '"]')
    .forEach(c => c.classList.remove('on'));
  el.classList.add('on');
  const raw = el.dataset.val;
  S[group] = el.dataset.num === 'true' ? parseFloat(raw) : raw;
  updateSubtitles();
  markPending();
}}

// ── Slider ───────────────────────────────────────────────────────
function slUpdate(el, key, decimals, suffix) {{
  const v = parseFloat(el.value);
  S[key] = v;
  const disp = suffix ? v.toFixed(decimals) + suffix : v.toFixed(decimals);
  document.getElementById('sv-' + key).textContent = disp;
  updateSubtitles();
  if (key === 't1_mom3' || key === 't1_mom6' || key === 't1_cci_window' || key === 't1_adx_min') updateGatePreview();
  markPending();
}}
function slUpdateDiv100(el, key) {{
  const v = parseFloat(el.value) / 100;
  S[key] = v;
  document.getElementById('sv-' + key).textContent = v.toFixed(2);
  updateSubtitles();
  markPending();
}}
function slUpdateDiv10(el, key) {{
  const v = parseFloat(el.value) / 10;
  S[key] = v;
  document.getElementById('sv-' + key).textContent = v.toFixed(1) + '×';
  updateSubtitles();
  markPending();
}}

// ── Toggle ───────────────────────────────────────────────────────
function togUpdate(el, key) {{
  S[key] = el.checked;
  updateSubtitles();
  updateGatePreview();
  markPending();
}}

// ── Subtitles ────────────────────────────────────────────────────
function updateSubtitles() {{
  const style = document.getElementById('sub-style');
  if (style) style.textContent = S.trading_style + ' · ' + S.entry_preference + ' · ' + S.min_risk_reward;

  const univ = document.getElementById('sub-universe');
  if (univ) univ.textContent = 'Workers ' + S.workers + ' · Hold ' + S.hold_days + 'd · Score ' + S.min_score + ' · Execute ' + S.execute_threshold;

  const t1 = document.getElementById('sub-t1');
  if (t1) t1.textContent = 'mom3 ' + S.t1_mom3 + '% · Fib ' + S.t1_fib_hi + '–' + S.t1_fib_lo + ' · CCI ' + S.t1_cci_window + '-bar · ADX ' + S.t1_adx_min;

  const t2 = document.getElementById('sub-t2');
  if (t2) t2.textContent = S.t2_comp_bars + '-bar window · ATR ' + S.t2_atr_ratio.toFixed(2) + ' · Vol ' + S.t2_vol_mult.toFixed(1) + '×';
}}

// ── Gate preview ─────────────────────────────────────────────────
function updateGatePreview() {{
  const el = document.getElementById('gate-t1');
  if (!el) return;
  const strength = S.t1_use_adx ? 'ADX > ' + S.t1_adx_min : 'EMA20 slope positive';
  const cloud    = S.t1_cloud ? ' AND above/inside cloud' : ' (cloud gate OFF)';
  const nifty    = S.nifty_regime_filter ? ' AND nifty_regime=bull' : '';
  el.innerHTML =
    '<span class="b">Path A — Pullback to structure</span>\\n' +
    'trend_up <span class="ok">AND</span> in_golden [' + S.t1_fib_lo + '%..' + S.t1_fib_hi + '%]' + cloud + '\\n' +
    '<span class="ok">AND</span> cci_recovery (crossed > ' + S.cci_os + ' in last <b>' + S.t1_cci_window + '</b> bars)\\n' +
    '<span class="ok">AND</span> mom3 > <b>' + S.t1_mom3 + '%</b>  <span class="ok">AND</span> mom6 > <b>' + S.t1_mom6 + '%</b>\\n' +
    '<span class="ok">AND</span> ' + strength + '  <span class="ok">AND</span> rs_5bar > <b>' + S.t1_rs_min + '</b>' + nifty + '\\n\\n' +
    '<span class="b">Path B — Momentum / Norm buy</span>\\n' +
    'is_norm_buy  <span class="ok">AND</span> norm_score ≥ <b>75</b> (or <b>70</b> if EMERGING)\\n' +
    '<span class="ok">AND</span> ' + strength + '  <span class="ok">AND</span> rs_5bar > <b>' + S.t1_rs_min + '</b>' + nifty;
}}

// ── Pending / apply ──────────────────────────────────────────────
function markPending() {{
  document.getElementById('save-status').textContent = 'Unsaved changes — click Apply to update scanner.';
  document.getElementById('save-status').style.color = '#d29922';
}}

function applySettings() {{
  const json = JSON.stringify(S);
  // Send to Streamlit via query param + reload
  const url = new URL(window.parent.location.href);
  url.searchParams.set('settings_json', json);
  window.parent.location.href = url.toString();
}}

// Init gate preview
updateGatePreview();
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════
#  WATCHLIST
# ══════════════════════════════════════════════════════════════════

def _section_watchlist() -> None:
    supabase_ok = _is_available()
    if "watchlist_loaded" not in st.session_state:
        st.session_state["watchlist"] = load_watchlist() if supabase_ok else []
        st.session_state["watchlist_loaded"] = True

    wl = st.session_state.get("watchlist", [])
    if wl:
        wl_df = pd.DataFrame(wl)[["symbol", "notes"]].rename(
            columns={"symbol": "Symbol", "notes": "Notes"})
        st.dataframe(wl_df, use_container_width=True, hide_index=True, height=150)

    bulk_raw = st.text_area("Symbols (one per line)",
        value="\n".join(w["symbol"] for w in wl),
        height=80, key="bulk_wl", label_visibility="collapsed")

    wl_c1, wl_c2 = st.columns([1, 1])
    with wl_c1:
        if st.button("💾 Save Watchlist", type="primary", key="btn_save_wl"):
            new_syms = [s.strip().upper() for s in bulk_raw.splitlines() if s.strip()]
            if supabase_ok:
                ok = save_watchlist(new_syms)
                if ok:
                    st.session_state["watchlist"] = load_watchlist()
                    st.success(f"✅ Saved {len(new_syms)} symbols.")
                else:
                    st.error("❌ Supabase error.")
            else:
                st.session_state["watchlist"] = [{"symbol": s, "notes": ""} for s in new_syms]
                st.success(f"✅ {len(new_syms)} symbols (session only).")
    with wl_c2:
        if wl:
            rm = st.selectbox("Remove", ["—"] + [w["symbol"] for w in wl],
                key="wl_rm_sel", label_visibility="collapsed")
            if st.button("✕ Remove", key="btn_rm_wl") and rm != "—":
                st.session_state["watchlist"] = [
                    w for w in st.session_state.get("watchlist", []) if w["symbol"] != rm]
                st.rerun()


# ══════════════════════════════════════════════════════════════════
#  SYSTEM
# ══════════════════════════════════════════════════════════════════

def _section_system() -> None:
    if _is_available():
        st.success("✅ Supabase connected.")
    else:
        st.warning(
            "Not configured. Add to `.streamlit/secrets.toml`:\n\n"
            "```toml\nSUPABASE_URL = \"https://xxx.supabase.co\"\n"
            "SUPABASE_KEY = \"your-anon-key\"\n```"
        )
    with st.expander("Database schema SQL", expanded=False):
        st.code(SCHEMA_SQL, language="sql")

    if st.button("Load last 10 scan runs", key="btn_hist"):
        history = load_scan_history(limit=10)
        if not history.empty:
            for ts, grp in history.groupby("run_at"):
                with st.expander(f"🕐 {ts} — {len(grp)} stocks"):
                    st.dataframe(
                        grp[["symbol", "score", "action", "cci", "entry", "sl", "t1"]]
                        .rename(columns=str.title).reset_index(drop=True),
                        use_container_width=True,
                    )
        else:
            st.info("No scan history found.")


# ══════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════

def render() -> dict:
    # Apply any incoming settings from query params
    _sync_params()

    # Build current values dict for the HTML component
    vals = {k: _g(k) for k in DEFAULTS}

    # Render the HTML settings component
    import streamlit.components.v1 as components
    components.html(
        _settings_html(vals),
        height=1600,
        scrolling=False,
    )

    # Watchlist & System remain as Streamlit expanders
    with st.expander("📋 Watchlist", expanded=False):
        _section_watchlist()

    with st.expander("🛠️ System & database", expanded=False):
        _section_system()

    # Return the settings dict (always from session_state after _sync_params)
    ss = st.session_state
    return {k: ss.get(k, DEFAULTS[k]) for k in DEFAULTS if k != "custom_symbols"} | {
        "symbols":        ss.get("symbols", NIFTY500_SYMBOLS),
        "custom_symbols": ss.get("custom_symbols", []),
    }

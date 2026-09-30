"""
utils/settings_defaults.py — single canonical source for every scanner/DORE
settings default (2026-09-29 production-issues pass).

WHY THIS FILE EXISTS: app.py used to build its own `settings` dict with a
second, independently hand-typed set of literal fallbacks
(`ss.get("t1_cci_window", 5)`) instead of reading pages/settings.py's
DEFAULTS (`ss.get("t1_cci_window", DEFAULTS["t1_cci_window"])`). The two
had drifted apart on 15 keys — some materially (nifty_regime_filter: True
here vs False in app.py's old literal; t1_adx_min 23 vs 20; t1_rs_min 0.01
vs 0.0; t2_vol_mult 1.5 vs 1.2). Whichever value ss.get() actually returns
in production depends on whether the user has ever touched that specific
Settings widget in this browser session — before that, every fresh session
silently scored on app.py's forgotten duplicate, not on the value shown on
the Settings page (or in utils/conviction_score_v1.py's own dataclass
defaults, which had a THIRD, independent value for t1_cci_window — 2 —
never reachable in production at all). See
tests/test_settings_defaults_single_source.py, which fails if app.py or
pages/settings.py ever reintroduce a duplicate literal instead of reading
DEFAULTS from here.

DEFAULTS is verbatim what pages/settings.py previously defined in-file
(moved, not changed) — it remains the UI-facing source of truth; this
module only gives it a location both app.py and pages/settings.py can
import without a circular/heavy (streamlit) import.
"""

DEFAULTS = {
    "universe_mode":     "Nifty 500 (default)",
    "custom_symbols":    [],
    "data_source":       "yfinance",
    "workers":           15,
    "hold_days":         20,
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
    # ── EMA periods — Leadership (utils/scoring_core.py build_indicators) ──
    "ema_fast_period":   20,
    "ema_mid_period":    50,
    "ema_slow_period":   200,
    # ── RS vs Sector (utils/sector_map.py) — OFF by default; see 2026-08-06
    # incident note in scanner_engine.py's scan loop for why. Turn on only
    # after confirming RAM/scan-time is acceptable for your deployment.
    "enable_sector_rs":  False,
    # ── Liquidity/tradability floor (utils/scanner_engine.py process()) ──
    # Avg daily turnover in INR crores, trailing 20 bars, below which a
    # Nifty 500 constituent is rejected before scoring. 0 disables. NOT
    # a substitute for a circuit-limit or ASM/GSM surveillance check —
    # neither is implemented anywhere in this pipeline yet.
    "min_avg_turnover_cr": 5.0,
    # ── EMA periods — DORE Engine (utils/dore_settings.py DORESettings) ────
    # Mirrors, at the UI layer, the ema_fast_period/ema_slow_period keys
    # inside st.session_state["dore_settings"] — see the "EMA Periods —
    # DORE Engine" expander in _tab_advanced() for how the two stay synced.
    "dore_ema_fast_period": 9,
    "dore_ema_slow_period": 21,
    # ── EMA periods — DORE Leadership (utils/dore_settings.py DORESettings) ─
    # Independent triad, mirroring the Live Scanner's Leadership EMA block
    # above (ema_fast/mid/slow_period) at DORE's own defaults (9/21/50).
    # Does NOT feed dore_ema_fast_period/dore_ema_slow_period above (DORE's
    # Stage 1/2 trend-engine EMAs) — changing one must never affect the other.
    "dore_leadership_fast_ema": 9,
    "dore_leadership_mid_ema":  21,
    "dore_leadership_slow_ema": 50,
    "t2_comp_bars":      12,
    "t2_atr_ratio":      0.80,
    "t2_vol_mult":       1.5,
    "nifty_regime_filter": True,
    # ── Position Sizing (utils/position_sizing.py) ─────────────────
    # Not fabricated NSE lot sizes — exchange-published lot sizes
    # change quarterly (last Friday of the expiry cycle), so these
    # default to 1 and MUST be set to the current published value
    # before sizing is meaningful. See NSE's F&O lot-size circular.
    "available_capital":  0.0,
    "stock_lot_size":     1,   # single value applied to every F&O stock
                                # in the screener — the app has no
                                # per-stock lot-size table; if that
                                # matters, size individual trades by
                                # hand for now rather than trust this.
    "nifty_lot_size":     1,
    "banknifty_lot_size": 1,
    "sensex_lot_size":    1,
    # [A/B flag — pending 2-4wk backtest before default-on, see decision review 2026-06-17]
    # Adds a Fib golden-zone-pullback pattern rung (independent of CCI recovery state)
    # to the Conviction pattern slot. Targets stocks stuck at pat=8 purely because no
    # other pattern condition fires despite a clean, controlled Fib pullback.
    "ENABLE_GOLDEN_PULLBACK_PATTERN": False,
    # [A/B flag — pending backtest before default-on]
    # Gates Promotion Engine on Decision Engine's structural read: forces
    # Skip on hard_stop/t4_hard_stop, downgrades Actionable→Watch when
    # Lifecycle == EXTENDED or AVOID. See utils/scanner_engine.py recommendation
    # funnel comment block for full rationale. Off by default — run the
    # 1,732-trade backtest with this on vs off before flipping the default.
    "ENABLE_STRUCTURAL_GATE": False,
    # ── Institutional Continuation (VWAP Reclaim) ──────────────────
    "ic_enable_vwap_reclaim":    True,
    "ic_enable_vwap_stoch_conf": True,
    "ic_vwap_touch_atr_mult":    0.25,
    "ic_vwap_touch_lookback":    3,
    "ic_reaction_max_atr":       1.5,
    "ic_confluence_window":      2,
    "ic_require_ema_trend":      True,
    "ic_require_rising_vwap":    True,
    "ic_require_bullish_return": True,
    # [2026-09-30] Confirmed dead at this default: pillar_engine.py gates
    # on `reaction_str < ic_min_reaction_score`, and reaction_str
    # (utils/continuation_patterns.py) is a 0-100 normalised score that
    # can never go negative — so a floor of 0 can never reject anything.
    # Left at 0 rather than picking an arbitrary nonzero value myself,
    # since raising it changes which VWAP-reclaim setups confirm. Pick a
    # real floor (e.g. requires backtesting reaction_str's distribution
    # on confirmed reclaims) before this setting does anything.
    "ic_min_reaction_score":     0,
    "ic_momentum_weight":        15,
    "ic_confluence_weight":      10,
    # ── Backtest engine default (Backtest page reads this to pick its
    # initial Signal Source; still overridable per-run on that page) ──
    "bt_default_engine":         "scanner",
    # ── CV4 tier / signal thresholds (see utils/conviction_score_v1.py
    # V4_THRESHOLD_DEFAULTS) — the Live Scanner's ACTUAL live gate since
    # July 2026. [2026-09-30 fix] These 16 keys were never mirrored here
    # despite the v3 block below being carried over verbatim during the
    # "single source of truth" settings consolidation — meaning the real
    # live floors had no Settings-page control and no default anyone
    # could see outside conviction_score_v1.py itself. Keep in sync
    # manually whenever V4_THRESHOLD_DEFAULTS changes; this dict's
    # values win whenever a key is present (see _g()/_s() in
    # pages/settings.py).
    "v4_watch_leadership_min":         50,
    "v4_watch_conviction_min":         50,
    "v4_watch_entry_quality_min":      50,
    "v4_actionable_leadership_min":    70,
    "v4_actionable_conviction_min":    60,
    "v4_actionable_entry_quality_min": 60,
    "v4_actionable_composite_min":     66,
    "v4_execute_leadership_min":       80,
    "v4_execute_conviction_min":       70,
    "v4_execute_entry_quality_min":    70,
    "v4_execute_composite_min":        76,
    "v4_elite_leadership_min":         85,
    "v4_elite_conviction_min":         75,
    "v4_elite_entry_quality_min":      80,
    "v4_elite_composite_min":          82,
    # ── LEGACY — CV1 v3 tier / signal thresholds (see utils/conviction_score_v1.py
    # V3_THRESHOLD_DEFAULTS — decile-backtest calibrated, 2026-07). The Live
    # Scanner has run on CV4 (v4_* above) since July 2026; these v3_* keys
    # now only feed the backtest engine's legacy v3 comparison path, NOT
    # live scoring. Kept for that path rather than removed — do not treat
    # as tunable knobs for the live scanner. This dict is the UI-side
    # mirror of that module's defaults; keep them in sync manually
    # whenever conviction_score_v1.py's defaults change, since this
    # dict's values win whenever a key is present (which is always, for
    # keys not yet touched by the user — see _g()/_s() below).
    "v3_watch_leadership_min":      50,
    "v3_watch_conviction_min":      50,
    "v3_watch_entry_quality_min":   50,
    "v3_watch_composite_min":       50,
    # ── Developing — previously composite-only; now has its own
    # Leadership/Conviction/Entry Quality floors (2026-07 revision), same
    # AND-gated pattern as Actionable/Execute/Elite. See classify_tier_v3()
    # in utils/conviction_score_v1.py.
    "v3_developing_leadership_min":    70,
    "v3_developing_conviction_min":    55,
    "v3_developing_entry_quality_min": 80,
    "v3_developing_composite_min":     55,
    "v3_actionable_leadership_min":    70,
    "v3_actionable_conviction_min":    60,
    # Was 36 (module-default only, not previously mirrored here — this
    # key was missing from Settings entirely, so the UI never actually
    # controlled it). Now 80 per 2026-07 revision.
    "v3_actionable_entry_quality_min": 80,
    "v3_actionable_composite_min":     60,
    "v3_execute_leadership_min":       80,
    "v3_execute_conviction_min":       70,
    "v3_execute_entry_quality_min":    80,
    "v3_execute_composite_min":        60,
    "v3_elite_leadership_min":         85,
    "v3_elite_conviction_min":         75,
    "v3_elite_entry_quality_min":      85,
    "v3_elite_composite_min":          66,
    # ── Promotion Engine (utils/promotion_engine.py) — Promo Score and
    # R:R thresholds are all plain overrides, freely adjustable either
    # direction (see evaluate_promotion docstring) ──
    "promo_execute_score_min": 50,
    "promo_elite_score_min":   75,
    "promo_min_rr_elite":      2.0,
    # ── Backtest admission (utils/backtest_engine.py Gate 3) — separate
    # from Promotion Engine's R:R gates; a backtest population choice,
    # freely adjustable either direction ──
    "backtest_min_rr": 2.0,
}

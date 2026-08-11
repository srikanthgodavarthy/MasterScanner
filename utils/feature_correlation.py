"""
utils/feature_correlation.py
────────────────────────────
P1 "Feature Correlation Analysis" + "Momentum Diagnostic" from the
2026-08-10 DORE + Live Scanner Diagnostic & Outcome-Tracking audit.

Diagnostic-only, by the audit's explicit instruction ("Do not automatically
modify weights based on correlation. This is diagnostic only."). This
module never writes to DORESettings/cfg and is never imported by
utils.dore_engine's scoring path — it's a standalone analysis tool over
ALREADY-PERSISTED utils.entry_snapshot / utils.outcome_tracking rows,
meant to be run from a notebook, a diagnostics page, or an ad-hoc script
once enough closed trades exist (the audit is explicit that weights/
thresholds should not change until then either).

Two questions this answers (see DORE_LIVE_SCANNER_AUDIT.md "Final
Principle"):
  1. "Are RS Momentum, EMA slope and Premium Behaviour double-counting
     the same information?" -> feature_correlation_matrix()
  2. "Which DORE pillar actually predicts MFE/MAE?" -> pillar_outcome_correlation()
"""

from __future__ import annotations

import logging
from typing import Optional

from utils import db

logger = logging.getLogger(__name__)

# The audit's named feature list — every one of these is a plain column
# on dore_entry_snapshots (utils.entry_snapshot) or, for the underlying-
# side ones, on live_scanner_entry_snapshots.
DORE_FEATURE_COLUMNS: tuple[str, ...] = (
    "trend_score", "execution_score", "derivatives_score",
    "option_intelligence_score", "premium_behavior",
)

LIVE_SCANNER_FEATURE_COLUMNS: tuple[str, ...] = (
    "rs_momentum", "ema_slope", "adx", "volume_ratio",
    "leadership_score", "conviction_score", "entry_quality_score",
)

# [Fix, this review round] source -> (table, id_col). "DORE" here means
# utils.dore_options_engine (the OptionTradePlan-based pipeline); its
# entry_snapshot columns are ONLY approximate stand-ins for the audit's
# named pillars (see utils.entry_snapshot.build_dore_entry_snapshot's
# docstring). "DORE_STAGE5" is the RFC-001 Stage 1-5 engine
# (utils.dore_engine / utils.fo_scan) whose trend_score/execution_score/
# derivatives_score/option_intelligence_score ARE the audit's literal
# field names — that's the one "which DORE pillar actually predicts
# MFE/MAE" (this module's whole purpose) should be run against.
_SOURCE_TABLES: dict[str, tuple[str, str]] = {
    "DORE": ("dore_entry_snapshots", "plan_id"),
    "DORE_STAGE5": ("dore_stage5_entry_snapshots", "setup_id"),
    "LIVE_SCANNER": ("live_scanner_entry_snapshots", "setup_id"),
}


def _load_dataframe(query: str, params: tuple = ()):
    """Returns a pandas DataFrame, or None if Neon isn't configured /
    the query fails / pandas isn't importable. Never raises."""
    if not db.is_available():
        logger.warning("[feature_correlation] Neon not configured — nothing to analyze")
        return None
    try:
        import pandas as pd
        rows = db.fetch_all(query, params)
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:
        logger.exception("[feature_correlation] query failed")
        return None


def feature_correlation_matrix(source: str = "DORE_STAGE5", min_rows: int = 30):
    """Pearson correlation matrix among the named feature columns for
    `source` ("DORE_STAGE5", "DORE", or "LIVE_SCANNER" — see
    _SOURCE_TABLES), computed over every entry snapshot on file —
    independent of outcome, this just answers "are these features
    redundant with each other". Returns a pandas DataFrame (features x
    features), or None if there isn't enough data yet (fewer than
    `min_rows` snapshots) or Neon isn't configured.

    Deliberately does NOT flag/print anything about DORESettings weights
    — that judgment call belongs to a human reading the output, per the
    audit's "diagnostic only" instruction.
    """
    if source not in _SOURCE_TABLES:
        logger.warning("[feature_correlation] unknown source=%r — expected one of %s",
                        source, list(_SOURCE_TABLES))
        return None
    table, _id_col = _SOURCE_TABLES[source]
    cols = LIVE_SCANNER_FEATURE_COLUMNS if source == "LIVE_SCANNER" else DORE_FEATURE_COLUMNS
    df = _load_dataframe(f"SELECT * FROM {table}")
    if df is None or len(df) < min_rows:
        logger.info("[feature_correlation] %s has %s row(s) — need >= %d for a stable correlation read",
                    table, 0 if df is None else len(df), min_rows)
        return None
    present = [c for c in cols if c in df.columns]
    return df[present].apply(lambda s: s.astype(float, errors="ignore")).corr()


def flag_highly_correlated_pairs(corr_matrix, threshold: float = 0.75) -> list[dict]:
    """corr_matrix -> [{"feature_a": ..., "feature_b": ..., "correlation": ...}, ...]
    for every pair whose |correlation| >= threshold. Report only — the
    audit's own output shape ("feature A / feature B / correlation").
    Never mutates anything; the caller decides what (if anything) to do
    with the flagged pairs."""
    if corr_matrix is None:
        return []
    out = []
    cols = list(corr_matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                r = float(corr_matrix.loc[a, b])
            except Exception:
                continue
            if r == r and abs(r) >= threshold:   # r == r excludes NaN
                out.append({"feature_a": a, "feature_b": b, "correlation": round(r, 3)})
    return sorted(out, key=lambda d: -abs(d["correlation"]))


def pillar_outcome_correlation(source: str = "DORE_STAGE5", checkpoint_minutes: int = 60, min_rows: int = 30):
    """"Which DORE pillar actually predicts MFE/MAE?" — correlates each
    entry-snapshot pillar score against that same plan's outcome
    checkpoint (mfe_pct/mae_pct) at `checkpoint_minutes`. Defaults to
    "DORE_STAGE5" — the RFC-001 Stage 1-5 engine — since that's the
    pipeline whose trend_score/execution_score/derivatives_score/
    option_intelligence_score are the audit's actual named pillars (see
    _SOURCE_TABLES's comment). Returns a pandas DataFrame indexed by
    pillar, columns ["corr_with_mfe", "corr_with_mae", "n"], or None if
    there isn't enough closed/checkpointed data yet. Joins
    utils.entry_snapshot to utils.outcome_tracking's outcome_checkpoints
    on (plan_key/setup_id, source) — plans with no checkpoint at this
    interval yet are simply excluded, not treated as zero."""
    if not db.is_available():
        return None
    if source not in _SOURCE_TABLES:
        logger.warning("[feature_correlation] unknown source=%r — expected one of %s",
                        source, list(_SOURCE_TABLES))
        return None
    snap_table, id_col = _SOURCE_TABLES[source]
    cols = LIVE_SCANNER_FEATURE_COLUMNS if source == "LIVE_SCANNER" else DORE_FEATURE_COLUMNS

    query = f"""
        SELECT s.*, c.mfe_pct AS outcome_mfe_pct, c.mae_pct AS outcome_mae_pct
        FROM {snap_table} s
        JOIN outcome_checkpoints c
          ON c.plan_key = s.{id_col} AND c.source = %s AND c.interval_minutes = %s
    """
    df = _load_dataframe(query, (source, checkpoint_minutes))
    if df is None or len(df) < min_rows:
        logger.info("[feature_correlation] %s joined-with-outcomes has %s row(s) — need >= %d",
                    snap_table, 0 if df is None else len(df), min_rows)
        return None

    import pandas as pd
    present = [c for c in cols if c in df.columns]
    out_rows = []
    for pillar in present:
        series = df[pillar].astype(float, errors="ignore")
        n = int(series.notna().sum())
        out_rows.append({
            "pillar": pillar,
            "corr_with_mfe": round(float(series.corr(df["outcome_mfe_pct"])), 3) if n >= 3 else None,
            "corr_with_mae": round(float(series.corr(df["outcome_mae_pct"])), 3) if n >= 3 else None,
            "n": n,
        })
    return pd.DataFrame(out_rows).set_index("pillar")

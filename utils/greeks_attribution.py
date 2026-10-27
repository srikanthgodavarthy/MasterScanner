"""
utils/greeks_attribution.py — Greeks Phase 1 measurement harness
(2026-10-23, SG request).

PURPOSE
  Decide, from data, which Greek-derived diagnostics are worth letting
  influence a DORE decision -- instead of adding all of them to a weighted
  score on faith. OBSERVATION ONLY: nothing here is imported by the live
  pipeline, and it writes nothing to the database.

WHERE THE DATA COMES FROM
  * Greeks at mint: OptionTradePlan.greeks is part of OptionTradePlan.
    to_dict(), which the scan already stores inside
    dore_technical_plans_snapshots.payload->'technical_plans' (JSONB, no
    schema change). Only plans minted AFTER the Greeks Phase 1 deploy carry
    it -- older closed plans have nothing to join, so this reports
    "insufficient" until enough post-deploy plans have closed.
  * Outcomes: dore_options_plans rows with status CLOSED (mfe/mae, T1 hit,
    close reason).

  A closed plan is matched to the snapshot row for the same (symbol,
  direction, strike, expiry) whose timestamp is nearest the plan's
  created_at, within MATCH_TOLERANCE -- the plan is minted from that
  cycle's plan dict, and the snapshot is written at the end of the same
  cycle, so it can land slightly before or after created_at.

WHAT IT REPORTS, per diagnostic
  n, Spearman vs MFE% and vs MAE%, partial Spearman vs MFE% controlling
  for confidence_score (does it add anything the existing score doesn't
  already say?), and stop-loss / T1-hit rates by tercile. Below MIN_N
  matched plans it labels everything INSUFFICIENT and draws no conclusion
  -- 44 historical closes were already too few to separate noise from
  signal, and a handful of post-deploy closes are fewer still.

USAGE
    python -m utils.greeks_attribution                 # all matchable history
    python -m utils.greeks_attribution --since 2026-10-24 --min-n 30
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, Optional

import pandas as pd

MIN_N = 30
MATCH_TOLERANCE = pd.Timedelta(minutes=45)

# Greek-derived diagnostics under test. `theta_pct_day`..`delta_vs_proxy` are
# flattened out of OptionTradePlan.greeks; abs_delta is derived.
FEATURES = (
    "abs_delta",        # Delta   -- how directional the contract is
    "blast_score",      # Gamma blast detector score (0-100; utils/dore_gamma_blast.py)
    "gamma_share",      # Gamma   -- share of the favorable estimate that is convexity
    "gamma_pct",        # Gamma   -- convexity gain in the favorable scenario, % of premium
    "theta_pct_day",    # Theta   -- daily decay as % of premium
    "be_sigma",         # Theta+Delta+Gamma -- sigmas of move needed just to pay for decay
    "vega_pct_pt",      # Vega    -- premium sensitivity to 1 IV point, % of premium
    "iv",               # IV level
    "fav_pct",          # Scenario -- estimated premium change on a 1-sigma favorable move
    "adv_pct",          # Scenario -- estimated premium change on a 1-sigma adverse move + IV drop
    "t1_sigma_req",     # Scenario -- sigmas of move needed to deliver T1
    "t2_sigma_req",
    "delta_vs_proxy",   # Delta   -- |real delta| minus DORE's POP-as-delta proxy
)


# ── data shaping (pure; no DB) ──────────────────────────────────────────
def _is_set(v) -> bool:
    """True when a nullable timestamp/text column actually holds a value.

    `bool(v)` is NOT safe here: once any row in a column holds a datetime, pandas
    stores the column as datetime64 and a missing value becomes NaT, and
    bool(pd.NaT) / bool(float("nan")) are both True. Treat None, NaN, NaT and an
    empty/blank string all as "not set"."""
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):      # array-likes etc.: fall through to the text check
        pass
    return str(v).strip() != ""


def greeks_rows_from_plans(snap_ts, plans: Iterable[dict]) -> list[dict]:
    """Flatten one snapshot's technical_plans into one row per plan that
    carries a usable Greeks read. `snap_ts` is the snapshot timestamp."""
    rows = []
    for p in plans or []:
        g = p.get("greeks")
        if not isinstance(g, dict) or not g.get("ok"):
            continue
        prim = p.get("primary") or {}
        row = {
            "snap_ts": pd.Timestamp(snap_ts),
            "symbol": p.get("symbol"), "direction": p.get("direction"),
            "strike": prim.get("strike"), "expiry": str(p.get("expiry") or ""),
            "confidence": p.get("confidence_score"), "dte": p.get("dte"),
        }
        row.update({f"g_{k}": v for k, v in g.items()})
        _gb = p.get("gamma_blast")          # utils/dore_gamma_blast.py read, when present
        if isinstance(_gb, dict):
            row["g_blast_score"] = _gb.get("score")
            row["g_blast_level"] = _gb.get("level")
        rows.append(row)
    return rows


def match_closed_plans(greeks_df: pd.DataFrame, plans_df: pd.DataFrame,
                       tolerance: pd.Timedelta = MATCH_TOLERANCE) -> pd.DataFrame:
    """One row per CLOSED plan that has a Greeks snapshot within tolerance,
    with outcome columns and the g_* diagnostics attached."""
    if greeks_df.empty or plans_df.empty:
        return pd.DataFrame()
    g = greeks_df.copy()
    g["strike"] = pd.to_numeric(g["strike"], errors="coerce")
    out = []
    for _, p in plans_df.iterrows():
        try:
            strike = float(p["strike"])
            created = pd.Timestamp(p["created_at"])
        except Exception:
            continue
        cand = g[(g["symbol"] == p["symbol"]) & (g["direction"] == p["direction"])
                 & ((g["strike"] - strike).abs() < 1e-6) & (g["expiry"] == str(p["expiry"])[:10])]
        if cand.empty:
            continue
        # tz-align defensively
        ts = cand["snap_ts"]
        if created.tzinfo is None and ts.dt.tz is not None:
            created = created.tz_localize(ts.dt.tz)
        elif created.tzinfo is not None and ts.dt.tz is None:
            ts = ts.dt.tz_localize(created.tzinfo)
        gap = (ts - created).abs()
        best = gap.idxmin()
        if gap.loc[best] > tolerance:
            continue
        row = cand.loc[best].to_dict()
        row.update({
            "plan_id": p.get("plan_id"),
            "entry_locked": p.get("entry_locked"),
            "mfe_premium": p.get("mfe_premium"), "mae_premium": p.get("mae_premium"),
            "target1_locked": p.get("target1_locked"), "target2_locked": p.get("target2_locked"),
            "last_premium": p.get("last_premium"),
            "closed_reason_code": p.get("closed_reason_code") or "",
            "closed_reason": p.get("closed_reason") or "",
            "t1_hit": _is_set(p.get("t1_hit_at")),
        })
        out.append(row)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    ent = pd.to_numeric(df["entry_locked"], errors="coerce")
    df = df[ent.notna() & (ent > 0)].copy()          # plans that never entered have no outcome
    if df.empty:
        return df
    ent = pd.to_numeric(df["entry_locked"], errors="coerce")
    # Realized exit: target closes exit at the target, everything else at the
    # last observed premium (an approximation -- fills aren't stored).
    code = df["closed_reason_code"].astype(str)
    exit_p = pd.to_numeric(df["last_premium"], errors="coerce")
    exit_p = exit_p.where(code != "TARGET_2", pd.to_numeric(df["target2_locked"], errors="coerce"))
    exit_p = exit_p.where(code != "TARGET_1", pd.to_numeric(df["target1_locked"], errors="coerce"))
    df["realized_pct"] = (exit_p / ent - 1.0) * 100.0
    for src, dst in (("mfe_premium", "mfe_pct"), ("mae_premium", "mae_pct")):
        df[dst] = (pd.to_numeric(df[src], errors="coerce") / ent - 1.0) * 100.0
    code = df["closed_reason_code"].astype(str)
    reason = df["closed_reason"].astype(str).str.lower()
    df["hit_sl"] = (code == "STOP_LOSS") | reason.str.contains("stop-loss")
    df["abs_delta"] = pd.to_numeric(df.get("g_delta"), errors="coerce").abs()
    for f in FEATURES:
        if f != "abs_delta" and f"g_{f}" in df.columns:
            df[f] = pd.to_numeric(df[f"g_{f}"], errors="coerce")
    return df


# ── statistics (pure; pandas only, no scipy) ────────────────────────────
def _spearman(a: pd.Series, b: pd.Series) -> Optional[float]:
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return None
    ra, rb = a[m].rank(), b[m].rank()
    if ra.nunique() < 2 or rb.nunique() < 2:
        return None
    r = ra.corr(rb)
    return None if r is None or (isinstance(r, float) and math.isnan(r)) else float(r)


def _partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> Optional[float]:
    """Spearman of x vs y controlling for z (first-order partial)."""
    rxy, rxz, ryz = _spearman(x, y), _spearman(x, z), _spearman(y, z)
    if None in (rxy, rxz, ryz):
        return None
    den = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return float((rxy - rxz * ryz) / den)


def attribute(df: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    """Per-diagnostic screen. Returns one row per feature present in df."""
    rows = []
    if df is None or df.empty:
        return pd.DataFrame(rows)
    conf = pd.to_numeric(df.get("confidence"), errors="coerce")
    for f in FEATURES:
        if f not in df.columns:
            continue
        x = pd.to_numeric(df[f], errors="coerce")
        n = int(x.notna().sum())
        row = {"feature": f, "n": n, "status": "ok" if n >= min_n else f"INSUFFICIENT (n<{min_n})",
               "rho_mfe": _spearman(x, df["mfe_pct"]), "rho_mae": _spearman(x, df["mae_pct"]),
               "partial_mfe_ctrl_conf": _partial_spearman(x, df["mfe_pct"], conf)}
        try:
            tier = pd.qcut(x, 3, labels=["low", "mid", "high"], duplicates="drop")
            for name, col in (("sl_rate", "hit_sl"), ("t1_rate", "t1_hit")):
                by = df.groupby(tier, observed=True)[col].mean()
                for lvl in ("low", "mid", "high"):
                    row[f"{name}_{lvl}"] = float(by[lvl]) if lvl in by.index else None
        except Exception:
            pass
        rows.append(row)
    return pd.DataFrame(rows)


def rule_sweep(df: pd.DataFrame, feature: str, thresholds: Iterable[float],
               min_n: int = MIN_N, min_excluded: int = 5) -> pd.DataFrame:
    """Counterfactual for a veto rule `feature > threshold`: what would it
    have excluded, and was that a good trade to skip? `net_pct_gain` is the
    summed realized % of the excluded plans with the sign flipped, so
    positive = the rule would have helped (it skipped net losers). Reports
    winners_forgone so a rule that mostly skips winners is visible. A row is
    only `ok` with >= min_n matched plans AND >= min_excluded exclusions."""
    cols = ["threshold", "n_total", "n_excluded", "winners_forgone", "mean_excluded_pct",
            "mean_kept_pct", "net_pct_gain", "status"]
    if df is None or df.empty or feature not in df.columns or "realized_pct" not in df.columns:
        return pd.DataFrame(columns=cols)
    x = pd.to_numeric(df[feature], errors="coerce")
    r = pd.to_numeric(df["realized_pct"], errors="coerce")
    rows = []
    for t in thresholds:
        ex = (x > t) & r.notna()
        kept = (~(x > t)) & r.notna()
        n_ex = int(ex.sum())
        ok = len(df) >= min_n and n_ex >= min_excluded
        rows.append({
            "threshold": t, "n_total": int(r.notna().sum()), "n_excluded": n_ex,
            "winners_forgone": int((r[ex] > 0).sum()),
            "mean_excluded_pct": float(r[ex].mean()) if n_ex else None,
            "mean_kept_pct": float(r[kept].mean()) if kept.sum() else None,
            "net_pct_gain": float(-r[ex].sum()) if n_ex else 0.0,
            "status": "ok" if ok else f"INSUFFICIENT (need n>={min_n}, excluded>={min_excluded})",
        })
    return pd.DataFrame(rows, columns=cols)


def estimator_check(df: pd.DataFrame, min_n: int = MIN_N) -> dict:
    """Does the scenario estimator's favorable read line up with what trades
    actually did, and does the `reliable` flag matter? MFE is the best
    premium reached, not the 1-sigma/2-day scenario, so this is a sanity
    comparison, not a calibration."""
    if df is None or df.empty or "fav_pct" not in df.columns:
        return {"n": 0, "status": f"INSUFFICIENT (n<{min_n})"}
    out = {"n": int(len(df)), "status": "ok" if len(df) >= min_n else f"INSUFFICIENT (n<{min_n})"}
    rel = df.get("g_reliable")
    for label, mask in (("reliable", rel == True), ("unreliable", rel == False)):   # noqa: E712
        sub = df[mask] if rel is not None else df.iloc[0:0]
        out[label] = {"n": int(len(sub)),
                      "mean_fav_pct": float(sub["fav_pct"].mean()) if len(sub) else None,
                      "mean_mfe_pct": float(sub["mfe_pct"].mean()) if len(sub) else None,
                      "rho_fav_vs_mfe": _spearman(sub["fav_pct"], sub["mfe_pct"]) if len(sub) else None}
    if "t1_sigma_req" in df.columns:
        out["rho_t1_sigma_req_vs_t1_hit"] = _spearman(df["t1_sigma_req"], df["t1_hit"].astype(float))
    return out


def format_report(df: pd.DataFrame, min_n: int = MIN_N) -> str:
    lines = [f"Greeks attribution — {len(df)} closed plan(s) matched to a Greeks snapshot (MIN_N={min_n})"]
    if df.empty:
        lines.append("  No matches yet: only plans minted after the Greeks Phase 1 deploy carry Greeks.")
        return "\n".join(lines)
    lines.append(f"  outcomes: SL {df['hit_sl'].mean():.0%} | T1 touched {df['t1_hit'].mean():.0%} | "
                 f"median MFE {df['mfe_pct'].median():.1f}% | median MAE {df['mae_pct'].median():.1f}%")
    table = attribute(df, min_n)
    if not table.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
            lines.append(table.to_string(index=False))
    lines.append("estimator: " + str(estimator_check(df, min_n)))
    if "realized_pct" in df.columns:
        lines.append(f"  realized (approx): mean {df['realized_pct'].mean():.1f}% | "
                     f"win rate {(df['realized_pct'] > 0).mean():.0%}")
        # The two thresholds the (default-off) Greeks gate would use.
        for feat, ths in (("be_sigma", (0.5, 0.75, 1.0, 1.5)), ("t1_sigma_req", (1.0, 1.5, 2.0, 3.0))):
            sw = rule_sweep(df, feat, ths, min_n)
            if not sw.empty:
                with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
                    lines.append(f"gate sweep: {feat} > threshold\n" + sw.to_string(index=False))
    if len(df) < min_n:
        lines.append(f"  n={len(df)} < {min_n}: descriptive only — do not wire any diagnostic into a score or gate yet.")
    return "\n".join(lines)


# ── DB loading (CLI only; isolated so the pure functions stay testable) ──
# Plans first, then only the snapshot rows that can possibly match them
# (same symbols, a window around the closed plans' creation times), and
# only the handful of fields needed -- not whole payloads -- to keep Neon
# egress small (see the 2026-08-18 egress note in utils.supabase_client).
_PLANS_SQL = "SELECT * FROM dore_options_plans WHERE status = 'CLOSED' AND created_at >= %s"
_SNAP_SQL = """
SELECT s.created_at AS snap_ts,
       e->>'symbol' AS symbol, e->>'direction' AS direction,
       e->'primary'->>'strike' AS strike, e->>'expiry' AS expiry,
       e->>'confidence_score' AS confidence, e->>'dte' AS dte,
       e->'greeks' AS greeks
FROM dore_technical_plans_snapshots s
CROSS JOIN LATERAL jsonb_array_elements(
        CASE WHEN jsonb_typeof(s.payload->'technical_plans') = 'array'
             THEN s.payload->'technical_plans' ELSE '[]'::jsonb END) AS e
WHERE s.created_at BETWEEN %s AND %s
  AND e->>'symbol' = ANY(%s)
  AND jsonb_typeof(e->'greeks') = 'object'
ORDER BY s.created_at
"""


def load_and_report(since: str = "2026-10-01", min_n: int = MIN_N) -> str:
    from utils import db   # local import: keeps the pure functions free of psycopg2
    plans = db.fetch_all(_PLANS_SQL, (since,))
    if not plans:
        return format_report(pd.DataFrame(), min_n)
    created = pd.to_datetime([p["created_at"] for p in plans], utc=True)
    lo, hi = created.min() - MATCH_TOLERANCE, created.max() + MATCH_TOLERANCE
    symbols = sorted({p["symbol"] for p in plans})
    snaps = db.fetch_all(_SNAP_SQL, (lo.to_pydatetime(), hi.to_pydatetime(), symbols))
    rows: list[dict] = []
    for r in snaps:
        rows.extend(greeks_rows_from_plans(r["snap_ts"], [{
            "symbol": r["symbol"], "direction": r["direction"], "primary": {"strike": r["strike"]},
            "expiry": r["expiry"], "confidence_score": r["confidence"], "dte": r["dte"], "greeks": r["greeks"],
        }]))
    matched = match_closed_plans(pd.DataFrame(rows), pd.DataFrame(plans))
    return format_report(matched, min_n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default="2026-10-01")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    a = ap.parse_args()
    print(load_and_report(a.since, a.min_n))

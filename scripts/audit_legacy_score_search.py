"""
Evolutionary search for the highest legacy raw score compute_bar() can actually produce.

Read-only audit tool: it builds synthetic OHLCV, calls the real compute_bar through
tests/legacy_score_audit_support.py, and records what it finds.  It never edits
scoring code, weights or thresholds.

A search can only ever give a LOWER bound on the attainable maximum (it may miss a
region).  The algebraic ceiling is the UPPER bound.  Report both.

usage: python scripts/audit_legacy_score_search.py OUT.json [restarts] [generations] [pop] [force_flag ...]
  force_flag: a top-tier flag name from top_tier_flags(); adds a large bonus when set,
              so the search reports the best total score reachable WITH that term at max.
"""
import json, sys, time
sys.path.insert(0, ".")
import numpy as np
from tests.legacy_score_audit_support import (
    bounds, noise, evaluate, live_params, top_tier_flags, recompute_raw, unconstrained_ceiling)

def main():
    out = sys.argv[1]
    restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    gens = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    pop = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    forced = sys.argv[5:]
    p = live_params(); lo, hi = bounds(); span = hi - lo
    results = []
    t0 = time.time()
    for rs in range(restarts):
        rng = np.random.default_rng(1000 + rs)
        z = noise(320, 5000 + rs)

        def fit(x):
            s, cap, res = evaluate(x, z, p)
            if s is None:
                return -1e9, None, None
            f = s
            if forced:
                flags = top_tier_flags(cap, p)
                f += sum(300 for k in forced if flags.get(k)) - sum(300 for k in forced if not flags.get(k))
            return f, s, cap

        P = [lo + span * rng.random(len(lo)) for _ in range(pop * 4)]
        scored = []
        for x in P:
            f, s, cap = fit(x); scored.append((f, s, x, cap))
        scored.sort(key=lambda t: -t[0]); scored = scored[:pop]
        sigma = 0.15
        for g in range(gens):
            kids = []
            for _ in range(pop * 2):
                a = scored[int(rng.integers(0, max(3, pop // 3)))][2]
                b = scored[int(rng.integers(0, max(3, pop // 3)))][2]
                m = rng.random(len(lo)) < 0.5
                child = np.where(m, a, b)
                mut = rng.random(len(lo)) < 0.25
                child = child + mut * rng.standard_normal(len(lo)) * sigma * span
                child = np.clip(child, lo, hi)
                f, s, cap = fit(child); kids.append((f, s, child, cap))
            scored = sorted(scored + kids, key=lambda t: -t[0])[:pop]
            sigma = max(0.02, sigma * 0.985)
        f, s, x, cap = scored[0]
        flags = top_tier_flags(cap, p)
        results.append({
            "restart": rs, "noise_seed": 5000 + rs, "x": [float(v) for v in x], "raw_score": s, "fitness": f, "norm_pre_bonus": min(100, int(s * 100 / p.max_score)),
            "recompute_matches": abs(recompute_raw(cap, p) - s) < 1e-9,
            "trend_phase": cap["trend_phase"], "top_tier": flags,
            "missing_top_tier": [k for k, v in flags.items() if not v],
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(rs, "raw", s, "norm", results[-1]["norm_pre_bonus"], "phase", cap["trend_phase"],
              "missing", results[-1]["missing_top_tier"], flush=True)
    json.dump({"ceiling_live": unconstrained_ceiling(p), "forced": forced, "results": results}, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()

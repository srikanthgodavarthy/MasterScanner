"""
Pairwise co-occurrence of TOP-TIER legacy-score terms, sampled by a diversity-seeking search.

Read-only audit tool (never edits scoring).  Fitness = raw score + 15 x (# distinct top-tier
terms), so the sampler keeps visiting states that stack many terms.  Every evaluated
individual's top-tier flags are tallied.  A pair with count 0 while both members were seen
individually is a CANDIDATE exclusion (absence of evidence, not proof).

usage: python scripts/audit_legacy_score_cooccurrence.py OUT.json [restarts] [gens] [pop]
"""
import json, sys, itertools, time
sys.path.insert(0, ".")
import numpy as np
from tests.legacy_score_audit_support import bounds, noise, evaluate, live_params, top_tier_flags

def main():
    out = sys.argv[1]; restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    gens = int(sys.argv[3]) if len(sys.argv) > 3 else 40; pop = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    p = live_params(); lo, hi = bounds(); span = hi - lo
    names = None; single = {}; pair = {}; n_eval = 0; best = 0.0
    t0 = time.time()
    for rs in range(restarts):
        rng = np.random.default_rng(31000 + rs); z = noise(320, 9000 + rs)
        def fit(x):
            nonlocal names, n_eval, best
            s, cap, _ = evaluate(x, z, p)
            if s is None: return -1e9
            n_eval += 1; best = max(best, s)
            fl = top_tier_flags(cap, p)
            if names is None: names = sorted(fl)
            on = [k for k, v in fl.items() if v]
            for k in on: single[k] = single.get(k, 0) + 1
            for a, b in itertools.combinations(sorted(on), 2): pair[(a, b)] = pair.get((a, b), 0) + 1
            return s + 15 * len(on)
        P = sorted(((fit(x), x) for x in (lo + span * rng.random(len(lo)) for _ in range(pop * 4))), key=lambda t: -t[0])[:pop]
        sigma = 0.15
        for g in range(gens):
            kids = []
            for _ in range(pop * 2):
                a = P[int(rng.integers(0, max(3, pop // 3)))][1]; b = P[int(rng.integers(0, max(3, pop // 3)))][1]
                c = np.where(rng.random(len(lo)) < 0.5, a, b)
                c = np.clip(c + (rng.random(len(lo)) < 0.25) * rng.standard_normal(len(lo)) * sigma * span, lo, hi)
                kids.append((fit(c), c))
            P = sorted(P + kids, key=lambda t: -t[0])[:pop]; sigma = max(0.02, sigma * 0.97)
        print("restart", rs, "evals", n_eval, "elapsed %.0fs" % (time.time() - t0), flush=True)
    json.dump({"evaluations": n_eval, "best_raw_seen": best, "single": single,
               "pair": {f"{a}|{b}": c for (a, b), c in pair.items()}, "names": names}, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()

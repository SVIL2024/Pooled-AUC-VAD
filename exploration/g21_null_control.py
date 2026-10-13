"""G21: is the per-pair "gain" distinguishable from picking a class at random?

The control G15 never got
-------------------------
G15 sampled 40 random bias VECTORS on UCF->XD and found mean AP 0.6713 versus
a baseline of 0.6717, with p90 0.6992 and max 0.7068.  So on that pair random
perturbation does NOT help on average while the oracle clearly does -- the
+0.035 there is real signal.

XD->UCF was never given that control, and its numbers look much weaker: the
best single class gains +0.0174 (Robbery), but every class that actually has a
counterpart in the source table gains at most +0.0063, and the winners are
classes with no XD counterpart at all.  Taking the max over 13 classes selects
an upward-biased maximum, so the question "would a random class have done as
well?" decides whether XD->UCF has any effect at all.

What is measured, per pair
--------------------------
  1. N random SINGLE-CLASS boosts, one class drawn uniformly, lambda=2.0.
     Compare the oracle pick against this null distribution -- the honest
     comparison is oracle vs the null's upper tail, not vs the baseline.
  2. N random full BIAS VECTORS, to close the loop on G15 for XD->UCF.
  3. The percentile of the oracle pick inside the null.

Interpretation
--------------
  * oracle far into the null tail (p < 0.05) -> that pair has a real effect.
  * oracle inside the bulk of the null        -> it was selection on noise and
    XD->UCF has no per-class effect at all, which changes the whole claim from
    "direction-dependent gain" to "gain exists in one direction only".
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score

EPS = 1e-12
XD_CLASSES = ["fighting", "shooting", "riot", "abuse", "car accident",
              "explosion"]
UCF_CLASSES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
               "Fighting", "RoadAccidents", "Robbery", "Shooting",
               "Shoplifting", "Stealing", "Vandalism"]


def check_identity(path, want_cols, want_tokens, tag):
    z = np.load(path, allow_pickle=True)
    ncol = z["cls"].shape[1]
    blob = " ".join(str(x) for x in
                    np.unique(np.asarray(z["names"], dtype=object)))
    if ncol != want_cols or (blob and not any(t in blob
                                              for t in want_tokens)):
        raise SystemExit(f"[GUARD] {tag}: {os.path.basename(path)} cols={ncol}")
    print(f"[guard ok] {tag}: {os.path.basename(path)} cols={ncol}")


def load(path):
    z = np.load(path, allow_pickle=True)
    cls, gt = z["cls"], z["gt"]
    logP = np.log(np.clip(cls, EPS, 1.0))
    logit = logP[:, 1:] - logP[:, [0]]
    return logit, (gt[:len(cls)] > 0.5).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def find(a, b):
        p = glob.glob(os.path.join(args.scores, f"classprobs_{a}2{b}.npz"))
        return p[0] if p else None

    UCF_TOK, XD_TOK = UCF_CLASSES + ["Normal"], ["B1", "B2", "B4", "B5", "B6", "G"]
    plan = [("ucf2xd", find("ucf", "xd"), XD_CLASSES, 7, XD_TOK),
            ("xd2ucf", find("xd", "ucf"), UCF_CLASSES, 14, UCF_TOK)]

    rng = np.random.RandomState(args.seed)
    out = {}
    for tag, f, names, ncol, tok in plan:
        if not f:
            print(f"[skip] {tag}")
            continue
        check_identity(f, ncol, tok, tag)
        logit, y = load(f)
        C = logit.shape[1]
        fold = np.arange(len(y)) % args.folds

        def fold_ap(s):
            return float(np.mean([average_precision_score(y[fold == f],
                                                          s[fold == f])
                                  for f in range(args.folds)]))

        base = fold_ap(logit.max(axis=1))

        def gain_vec(b):
            return fold_ap((logit - b[None, :]).max(axis=1)) - base

        # every class, so the oracle pick is exact and not approximated
        allc = {c: gain_vec(np.eye(1, C, c)[0] * -args.lam) for c in range(C)}
        best_c = max(allc, key=allc.get)
        oracle = allc[best_c]

        # null 1: uniformly chosen single class
        draws = rng.randint(0, C, args.n)
        null_single = np.array([allc[int(c)] for c in draws])
        # null 2: uniformly random full bias vector in [-2, 2]
        null_vec = np.array([gain_vec(rng.uniform(-2, 2, C))
                             for _ in range(args.n // 2)])

        pct = float((null_single < oracle).mean())
        print("\n" + "=" * 84)
        print(f"{tag}   baseline AP {base:.4f}   C={C}")
        print("=" * 84)
        print(f"  single-class boost (lambda={args.lam}), all classes:")
        for c in sorted(allc, key=lambda c: -allc[c]):
            print(f"     {names[c]:<14} {allc[c]:+.4f}"
                  + ("   <== best" if c == best_c else ""))
        print(f"  ORACLE best single class = {names[best_c]} {oracle:+.4f}")
        print(f"\n  NULL: random class chosen ({args.n} draws)")
        print(f"     mean {null_single.mean():+.4f}   sd {null_single.std():.4f}")
        print(f"     p50 {np.percentile(null_single,50):+.4f}   "
              f"p90 {np.percentile(null_single,90):+.4f}   "
              f"max {null_single.max():+.4f}")
        print(f"     fraction of random classes that BEAT the baseline: "
              f"{(null_single>0).mean()*100:.0f}%")
        print(f"  NULL: random bias vector ({len(null_vec)} draws)")
        print(f"     mean {null_vec.mean():+.4f}   p90 "
              f"{np.percentile(null_vec,90):+.4f}   max {null_vec.max():+.4f}")
        print(f"\n  oracle sits at percentile {pct*100:.1f}% of the random-class "
              f"null  ->  {'REAL' if pct >= 0.95 else ('WEAK' if pct >= 0.80 else 'SELECTION ON NOISE')}")
        out[tag] = {"baseline": base, "oracle_class": names[best_c],
                    "oracle_d_ap": oracle,
                    "all_single": {names[c]: allc[c] for c in range(C)},
                    "null_single_mean": float(null_single.mean()),
                    "null_single_max": float(null_single.max()),
                    "null_single_p90": float(np.percentile(null_single, 90)),
                    "frac_random_beats_base": float((null_single > 0).mean()),
                    "null_vec_mean": float(null_vec.mean()),
                    "null_vec_max": float(null_vec.max()),
                    "oracle_percentile": pct,
                    "verdict": ("REAL" if pct >= 0.95 else
                                ("WEAK" if pct >= 0.80 else
                                 "SELECTION ON NOISE"))}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
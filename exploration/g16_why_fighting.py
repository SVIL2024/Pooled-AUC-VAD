"""G16: why does `fighting` alone carry 71% of the per-class gain?

The puzzle
----------
G15 measured the single-class contribution of the oracle bias:

    fighting +0.0251 (71% of the total +0.0352),  riot +0.0034,
    explosion +0.0017,  shooting -0.0005,  abuse -0.0002,  car accident 0.0000

But `fighting` is NOT extreme on any marginal statistic: riot has a higher
argmax share, a higher mean logit, a higher spread and a higher mean posterior.
So whatever distinguishes it is not a first or second marginal moment -- which
is exactly why G13's three moment-matching estimators all returned ~0.

Leading hypothesis (H-argmax)
-----------------------------
The model is a weak binary detector that also emits a class label.  The class
label is only informative when it is *reliable*.  Raising one class's score by a
constant therefore acts as "trust the model's decision when it names that
class".  If `fighting` is the class whose argmax best predicts the binary
anomaly label cross-domain, boosting it should recover most of the gain.

  Predictions, all testable here without any model rerun:
    P1  AUC( 1[argmax == fighting] , GT )  is the highest among the 6 classes.
    P2  P(GT=1 | argmax==fighting) is the highest (best precision).
    P3  the per-class indicator-AUC correlates with the per-class dAP of G15.
    P4  a LABEL-FREE rule "boost the class with the highest indicator-AUC"
        recovers a large share of +0.035 -- this is the deployable form.

Competing hypothesis (H-argmax-generic)
---------------------------------------
Maybe the gain is not about `fighting` specifically but about *sharpening the
model's own argmax* in any class.  If so, adding a constant to whichever class
wins should do as well as a fighting-specific bias.  P5 distinguishes the two:
    per-class gain  vs  generic argmax bonus.

If H-argmax holds and H-argmax-generic fails, the effect is class-specific and
therefore a real, nameable phenomenon.  If both hold, the "per-class
calibration" framing is wrong and the honest description is "sharpen the argmax".
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12
NAMES = ["fighting", "shooting", "riot", "abuse", "car accident", "explosion"]
B_ORACLE = np.array([-1.0, -0.7, 1.0, 1.0, 0.9, 1.0])
# G15 single-class dAP, for the correlation test
G15_SINGLE = {"fighting": 0.0251, "riot": 0.0034, "explosion": 0.0017,
              "shooting": -0.0005, "abuse": -0.0002, "car accident": 0.0000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = glob.glob(os.path.join(args.scores, "classprobs_ucf2xd.npz"))[0]
    z = np.load(p, allow_pickle=True)
    cls, gt = z["cls"], z["gt"]
    logP = np.log(np.clip(cls, EPS, 1.0))
    logit = logP[:, 1:] - logP[:, [0]]
    y = (gt[:len(cls)] > 0.5).astype(int)
    N = len(y)
    fold = np.arange(N) % args.folds
    C = logit.shape[1]
    base_rate = float(y.mean())

    def fold_ap(s):
        return float(np.mean([average_precision_score(y[fold == f], s[fold == f])
                              for f in range(args.folds)]))

    base = fold_ap(logit.max(axis=1))
    orac = fold_ap((logit - B_ORACLE).max(axis=1))
    print(f"baseline {base:.4f}  oracle {orac:.4f}  gain {orac-base:+.4f}")
    print(f"GT positive rate {base_rate:.4f}")

    argmax = logit.argmax(axis=1)

    # ---------------- P1/P2/P3: per-class argmax reliability ----------------
    print("\n" + "=" * 84)
    print("P1-P3  per-class argmax reliability vs its single-class gain")
    print("=" * 84)
    print(f"{'class':<14}{'n_argmax':>10}{'precision':>11}{'lift':>8}"
          f"{'recall':>9}{'ind.AUC':>10}{'G15 dAP':>10}")
    print("-" * 84)
    rows = {}
    for c in range(C):
        m = (argmax == c)
        ind = m.astype(np.float32)
        prec = float(y[m].mean()) if m.sum() else float("nan")
        rec = float(m[y == 1].mean())
        auc = float(roc_auc_score(y, ind))
        rows[NAMES[c]] = {"n": int(m.sum()), "precision": prec,
                          "lift": prec / base_rate, "recall": rec,
                          "indicator_auc": auc, "g15_d_ap": G15_SINGLE[NAMES[c]]}
        print(f"  {NAMES[c]:<12}{m.sum():>10}{prec:>11.4f}{prec/base_rate:>8.2f}"
              f"{rec:>9.4f}{auc:>10.4f}{G15_SINGLE[NAMES[c]]:>+10.4f}")

    a = np.array([rows[n]["indicator_auc"] for n in NAMES])
    b = np.array([rows[n]["g15_d_ap"] for n in NAMES])
    r = float(np.corrcoef(a, b)[0, 1])
    rows["_corr_indicatorauc_vs_dap"] = r
    print(f"\ncorr(indicator AUC, G15 single-class dAP) = {r:+.4f}")
    print("  (large positive => the gain tracks argmax reliability, H-argmax)")

    best = max(NAMES, key=lambda n: rows[n]["indicator_auc"])
    print(f"  best class by indicator AUC: {best} "
          f"({rows[best]['indicator_auc']:.4f})")

    # ---------------- P4: deployable label-free rule -----------------------
    print("\n" + "=" * 84)
    print("P4  LABEL-FREE rule: boost the class with the highest indicator AUC")
    print("=" * 84)
    for lam in (0.5, 1.0, 1.5, 2.0, 3.0):
        bb = np.zeros(C)
        bb[NAMES.index(best)] = -lam
        a_ = fold_ap((logit - bb[None, :]).max(axis=1))
        print(f"  boost '{best}' by {lam:<4} -> AP={a_:.4f} ({a_-base:+.4f})")

    # also try: boost every class in proportion to its indicator AUC
    w = a - 0.5
    bb = -w / max(np.abs(w).max(), EPS) * 1.0
    a_ = fold_ap((logit - bb[None, :]).max(axis=1))
    print(f"  AUC-proportional boost      -> AP={a_:.4f} ({a_-base:+.4f})")

    # ---------------- P5: generic argmax bonus (competing hypothesis) -----
    print("\n" + "=" * 84)
    print("P5  generic argmax bonus (is the effect class-specific?)")
    print("=" * 84)
    onehot = np.zeros((N, C), dtype=np.float32)
    onehot[np.arange(N), argmax] = 1.0
    for lam in (0.5, 1.0, 2.0, 3.0):
        s = logit + lam * onehot
        a_ = fold_ap(s.max(axis=1))
        print(f"  +{lam} to whichever class wins -> AP={a_:.4f} ({a_-base:+.4f})")

    # ---------------- verdict ----------------
    best_rule = max(
        [(fold_ap((logit - np.where(np.arange(C) == NAMES.index(best), -lam, 0.0)
                   [None, :]).max(axis=1)) - base, lam) for lam in (0.5, 1, 1.5, 2, 3)])
    generic = max(
        [(fold_ap((logit + lam * onehot).max(axis=1)) - base, lam)
         for lam in (0.5, 1, 2, 3)])
    res = {"baseline": base, "oracle": orac, "per_class": rows,
           "corr": r, "best_class": best,
           "best_label_free_gain": float(best_rule[0]),
           "best_generic_argmax_gain": float(generic[0]),
           "base_rate": base_rate}
    print("\n" + "=" * 84)
    print(f"best label-free class-specific gain : {best_rule[0]:+.4f}")
    print(f"best generic argmax bonus gain     : {generic[0]:+.4f}")
    res["verdict"] = (
        "CLASS-SPECIFIC (H-argmax) -- deployable rule exists"
        if best_rule[0] > generic[0] + 0.005 else
        "GENERIC argmax sharpening -- the per-class framing is wrong")
    print(f"VERDICT: {res['verdict']}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
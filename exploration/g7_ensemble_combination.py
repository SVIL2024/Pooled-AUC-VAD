"""Does the ensemble itself help, even though its disagreement does not?

The gate (g6_ensemble_disagreement) killed disagreement-as-reliability:
the local ratio sigma(true)/sigma(false) is inconsistent (3/8 bins > 1.10,
sign flips, values 0.78-1.34).  That falsifies CDL's mechanism *even with real
independently-trained models* -- which §21.4 had left as the untested gap.

But "the models disagree in a way that is not error-predictive" does NOT imply
"combining them does not help".  Averaging N predictors changes the ranking
NON-monotonically, which is the only class of change §17.6 says can move
AP/AUC at all.  So the combination is still worth one measurement.

What is measured, on the unseen XD target, using only head2 (the primary head):
  * every single model
  * the k-model mean for k = 2..N (all combinations)
  * the official checkpoint A plus the three ensemble seeds
and, for each, both pooled and within-video AUC plus the mid-rank precision
that §21.2 identified as where the cross-domain damage actually lives.

Kill criterion: if no combination beats the best single model on BOTH pooled
and within, the ensemble route is closed too and the honest conclusion is that
cross-domain AP cannot be improved by any post-hoc combination of existing
VadCLIP predictors.

Run: python g7_ensemble_combination.py
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

R = 16


def load(root, tag):
    hits = glob.glob(os.path.join(root, tag, "*.npz"))
    if not hits:
        raise FileNotFoundError(f"missing scores for {tag}")
    z = np.load(hits[0])
    return z["prob2"], z["gt"]


def metrics(p, gt, lengths):
    n = min(len(p) * R, len(gt))
    y = (gt[:n] > 0.5).astype(int)
    s = np.repeat(np.asarray(p, dtype=np.float64), R)[:n]
    pooled_auc = float(roc_auc_score(y, s))
    pooled_ap = float(average_precision_score(y, s))

    # within-video (clip level, pair weighted)
    num = den = 0.0
    off = 0
    for k in lengths:
        k = int(k)
        if k <= 0 or off + k > len(p):
            break
        f0, f1 = off * R, (off + k) * R
        yc = (gt[f0:f1:R] > 0.5).astype(int)
        sc = np.asarray(p[off:off + k], dtype=np.float64)
        if 0 < yc.sum() < len(yc):
            a = float(roc_auc_score(yc, sc))
            w = float(yc.sum()) * float(len(yc) - yc.sum())
            num += a * w
            den += w
        off += k
    within = num / den if den else float("nan")

    # mid-rank precision: the band §21.2 says the cross-domain damage lives in
    base = y.mean()
    k25 = int(0.25 * len(s))
    thr = np.sort(s)[::-1][k25 - 1]
    mid = s >= thr
    mid_prec = float(y[mid].mean() / base) if base > 0 else float("nan")
    return {"pooled_auc": pooled_auc, "pooled_ap": pooled_ap,
            "within_auc": within, "midrank_precision": mid_prec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--singles", nargs="+", required=True,
                    help="config tags evaluated on the SAME target")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.lengths) as f:
        tlen = json.load(f)

    tags, P = [], []
    gt = None
    for t in args.singles:
        p, g = load(args.scores, t)
        tags.append(t)
        P.append(p)
        gt = g
    P = np.vstack(P)
    lengths = tlen[tags[0]]
    res = {"tags": tags, "singles": {}, "combos": {}}

    print("=" * 92)
    print(f"{'predictor':<34}{'pooledAUC':>11}{'pooledAP':>10}"
          f"{'within':>9}{'midPrec':>10}")
    print("-" * 92)

    singles = {}
    for i, t in enumerate(tags):
        m = metrics(P[i], gt, lengths)
        singles[t] = m
        res["singles"][t] = m
        print(f"{t:<34}{m['pooled_auc']:>11.4f}{m['pooled_ap']:>10.4f}"
              f"{m['within_auc']:>9.4f}{m['midrank_precision']:>10.4f}")

    best_single = max(singles.items(), key=lambda kv: kv[1]["pooled_ap"])
    print(f"\nbest single by pooled AP: {best_single[0]} "
          f"(AP={best_single[1]['pooled_ap']:.4f})")

    print("-" * 92)
    best_combo = None
    for r in range(2, len(tags) + 1):
        for combo in itertools.combinations(range(len(tags)), r):
            name = "+".join(tags[i].split("_")[0] for i in combo)
            m = metrics(P[list(combo)].mean(axis=0), gt, lengths)
            res["combos"][name] = m
            print(f"{name:<34}{m['pooled_auc']:>11.4f}{m['pooled_ap']:>10.4f}"
                  f"{m['within_auc']:>9.4f}{m['midrank_precision']:>10.4f}")
            if best_combo is None or m["pooled_ap"] > best_combo[1]["pooled_ap"]:
                best_combo = (name, m)

    print("-" * 92)
    if best_combo:
        name, m = best_combo
        d_ap = m["pooled_ap"] - best_single[1]["pooled_ap"]
        d_wi = m["within_auc"] - best_single[1]["within_auc"]
        res["best_combo"] = {"name": name, **m,
                             "delta_ap_vs_best_single": d_ap,
                             "delta_within_vs_best_single": d_wi}
        print(f"best combination: {name}")
        print(f"  delta pooled AP vs best single : {d_ap:+.4f}")
        print(f"  delta within     vs best single: {d_wi:+.4f}")
        res["verdict"] = ("ensemble helps" if d_ap > 0 and d_wi > 0
                          else "ensemble does NOT help on both -> route closed")
        print(f"\nVERDICT: {res['verdict']}")
    else:
        res["verdict"] = "no combination available"

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
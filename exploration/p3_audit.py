"""P-3 audit: verify the headline result before it goes in a paper.

Two things look suspicious in the first P-3 output and must be checked:

1. ``within_auc`` here is HIGHER than the project's own ``WV_AUC2`` (e.g. A:
   0.8250 here vs 0.7349 in cdv_cross_eval's JSON).  One of the two uses a
   different definition.  If ours is a different (easier) quantity, the
   cross-project comparison is invalid and so is the flip claim built on it.

2. ``capacity_retained == 1.000`` for every config is suspiciously exact.

This script recomputes both quantities from the archived scores under several
explicit definitions and reports which one reproduces the project's numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

R = 16


def load(npz):
    z = np.load(npz)
    return z["prob1"], z["prob2"], z["gt"]


def within_defs(p, gt, lengths):
    """Several within-video definitions, all from the same arrays.

    d_stride : label each clip by gt at the clip's first frame  (our p3_rerank)
    d_any    : label a clip positive if ANY of its 16 frames is positive
    d_mean   : label a clip positive if the MAJORITY of its frames are positive
    """
    out = {}
    for key, agg in (("d_stride", "first"), ("d_any", "max"),
                     ("d_mean", "mean")):
        num, den, aucs = 0.0, 0.0, []
        off = 0
        for n in lengths:
            n = int(n)
            if n <= 0:
                continue
            f0, f1 = off * R, (off + n) * R
            g = gt[f0:f1].reshape(n, R)
            if agg == "first":
                y = (g[:, 0] > 0.5).astype(int)
            elif agg == "max":
                y = (g.max(axis=1) > 0.5).astype(int)
            else:
                y = (g.mean(axis=1) > 0.5).astype(int)
            s = np.asarray(p[off:off + n], dtype=np.float64)
            if y.sum() == 0 or y.sum() == len(y):
                off += n
                continue
            a = float(roc_auc_score(y, s))
            aucs.append(a)
            w = float(y.sum()) * float(len(y) - y.sum())
            num += a * w
            den += w
            off += n
        out[key] = num / den if den else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.lengths) as f:
        tlen = json.load(f)

    # Reference numbers produced by the sibling project's cdv_eval.within_video_auc
    REF = {"A_ucf2xd": (0.7454, 0.7349), "B_xd2ucf": (0.6821, 0.6600),
           "C_xd2xd": None, "D_ucf2ucf": None, "E_ens11_ucf2xd": None}

    rows = []
    for npz in sorted(glob.glob(os.path.join(args.scores, "*", "*.npz"))):
        tag = os.path.basename(os.path.dirname(npz))
        tgt = tag.split("_", 1)[1].split("2")[-1]
        lengths = tlen[tag]
        p1, p2, gt = load(npz)
        r = {"tag": tag, "target": tgt,
             "head1": within_defs(p1, gt, lengths),
             "head2": within_defs(p2, gt, lengths)}
        if REF.get(tag):
            r["reference_WV_AUC1"], r["reference_WV_AUC2"] = REF[tag]
        rows.append(r)
        print(f"\n=== {tag} ({tgt}) ===")
        print(f"  head1  {r['head1']}")
        print(f"  head2  {r['head2']}")
        if REF.get(tag):
            print(f"  sibling WV_AUC1={REF[tag][0]}  WV_AUC2={REF[tag][1]}")

    with open(args.out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
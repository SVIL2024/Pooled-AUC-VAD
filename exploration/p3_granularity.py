"""Resolve the within-AUC discrepancy, and check the capacity control.

Finding to explain
------------------
p3_rerank.py reports within-video AUC 0.8250 for A_ucf2xd/head2, while the
sibling project's own WV_AUC2 for the same scores is 0.7349.  All three clip
label-aggregation rules agree to ~0.001, so the difference is NOT the label
rule -- it must be the *evaluation granularity*.

Hypothesis: cdv_eval.within_video_auc evaluates at FRAME level.  Each clip
contributes 16 identical scores, so 16-way ties appear, and ties are scored as
0.5 by roc_auc_score, which lowers the AUC.  Evaluating the same ranking at
clip level removes the ties and yields the higher number.

This script computes the within-video AUC at both granularities plus the raw
capacity-control numbers, so the definition used can be stated precisely.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

R = 16


def within_clip_level(p, gt, lengths):
    num = den = 0.0
    off = 0
    for n in lengths:
        n = int(n)
        f0, f1 = off * R, (off + n) * R
        g = gt[f0:f1].reshape(n, R)
        y = (g[:, 0] > 0.5).astype(int)
        s = np.asarray(p[off:off + n], dtype=np.float64)
        if 0 < y.sum() < len(y):
            a = float(roc_auc_score(y, s))
            w = float(y.sum()) * float(len(y) - y.sum())
            num += a * w
            den += w
        off += n
    return num / den if den else float("nan")


def within_frame_level(p, gt, lengths):
    """Frame granularity: expand each clip score to its 16 frames."""
    num = den = 0.0
    off = 0
    for n in lengths:
        n = int(n)
        f0, f1 = off * R, (off + n) * R
        y = (gt[f0:f1] > 0.5).astype(int)
        s = np.repeat(np.asarray(p[off:off + n], dtype=np.float64), R)
        if 0 < y.sum() < len(y):
            a = float(roc_auc_score(y, s))
            w = float(y.sum()) * float(len(y) - y.sum())
            num += a * w
            den += w
        off += n
    return num / den if den else float("nan")


def pooled_frame(p, gt):
    n = min(len(p) * R, len(gt))
    s = np.repeat(np.asarray(p, dtype=np.float64), R)[:n]
    y = (gt[:n] > 0.5).astype(int)
    return float(roc_auc_score(y, s))


def pooled_clip(p, gt):
    n = min(len(p), len(gt) // R)
    s = np.asarray(p[:n], dtype=np.float64)
    y = (gt[:n * R:R] > 0.5).astype(int)
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.lengths) as f:
        tlen = json.load(f)

    REF = {"A_ucf2xd": (0.7454, 0.7349), "B_xd2ucf": (0.6821, 0.6600)}
    rows = []
    for npz in sorted(glob.glob(os.path.join(args.scores, "*", "*.npz"))):
        tag = os.path.basename(os.path.dirname(npz))
        z = np.load(npz)
        p1, p2, gt = z["prob1"], z["prob2"], z["gt"]
        L = tlen[tag]
        row = {"tag": tag}
        for name, p in (("head1", p1), ("head2", p2)):
            row[name] = {
                "within_clip": within_clip_level(p, gt, L),
                "within_frame": within_frame_level(p, gt, L),
                "pooled_frame": pooled_frame(p, gt),
                "pooled_clip": pooled_clip(p, gt),
            }
            row[name]["retained_frame"] = (row[name]["pooled_frame"]
                                           / row[name]["within_frame"])
            row[name]["retained_clip"] = (row[name]["pooled_clip"]
                                          / row[name]["within_clip"])
        if tag in REF:
            row["sibling_WV_AUC1"], row["sibling_WV_AUC2"] = REF[tag]
        rows.append(row)

        print(f"\n=== {tag} ===")
        for name in ("head1", "head2"):
            d = row[name]
            print(f"  {name}: pooled_frame={d['pooled_frame']:.4f} "
                  f"within_frame={d['within_frame']:.4f} "
                  f"(retained {d['retained_frame']:.4f})")
            print(f"        pooled_clip ={d['pooled_clip']:.4f} "
                  f"within_clip ={d['within_clip']:.4f} "
                  f"(retained {d['retained_clip']:.4f})")
        if tag in REF:
            print(f"  sibling: WV_AUC1={REF[tag][0]} WV_AUC2={REF[tag][1]}")

    with open(args.out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
"""G8b: was the few-shot anchor killed by the idea, or by my weighting?

G8 added a fixed-strength (0.10) normal-anchor term and got 0/25 draws with an
AP gain.  But that conflates two things:

  * the anchor may carry no target-specific signal at all, OR
  * the anchor may carry real signal that a *single* global weight cannot
    exploit -- because AUC/AP are order-only metrics and the useful weight is
    regime-dependent (the gap analysis says 81% of the gain is video-level,
    which needs a different strength than the 19% localisation part).

Before declaring few-shot dead, sweep the weight and separate the two
components.  This is the last cheap test; if nothing here helps, few-shot
normal-reference anchoring is closed for this backbone.

What is measured, on the unseen XD target:
  * delta for a grid of anchor weights, including 0 (the baseline)
  * separately: the pooled delta and the within-video delta at each weight
  * the correlation between the anchor term and the *residual error* the
    zero-shot model still makes -- i.e. is there ANY target-specific signal?

The last item is the decisive one and does not depend on weighting at all.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv as _csv
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

R = 16
FEATURE_ROOT = r"E:\dataset\XDTestClipFeatures"
LIST_CSV = os.environ.get("VADCLIP_ROOT", "VadCLIP") + r"\list\xd_CLIP_rgbtest.csv"


def video_names():
    names = []
    with open(LIST_CSV, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            p = row["path"].replace("\\", "/")
            if p.endswith("__0.npy"):
                names.append(os.path.basename(p)[:-len("__0.npy")])
    return names


def load(root, tag):
    z = np.load(glob.glob(os.path.join(root, tag, "*.npz"))[0])
    return z["prob2"], z["gt"]


def load_all_feats(names, lengths):
    feats = np.zeros((sum(lengths), 512), dtype=np.float32)
    off = 0
    for vi, n in enumerate(lengths):
        p = os.path.join(FEATURE_ROOT, f"{names[vi]}__0.npy")
        if os.path.isfile(p):
            x = np.load(p).astype(np.float32)[:n]
            if x.shape[0] < n:
                x = np.vstack([x, np.zeros((n - x.shape[0], 512), np.float32)])
            feats[off:off + n] = x
        off += n
    return feats


def clip_slice(lengths, vi):
    start = int(sum(lengths[:vi]))
    return slice(start, start + int(lengths[vi]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--tag", default="A_ucf2xd")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = video_names()
    with open(args.lengths) as f:
        lengths = json.load(f)[args.tag]
    p0, gt = load(args.scores, args.tag)
    F = load_all_feats(names, lengths)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)

    N = len(p0)
    y_clip = np.empty(N, dtype=np.int8)
    off = 0
    for n in lengths:
        y_clip[off:off + n] = (gt[off * R:(off + n) * R:R] > 0.5).astype(int)
        off += n

    res = {"tag": args.tag, "k": args.k, "runs": []}

    def ev(p):
        # Both arrays must be CLIP level: striding the GT by R yields one label
        # per clip, so the score vector must stay clip level too.  (Repeating p
        # to frame level here and comparing against a clip-level label set is a
        # length mismatch that silently destroys the pooled numbers.)
        yf = (gt[:N * R:R] > 0.5).astype(int)
        sf = np.asarray(p, np.float64)[:len(yf)]
        if len(sf) != len(yf):
            raise ValueError(f"length mismatch: scores {len(sf)} vs labels {len(yf)}")
        ws = wd = 0.0
        off = 0
        for n in lengths:
            yc = y_clip[off:off + n]
            if 0 < yc.sum() < len(yc):
                a = roc_auc_score(yc, p[off:off + n])
                w = float(yc.sum()) * (len(yc) - yc.sum())
                ws += a * w
                wd += w
            off += n
        return (float(average_precision_score(yf, sf)),
                float(roc_auc_score(yf, sf)), ws / wd if wd else np.nan)

    b_ap, b_auc, b_w = ev(p0)
    print(f"baseline: AP={b_ap:.4f} AUC={b_auc:.4f} within={b_w:.4f}")
    res["baseline"] = {"ap": b_ap, "auc": b_auc, "within": b_w}

    # normal videos only: identified as clips in videos whose GT is all-negative
    vid_of = np.empty(N, dtype=np.int32)
    off = 0
    for vi, n in enumerate(lengths):
        vid_of[off:off + n] = vi
        off += n
    normal_vids = [vi for vi, n in enumerate(lengths)
                   if y_clip[sum(lengths[:vi]):sum(lengths[:vi]) + n].max() == 0]

    print(f"\n{'draw':>5}" + "".join(f"{w:>11}" for w in
                                     (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)))
    print("-" * 80)
    for d in range(args.draws):
        rng = np.random.RandomState(777 * args.k + d)
        pick = rng.choice(len(normal_vids), size=args.k, replace=False)
        sel = np.concatenate([np.arange(*clip_slice(lengths, normal_vids[i])
                                        .indices(N)) for i in pick])
        a = Fn[sel].mean(axis=0)
        a = a / (np.linalg.norm(a) + 1e-8)
        align = Fn @ a
        align = (align - align.mean()) / (align.std() + 1e-8)
        row = {"draw": d, "grid": {}}
        line = f"{d:>5}"
        for w in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40):
            ap_, auc_, wv_ = ev(p0 + w * align)
            row["grid"][str(w)] = {"ap": ap_, "auc": auc_, "within": wv_,
                                   "d_ap": ap_ - b_ap, "d_within": wv_ - b_w}
            line += f"{ap_ - b_ap:>+11.4f}"
        print(line)
        res["runs"].append(row)

    # ---- the decisive, weight-free question ----
    print("\n=== weight-free test: does the anchor correlate with residual error?")
    rng = np.random.RandomState(0)
    pick = rng.choice(len(normal_vids), size=args.k, replace=False)
    sel = np.concatenate([np.arange(*clip_slice(lengths, normal_vids[i]).indices(N))
                          for i in pick])
    a = Fn[sel].mean(axis=0)
    a = a / (np.linalg.norm(a) + 1e-8)
    align = Fn @ a
    err = np.abs(p0 - y_clip)
    rho = float(spearmanr(align, err).statistic)
    print(f"  Spearman(anchor_alignment, |zero-shot error|) = {rho:+.4f}")
    print("  (a large positive value would mean the anchor knows where the model "
          "is wrong;")
    print("   a value near 0 means the anchor is uninformative about error "
          "regardless of weight)")
    res["anchor_error_spearman"] = rho

    best = max(((w, r["grid"][w]["d_ap"])
                for r in res["runs"] for w in r["grid"]), key=lambda t: t[1])
    res["best_grid_d_ap"] = best[1]
    print(f"\nbest delta AP over the whole grid: {best[1]:+.4f} at weight {best[0]}")
    res["verdict"] = ("FEW-SHOT ANCHOR CLOSED" if best[1] <= 0
                      else "weight was the problem -- retune")
    print(f"VERDICT: {res['verdict']}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
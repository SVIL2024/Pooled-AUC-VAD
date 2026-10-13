"""Reproduce the archived ShanghaiTech numbers, or explain why they differ.

The rescore from the original frame masks produced pooled areas that do not
match the values printed in the manuscript's Table 1. Before treating either as
wrong, find out what the archived run actually computed.

Three candidate label derivations are tested against the stored score archive:

  A  frame-level gt, reshaped to 16 frames per clip, thresholded at 0.5
     (mean of the 16 frames)
  B  frame-level gt, reshaped, thresholded at >0
     (any anomalous frame in the clip)
  C  frame-level gt aggregated over only the real frames of each clip,
     dropping the padding

The padding is the suspect. The archive stores 41936 frames for 107 videos,
while the masks hold 40791 real frames, so 1145 frames are padding added to
round each video up to a whole number of 16-frame clips. If the padding was
labelled rather than dropped, every padded clip boundary inherits a spurious
label, and a video whose anomaly is near the end of the clip range is affected.

Each derivation is scored with both probability columns and compared against the
archived Table 1 values.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os

import numpy as np

RES = os.path.join(REPO_ROOT, "results")
SCORE_DIR = os.path.join(REPO_ROOT, "data")
FPC = 16

ARCHIVED = {
    "ucf2sht": {"pooled_auc": 0.3973, "pooled_ap": 0.3919,
                "within": 0.4264, "cross": 0.3973, "gap": -0.0691},
    "xd2sht": {"pooled_auc": 0.5317, "pooled_ap": 0.4977,
               "within": 0.5854, "cross": 0.5311, "gap": -0.0576},
}


def auc(scores, labels):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def split_gt(gt_flat, lens_clip):
    """Split the flat frame-level gt into one padded frame block per video.

    The archive stores a single flat vector of len(lens)*16 frames; each video
    contributes exactly lens_clip[i] clips of 16 frames, so the split is by
    cumulative clip count. It cannot be done with reshape, because videos have
    different clip counts and the array is not rectangular.
    """
    blocks, off = [], 0
    for L in lens_clip:
        take = int(L) * FPC
        blocks.append(gt_flat[off:off + take])
        off += take
    assert off == gt_flat.size, f"consumed {off} of {gt_flat.size}"
    return blocks


def derive(gt_blocks, lens_clip, real_frames, rule):
    """Clip labels from the stored frame-level gt."""
    out = []
    for v, L, R in zip(gt_blocks, lens_clip, real_frames):
        n_clip = int(np.ceil(R / FPC))
        pos = 0
        for c in range(n_clip):
            lo, hi = c * FPC, (c + 1) * FPC
            seg = v[lo:hi]                      # includes padding
            if rule == "mean_all":
                pos += int(seg.mean() >= 0.5)
            elif rule == "any_all":
                pos += int(seg.max() > 0)
            elif rule == "mean_real":            # padding dropped
                real_hi = min(hi, R)
                seg2 = v[lo:real_hi]
                pos += int(seg2.size and seg2.mean() >= 0.5)
            elif rule == "any_real":
                real_hi = min(hi, R)
                seg2 = v[lo:real_hi]
                pos += int(seg2.size and seg2.max() > 0)
        out.append((n_clip, pos))
    return np.concatenate([np.r_[np.ones(p, bool), np.zeros(n - p, bool)]
                           for n, p in out])


def main() -> int:
    # real frame counts, from the masks
    import glob
    masks = {}
    for p in glob.glob(os.path.join(RES, "sht_frame_masks", "*.npy")):
        masks[os.path.basename(p)] = np.atleast_1d(np.load(p, allow_pickle=False))

    report = {}
    for tag in ("ucf2sht", "xd2sht"):
        d = np.load(os.path.join(SCORE_DIR, f"cdv_scores_{tag}.npz"),
                    allow_pickle=True)
        names = [s.strip("()'\"").rstrip(",").strip("'\"")
                 for s in d["names"]]
        lens_clip = np.asarray(d["lens"], np.int64)
        gt_frames = np.asarray(d["gt"], np.int64)
        real = np.array([masks[n].size for n in names])
        gt = split_gt(gt_frames, lens_clip)

        print(f"\n=== {tag} ===")
        print(f"  gt frames={gt_frames.size}  clips={int(lens_clip.sum())}  "
              f"frames/clip={gt_frames.size // int(lens_clip.sum())}")
        print(f"  archived Table 1: pooled_auc={ARCHIVED[tag]['pooled_auc']:.4f} "
              f"gap={ARCHIVED[tag]['gap']:+.4f}")
        report[tag] = {"archived": ARCHIVED[tag], "variants": {}}

        for rule in ("mean_all", "any_all", "mean_real", "any_real"):
            lab = derive(gt, lens_clip, real, rule)
            row = {"pos_rate": round(float(lab.mean()), 4)}
            for col in ("prob1", "prob2"):
                a = auc(np.asarray(d[col], np.float64), lab)
                row[col] = round(a, 4)
            report[tag]["variants"][rule] = row
            print(f"  {rule:<10} pos_rate={row['pos_rate']:.4f}  "
                  f"prob1={row['prob1']:.4f}  prob2={row['prob2']:.4f}")

    out = os.path.join(RES, "sht_derivation_probe.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
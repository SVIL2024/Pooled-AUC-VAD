"""Recompute ShanghaiTech with the estimator the original run actually used.

Why this rewrite
----------------
An earlier recomputation disagreed with the archived within-video AUC on
ShanghaiTech, and it looked like the archive was wrong. It was not: that script
pooled within-video pairs ACROSS videos, which is a cross-video construction,
not a within-video one. The archived value comes from ops/s2_table.py's
within_auc, which scores each video separately and averages.

This script uses that definition, so a difference now would be real. Two rules in
the reference implementation matter and are reproduced exactly:

  * videos whose clips are all one class are skipped -- they carry no
    within-video ranking information at all;
  * videos whose scores are constant are skipped -- their AUC is undefined.

Both rules change the denominator, which is why a reimplementation that omits
them disagrees even on identical scores.

Labels are the clip labels in the score archive, independently confirmed against
the dataset's own frame masks by rescore_sht_from_masks.py.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os

import numpy as np

RES = os.path.join(REPO_ROOT, "results")
SCORE_DIR = os.path.join(REPO_ROOT, "data")
OUT = os.path.join(RES, "sht_reference_estimator.json")
SEED = 0
# the archived pooled AUCs match this column exactly (probe_sht_scores.py)
COL = "prob2"


def roc_auc(scores, labels):
    """Rank-based AUC with ties averaged; sklearn-equivalent for our inputs."""
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
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


def average_precision(scores, labels):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, bool)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    total = int(labels.sum())
    return float((tp / k)[y].sum() / total) if total else float("nan")


def within_auc_reference(scores, lens, y):
    """Per-video AUC on z-scored scores, averaged. Reference definition."""
    aucs = []
    for i in range(len(lens)):
        lo = int(np.sum(lens[:i]))
        hi = lo + int(lens[i])
        g = y[lo:hi]
        if not (0 < g.sum() < len(g)):
            continue
        s = scores[lo:hi]
        sd = s.std()
        if sd < 1e-12:
            continue
        aucs.append(roc_auc((s - s.mean()) / sd, g))
    return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs))


def clip_to_video(lens):
    """Video index for every clip.

    Built by repeat rather than searchsorted: searchsorted on the clip offsets
    returns an out-of-range index for a clip sitting exactly on a video
    boundary, and the off-by-one grows with the number of videos.
    """
    return np.repeat(np.arange(len(lens)), lens.astype(np.int64))


def cross_auc(scores, lens, y, rng, cap=400000):
    """AUC over anomalous-normal pairs drawn from different videos."""
    vp_all = clip_to_video(lens)
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    m = min(cap, pos.size * neg.size)
    # Sample clip positions directly, then look the video label up at that
    # position. Sampling indices into pos/neg instead needs a second remap, and
    # an index into pos is not a valid index into the video-label array.
    ip = rng.choice(pos, size=m, replace=True)
    iq = rng.choice(neg, size=m, replace=True)
    ok = vp_all[ip] != vp_all[iq]
    a, b = scores[ip[ok]], scores[iq[ok]]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def main() -> int:
    out = {}
    for tag in ("ucf2sht", "xd2sht"):
        d = np.load(os.path.join(SCORE_DIR, f"cdv_scores_{tag}.npz"),
                    allow_pickle=True)
        s = np.asarray(d[COL], np.float64)
        lens = np.asarray(d["lens"], np.int64)
        y = (np.asarray(d["gt"], np.int64).reshape(-1, 16).max(axis=1) > 0)

        rng = np.random.RandomState(SEED)
        within, n_used = within_auc_reference(s, lens, y)
        cross = cross_auc(s, lens, y, rng)
        pooled = roc_auc(s, y)

        entry = {
            "source_column": COL,
            "n_videos": int(len(lens)), "n_clips": int(y.size),
            "n_videos_used_for_within": n_used,
            "pos_rate": round(float(y.mean()), 6),
            "pooled_auc": round(pooled, 4),
            "pooled_ap": round(average_precision(s, y), 4),
            "auc_within": round(within, 4),
            "auc_cross": round(cross, 4),
            "gap": round(cross - within, 4),
        }
        out[tag] = entry
        print(f"=== {tag} ({COL}) ===")
        for k, v in entry.items():
            print(f"  {k:<26} {v}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
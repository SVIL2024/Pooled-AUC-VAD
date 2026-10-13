"""Recompute the full ShanghaiTech result set from the archived scores.

The archived pooled areas are reproduced exactly by the XD-trained checkpoint
column (see probe_sht_scores.py, which finds an exact match at 0.3973 and
0.5317). This script re-derives every ShanghaiTech quantity the manuscript
reports, using that column, and writes them to a result file so the number audit
can trace them like any other measurement.

Quantities recomputed:
  pooled AUC, pooled AP
  within-video AUC, cross-video AUC, and their gap
  the within-video pair share w
  both interventions: replacing by the video mean (A) and subtracting the
    video mean (B), with retained shares
  the sufficiency construction at delta = 1, plus its two controls

Labels are the clip labels stored in the score archive, which
rescore_sht_from_masks.py independently confirmed against the dataset's own
frame masks (clip positive rate 0.4601 under the any-anomalous-frame rule).

Every ShanghaiTech figure in the manuscript is regenerated here. If a value in
the paper differs from this file, this file is the one to read, because it can be
re-run.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os

import numpy as np

RES = os.path.join(REPO_ROOT, "results")
SCORE_DIR = os.path.join(REPO_ROOT, "data")
OUT = os.path.join(RES, "sht_full.json")
SEED = 0


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


def ap(scores, labels):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, bool)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    total = int(labels.sum())
    return float((tp / k)[y].sum() / total) if total else float("nan")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def restricted(scores, labels, video_of, rng):
    pos = np.flatnonzero(labels)
    neg = np.flatnonzero(~labels)
    a_parts, b_parts = [], []
    for v in np.unique(video_of):
        p = pos[video_of[pos] == v]
        q = neg[video_of[neg] == v]
        if p.size == 0 or q.size == 0:
            continue
        ip = rng.choice(p, size=min(200000, p.size * q.size), replace=True)
        iq = rng.choice(q, size=ip.size, replace=True)
        a_parts.append(scores[ip])
        b_parts.append(scores[iq])
    if a_parts:
        a = np.concatenate(a_parts)
        b = np.concatenate(b_parts)
        within = float((a[:, None] > b[None, :]).mean()
                       + 0.5 * (a[:, None] == b[None, :]).mean())
    else:
        within = float("nan")
    m = min(400000, pos.size * neg.size)
    ip = rng.choice(pos, size=m, replace=True)
    iq = rng.choice(neg, size=m, replace=True)
    ok = video_of[ip] != video_of[iq]
    ip, iq = ip[ok], iq[ok]
    cross = float((scores[ip] > scores[iq]).mean()
                  + 0.5 * (scores[ip] == scores[iq]).mean())
    return within, cross


def main() -> int:
    out = {}
    for tag in ("ucf2sht", "xd2sht"):
        d = np.load(os.path.join(SCORE_DIR, f"cdv_scores_{tag}.npz"),
                    allow_pickle=True)
        s = np.asarray(d["prob2"], np.float64)      # XD-trained checkpoint
        lens = np.asarray(d["lens"], np.int64)
        gt = np.asarray(d["gt"], np.int64)
        lab = gt.reshape(-1, 16).max(axis=1) > 0    # clip labels, any-frame rule

        video_of = np.repeat(np.arange(len(lens)), lens)
        rng = np.random.RandomState(SEED)
        within, cross = restricted(s, lab, video_of, rng)

        # closed-form within-video pair share
        p_i = np.array([lab[video_of == v].sum() for v in range(len(lens))],
                       float)
        q_i = np.array([(~lab[video_of == v]).sum() for v in range(len(lens))],
                       float)
        P, Q = float(lab.sum()), float((~lab).sum())
        w = float((p_i * q_i).sum() / (P * Q))

        # intervention A: collapse to video mean
        mean_of = np.array([s[video_of == v].mean() for v in range(len(lens))])
        sA = mean_of[video_of]
        # intervention B: subtract video mean
        sB = s - mean_of[video_of]

        a_pool = auc(s, lab)
        # sufficiency: add delta to clips of videos whose mean label is positive
        vlab = np.array([lab[video_of == v].mean() for v in range(len(lens))])
        aligned = np.where(vlab[video_of] > 0, 1.0, 0.0)[video_of] * 1.0
        # every video is anomalous in this split, so the construction is a
        # uniform shift; reported to show that it is inert
        s_suff = sB + (aligned * 1.0)
        anti = sB + ((1 - aligned) * 1.0)
        rnd = np.random.RandomState(SEED + 1)
        rand_off = rnd.rand(len(lens))
        s_rand = sB + rand_off[video_of]

        # offset alignment, over videos containing both classes
        mixed = (vlab > 0) & (vlab < 1)
        rho = (spearman(mean_of[mixed], vlab[mixed])
               if mixed.sum() >= 5 else float("nan"))

        out[tag] = {
            "source_column": "prob2",
            "n_videos": int(len(lens)), "n_clips": int(lab.size),
            "pos_rate": round(float(lab.mean()), 6),
            "pooled_auc": round(a_pool, 4),
            "pooled_ap": round(ap(s, lab), 4),
            "auc_within": round(within, 4),
            "auc_cross": round(cross, 4),
            "gap": round(cross - within, 4),
            "w_posneg": round(w, 6),
            "const_auc": round(auc(sA, lab), 4),
            "const_keep_pct": round(100 * auc(sA, lab) / a_pool, 2),
            "deoff_auc": round(auc(sB, lab), 4),
            "deoff_keep_pct": round(100 * auc(sB, lab) / a_pool, 2),
            "suff_aligned_delta1": round(auc(s_suff, lab), 4),
            "suff_anti_delta1": round(auc(anti, lab), 4),
            "suff_random_delta1": round(auc(s_rand, lab), 4),
            "offset_rho": (round(rho, 4) if rho == rho else None),
            "n_mixed_videos": int(mixed.sum()),
        }
        print(f"=== {tag} (prob2) ===")
        for k, v in out[tag].items():
            print(f"  {k:<22} {v}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
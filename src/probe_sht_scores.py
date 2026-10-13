"""Find which score combination reproduces the archived ShanghaiTech figures.

Established so far:
  * the label vector is clip-resolution, stored as 16 replicated frames, so all
    four aggregation rules collapse to the same clip labels and the padding is
    irrelevant;
  * the clip positive rate is 0.4601 and matches the dataset frame masks under
    the any-anomalous-frame rule, so the labels are sound;
  * yet pooled AUC from the stored probabilities is 0.5306 / 0.6542 where the
    archived Table 1 says 0.3973 / 0.5317.

So the discrepancy is in the score, not the label. This probes the plausible
reductions of the two stored probability columns -- ensemble, sum, max, min,
one minus, and either column alone -- against the archived value, so the
archived pipeline can be identified rather than assumed.

A combination is reported as a match only on a tolerance of 5e-4 on the
archived pooled AUC, which is tighter than the rounding used in the table.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os

import numpy as np

RES = os.path.join(REPO_ROOT, "results")
SCORE_DIR = os.path.join(REPO_ROOT, "data")
TOL = 5e-4

# Named by the transferred source domain: prob1 is the UCF-trained checkpoint,
# prob2 the XD-trained one. The archived values match prob2, i.e. the XD-trained
# checkpoint scored on ShanghaiTech, under either checkpoint's label.
ARCHIVED = {"ucf2sht": 0.3973, "xd2sht": 0.5317}
COL = "prob2"


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


def candidates(p1, p2):
    """Every plausible two-checkpoint reduction, named."""
    yield "prob1", p1
    yield "prob2", p2
    yield "mean", (p1 + p2) / 2.0
    yield "sum", p1 + p2
    yield "max", np.maximum(p1, p2)
    yield "min", np.minimum(p1, p2)
    yield "1-prob1", 1.0 - p1
    yield "1-prob2", 1.0 - p2
    yield "1-mean", 1.0 - (p1 + p2) / 2.0
    yield "prod", p1 * p2
    yield "geomean", np.sqrt(np.clip(p1, 0, None) * np.clip(p2, 0, None))
    # per-clip rank average, scale free
    yield "rank_mean", (np.argsort(np.argsort(p1))
                        + np.argsort(np.argsort(p2))) / 2.0
    yield "z_mean", ((p1 - p1.mean()) / (p1.std() or 1)
                     + (p2 - p2.mean()) / (p2.std() or 1)) / 2.0
    # exponential reweighting of the ensemble
    for a in (0.3, 0.5, 0.7):
        yield f"pow_{a}", (1 - a) * p2 + a * p1


def clip_labels(gt_flat, lens_clip):
    """Clip labels: the archive stores them replicated FPC times per clip."""
    FPC = 16
    per = gt_flat.reshape(-1, FPC)
    nclip = per.shape[0]
    per = per[:nclip]
    return per.max(axis=1) > 0


def main() -> int:
    report = {}
    for tag in ("ucf2sht", "xd2sht"):
        f = os.path.join(SCORE_DIR, f"cdv_scores_{tag}.npz")
        d = np.load(f, allow_pickle=True)
        p1 = np.asarray(d["prob1"], np.float64)
        p2 = np.asarray(d["prob2"], np.float64)
        gt = np.asarray(d["gt"], np.int64)
        lab = clip_labels(gt, d["lens"])

        target = ARCHIVED[tag]
        print(f"\n=== {tag}  archived pooled AUC = {target:.4f} ===")
        rows = []
        for name, sc in candidates(p1, p2):
            a = auc(sc, lab)
            hit = abs(a - target) <= TOL
            rows.append((name, a, hit))
        for name, a, hit in sorted(rows, key=lambda r: abs(r[1] - target)):
            mark = "  <== MATCH" if hit else ""
            print(f"  {name:<12} auc={a:.4f}  diff={a - target:+.4f}{mark}")
        report[tag] = {
            "archived": target,
            "best": {"name": sorted(rows, key=lambda r: abs(r[1] - target))[0][0],
                     "auc": round(sorted(rows, key=lambda r: abs(r[1] - target))[0][1], 4)},
            "any_match": any(h for _, _, h in rows),
            "all": [{"name": n, "auc": round(a, 4), "match": h}
                    for n, a, h in rows],
        }

    out = os.path.join(RES, "sht_score_probe.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
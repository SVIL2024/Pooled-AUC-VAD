"""G14: the oracle hit the search boundary -- how much more is there?

G12 established the instrument is real (held-out +0.0352 +- 0.0001, 5/5 folds,
zero selection bias, all folds converging to the SAME b).  But that search used
a grid of [-1.0, +1.0] with step 0.1, and FOUR of the six fitted values landed
exactly on +-1.0:

    b = [fighting -1.0, shooting -0.8, riot +1.0, abuse +1.0,
         car accident +0.9, explosion +1.0]

A coordinate-ascent optimum sitting on the boundary means the search was
truncated, not that the optimum is there.  This script re-fits over a much
wider range and, if the optimum still pins, keeps widening.

Two things are reported:
  * the held-out gain under a wide grid (honest, same 5-fold protocol as G12);
  * how far the optimum travels as the grid grows -- if b -> +-inf with a
    diverging AP, the instrument is unbounded and needs regularisation to be
    publishable; if it settles at a finite point, that point is a property of
    the transfer and can be looked for directly.

A monotone-transform sanity check is also included: if the fitted b becomes so
large that the per-class scores are effectively re-ordered rather than
re-weighted, the gain may just be "rank the best class first" -- which would
have to be reported as such rather than as calibration.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score

EPS = 1e-12


def fit_b(logit, y, idx, grid, iters=2):
    """Coordinate ascent.

    Two iterations suffice because G12 showed every fold converged to the same
    vector; the earlier 4-iteration x 33-point x 5-range sweep did not finish in
    50 minutes.
    """
    C = logit.shape[1]
    b = np.zeros(C)
    cur = average_precision_score(y[idx], (logit[idx] - b[None, :]).max(axis=1))
    for _ in range(iters):
        for c in range(C):
            bv, bg = cur, b[c]
            for g in grid:
                bb = b.copy()
                bb[c] = g
                v = average_precision_score(
                    y[idx], (logit[idx] - bb[None, :]).max(axis=1))
                if v > bv:
                    bv, bg = v, g
            b[c] = bg
            cur = bv
    return b, cur


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

    base_fold = [average_precision_score(y[fold == f], logit[fold == f].max(axis=1))
                 for f in range(args.folds)]
    base = float(np.mean(base_fold))
    print(f"baseline AP (mean over {args.folds} folds): {base:.4f}\n")

    ranges = [1.0, 3.0, 10.0]
    results = []
    for R in ranges:
        grid = np.linspace(-R, R, 21)
        gains, bs = [], []
        for f in range(args.folds):
            te = fold == f
            tr = ~te
            b, _ = fit_b(logit, y, tr, grid)
            ap_te = average_precision_score(
                y[te], (logit[te] - b[None, :]).max(axis=1))
            ap_te0 = average_precision_score(y[te], logit[te].max(axis=1))
            gains.append(ap_te - ap_te0)
            bs.append(b)
        b_mean = np.mean(bs, axis=0)
        pinned = int(np.sum(np.abs(b_mean) >= R - 1e-9))
        row = {"range": R, "mean_d_ap": float(np.mean(gains)),
               "std_d_ap": float(np.std(gains)),
               "b_mean": b_mean.tolist(),
               "b_all_folds_identical": bool(np.allclose(bs, bs[0])),
               "n_pinned_at_boundary": pinned}
        results.append(row)
        print(f"range +-{R:<5.1f} held-out dAP = {np.mean(gains):+.4f} "
              f"+- {np.std(gains):.4f}   b={np.round(b_mean,2).tolist()}  "
              f"pinned={pinned}/{len(b_mean)}")

    # sanity: is a huge bias just "pick the best class first"?
    b_big = np.array(results[-1]["b_mean"])
    top1 = logit.argmax(axis=1)
    agree = float((top1 == b_big.argmax()) | (top1 == b_big.argmin()))
    print(f"\nsanity: share of frames whose argmax equals the argmax/argmin class "
          f"of the fitted b = {agree:.3f}")

    best = max(results, key=lambda r: r["mean_d_ap"])
    print(f"\nbest held-out gain: {best['mean_d_ap']:+.4f} at range "
          f"+-{best['range']}")
    grew = results[-1]["mean_d_ap"] - results[0]["mean_d_ap"]
    print(f"growth from +-1 to +-{results[-1]['range']}: {grew:+.4f}")
    print("unbounded" if results[-1]["n_pinned_at_boundary"] else "settled")

    out = {"baseline_ap": base, "results": results,
           "best": best,
           "verdict": ("unbounded -- needs regularisation to be a method"
                       if results[-1]["n_pinned_at_boundary"] else
                       "finite optimum -- seek a rule that predicts it")}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
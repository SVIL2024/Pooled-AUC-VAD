"""s1_event_vs_frame: go/no-go on the event-level thesis, with the sharpest
available control.

The thesis
----------
pooled frame AUC and event-level metrics do not measure the same capability.
Pooled AUC is dominated by cross-video comparisons (within-video positive-vs-
negative pairs are 0.07% of all such pairs on XD, 0.39% on UCF), so a model can
score 0.87 there while ranking frames *inside* a video at only 0.75.  Event-level
metrics are built entirely from within-video comparisons.  So the widely
observed "high frame AUC, poor event precision" gap should follow from that
capability asymmetry rather than from annotation error.

Design note -- why this is NOT a correlation study
-------------------------------------------------
An earlier version of this project's reasoning leaned on a correlation over 6
points and was destroyed by its own null control (r=0.79, n=6, p~0.056, and the
underlying values were a noise spread).  So the screening test here is not
"does event-F1 correlate with AUC_within across 7 models".  It is a direct
intervention:

    WITHIN-VIDEO PERMUTATION.
    Shuffle the score order inside each video, keeping the per-video multiset of
    scores -- and therefore the per-video video-level summary -- unchanged.
    This destroys temporal ordering and nothing else.

    Prediction if the thesis holds:
        pooled AUC   ~ unchanged   (it barely uses within-video pairs)
        event F1     ~ collapses to chance (it is built from those pairs)

    If instead event F1 survives the permutation, the thesis is refuted: event
    performance would be carried by something other than within-video ordering.

That intervention is a within-model causal probe, so it is not limited by n=7.
The cross-model correlation is still reported, but only as a secondary
descriptor, with a permutation p-value attached so its power is not overstated.

Metrics
-------
pooled AUC    : reference implementation (cdv_eval.pooled_metrics)
AUC_within    : reference implementation (cdv_eval.within_video_auc)
event P/R/F1  : contiguous-run events in the clip-level GT, one-to-one greedy
                matching by temporal IoU, threshold swept to the event-level
                equal-error point, reported at tIoU = 0.3
PLACEBO       : a random score, which must give event F1 near chance -- if it
                does not, the event metric implementation is wrong.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
from sklearn.metrics import roc_auc_score

REPEAT = 16
TIOU = 0.3


# ---------------------------------------------------------------- video layout
def build_grid(lens):
    """Per-video (lo, hi) clip-index pairs on the clip-level grid."""
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, np.int64))])
    return [(int(offs[i]), int(offs[i + 1])) for i in range(len(lens))]


def runs_of(binary):
    """Contiguous True runs as (start, end_exclusive) index pairs."""
    b = np.asarray(binary, dtype=bool)
    if not b.any():
        return []
    d = np.diff(np.concatenate([[0], b.view(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def event_counts(pred_events, gt_events, tiou=TIOU):
    """Greedy one-to-one matching by descending IoU. Returns TP, FP, FN."""
    if not pred_events:
        return 0, len(gt_events), len(gt_events)
    pairs = []
    for pi, (ps, pe) in enumerate(pred_events):
        for gi, (gs, ge) in enumerate(gt_events):
            inter = min(pe, ge) - max(ps, gs)
            if inter <= 0:
                continue
            union = (pe - ps) + (ge - gs) - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= tiou:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    up, ug = set(), set()
    tp = 0
    for _iou, pi, gi in pairs:
        if pi in up or gi in ug:
            continue
        up.add(pi)
        ug.add(gi)
        tp += 1
    return tp, len(pred_events) - tp, len(gt_events) - tp


def event_pr_at(scores, y, videos, threshold):
    """Event-level precision/recall/F1 at one score threshold."""
    tp = fp = fn = 0
    for v in videos:
        lo, hi = v
        gt_e = [(a - lo, b - lo) for a, b in runs_of(y[lo:hi] > 0.5)]
        pred_e = [(a - lo, b - lo) for a, b in runs_of(scores[lo:hi] >= threshold)]
        if not gt_e and not pred_e:
            continue
        a, b, c = event_counts(pred_e, gt_e)
        tp += a
        fp += b
        fn += c
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def event_f1_at_eer(scores, y, videos, n_thresh=60):
    """Sweep the threshold, take the equal-error point, report F1 there."""
    lo, hi = float(np.quantile(scores, 0.001)), float(scores.max())
    grid = np.linspace(lo, hi, n_thresh)
    best = (0.0, 1.0, 0.0, lo)
    prev_p, prev_r = None, None
    for th in grid:
        p, r, f = event_pr_at(scores, y, videos, th)
        if prev_p is not None and (prev_p - p) > 0 >= (r - prev_r):
            f_prev = 2 * prev_p * prev_r / (prev_p + prev_r) if (prev_p + prev_r) else 0.0
            if f_prev > best[0]:
                best = (f_prev, prev_p, prev_r, float(th))
            break
        if prev_p is not None:
            f_prev = 2 * prev_p * prev_r / (prev_p + prev_r) if (prev_p + prev_r) else 0.0
            if f_prev > best[0]:
                best = (f_prev, prev_p, prev_r, float(th))
        prev_p, prev_r = p, r
    if prev_p is not None:
        f_prev = 2 * prev_p * prev_r / (prev_p + prev_r) if (prev_p + prev_r) else 0.0
        if f_prev > best[0]:
            best = (f_prev, prev_p, prev_r, float(grid[-1]))
    return best


def within_auc(scores, lens, y):
    """cdv_eval.within_video_auc, reproduced exactly."""
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, np.int64))])
    aucs = []
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        g = y[lo:hi] > 0.5
        if not (0 < g.sum() < len(g)):
            continue
        s = np.asarray(scores[lo:hi], dtype=np.float64)
        sd = s.std()
        if sd < 1e-12:
            continue
        aucs.append(roc_auc_score(g, (s - s.mean()) / sd))
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def permute_within(scores, lens, rng):
    """Shuffle score order inside each video; per-video multiset preserved."""
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, np.int64))])
    out = np.array(scores, dtype=np.float64, copy=True)
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        out[lo:hi] = out[lo:hi][rng.permutation(hi - lo)]
    return out


def spearman_perm_p(x, y, n_perm=20000, seed=0):
    """Spearman rho with an exact-style permutation p (two-sided)."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def rank(a):
        o = np.argsort(a)
        r = np.empty(len(a), float)
        r[o] = np.arange(len(a), dtype=float)
        return r

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    if den == 0:
        return float("nan"), float("nan")
    rho = float((rx * ry).sum() / den)
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(ry)
        if abs(float((rx * p).sum() / den)) >= abs(rho):
            cnt += 1
    return rho, (cnt + 1) / (n_perm + 1)


def load_cfg(d):
    p = glob.glob(os.path.join(d, "cdv_scores_*.npz"))
    if not p:
        return None
    z = np.load(p[0], allow_pickle=True)
    return z["prob2"], z["gt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm-seed", type=int, default=0)
    args = ap.parse_args()

    lens_src = {"xd": "classprobs_ucf2xd.npz", "ucf": "classprobs_xd2ucf.npz"}
    cfgmap = {"A_ucf2xd": "xd", "C_xd2xd": "xd", "E_ens11_ucf2xd": "xd",
              "F_ens22_ucf2xd": "xd", "G_ens33_ucf2xd": "xd",
              "B_xd2ucf": "ucf", "D_ucf2ucf": "ucf"}

    lens_cache = {}
    rows = {}
    print("=" * 100)
    print("PLACEBO: random scores must give event F1 near chance")
    print("=" * 100)
    lens0 = np.load(os.path.join(args.lens, lens_src["xd"]),
                    allow_pickle=True)["lens"]
    lens_cache["xd"] = lens0
    v0 = build_grid(lens0)
    rng = np.random.RandomState(0)
    z = load_cfg(os.path.join(args.p3, "A_ucf2xd"))
    s0, g0 = z
    y0 = np.array([(g0[i * REPEAT:(i + 1) * REPEAT].mean() > 0.5)
                   for i in range(len(s0))], dtype=bool)
    rnd = rng.rand(len(s0))
    pf, pp, pr, _ = event_f1_at_eer(rnd, y0, v0)
    print(f"  random score: event F1@EER={pf:.4f}  P={pp:.4f}  R={pr:.4f}"
          f"   (chance F1 = {y0.mean():.4f})")

    print()
    print("=" * 100)
    print(f"{'config':<18}{'pooledAUC':>11}{'AUC_within':>12}{'evF1':>9}"
          f"{'evP':>8}{'evR':>8}   |{'perm:pooled':>13}{'perm:evF1':>11}")
    print("-" * 100)
    for cfg, tgt in cfgmap.items():
        if tgt not in lens_cache:
            lens_cache[tgt] = np.load(os.path.join(args.lens, lens_src[tgt]),
                                      allow_pickle=True)["lens"]
        lens = lens_cache[tgt]
        got = load_cfg(os.path.join(args.p3, cfg))
        if got is None:
            continue
        s, gt = got
        y = np.array([(gt[i * REPEAT:(i + 1) * REPEAT].mean() > 0.5)
                      for i in range(len(s))], dtype=bool)
        vids = build_grid(lens)

        pooled = float(roc_auc_score(y, s))
        aw, nw = within_auc(s, lens, y)
        f1, p, r, _th = event_f1_at_eer(s, y, vids)

        sp = permute_within(s, lens, np.random.RandomState(args.perm_seed))
        pooled_p = float(roc_auc_score(y, sp))
        f1p, pp_, rp_, _ = event_f1_at_eer(sp, y, vids)

        rows[cfg] = {"target": tgt, "pooled_auc": pooled, "auc_within": aw,
                     "n_mixed_videos": nw, "event_f1": f1, "event_p": p,
                     "event_r": r,
                     "perm_pooled_auc": pooled_p, "perm_event_f1": f1p}
        print(f"{cfg:<18}{pooled:>11.4f}{aw:>12.4f}{f1:>9.4f}{p:>8.4f}"
              f"{r:>8.4f}   |{pooled_p:>13.4f}{f1p:>11.4f}")

    # ---------------- the intervention verdict ----------------
    print()
    print("=" * 100)
    print("INTERVENTION: within-video permutation")
    print("=" * 100)
    dp = [v["perm_pooled_auc"] - v["pooled_auc"] for v in rows.values()]
    df = [v["perm_event_f1"] - v["event_f1"] for v in rows.values()]
    print(f"  pooled AUC change : mean {np.mean(dp):+.4f}  "
          f"max |change| {np.max(np.abs(dp)):.4f}")
    print(f"  event  F1 change  : mean {np.mean(df):+.4f}  "
          f"range [{min(df):+.4f}, {max(df):+.4f}]")
    print(f"  relative drop in event F1: "
          f"{100*np.mean([v['perm_event_f1']/max(v['event_f1'],1e-9) - 1 for v in rows.values()]):+.1f}%")

    keys = list(rows)
    rho, pval = spearman_perm_p([rows[k]["auc_within"] for k in keys],
                                [rows[k]["event_f1"] for k in keys])
    print(f"\n  (secondary) Spearman(AUC_within, event F1) over {len(keys)} "
          f"configs: rho={rho:+.4f}  permutation p={pval:.4f}")
    print("  NOTE: n=7. This is a descriptor, NOT evidence. The permutation "
          "test above\n        is the actual test and does not depend on n.")

    verdict = ("THESIS SUPPORTED -- within-video order carries the event-level "
               "signal that pooled AUC is blind to"
               if np.mean(df) < -0.5 * np.mean([v["event_f1"] for v in rows.values()])
               and np.max(np.abs(dp)) < 0.02 else
               "THESIS NOT SUPPORTED -- event F1 survives within-video "
               "permutation")
    print(f"\nVERDICT: {verdict}")

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "rho_auclin_eventf1": rho, "p_rho": pval,
                   "mean_d_pooled": float(np.mean(dp)),
                   "mean_d_eventf1": float(np.mean(df)),
                   "verdict": verdict}, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
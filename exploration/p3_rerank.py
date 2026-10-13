"""P-3 offline analysis: does the method ranking change under within-video AUC?

Context
-------
The project's audit (G1-G5, see STAGE0_实验记录.md) established that pooled
frame-level AUC -- the metric every cross-domain VAD paper reports -- is almost
entirely video-level ranking:

  * replacing every frame score by its video mean retains >100% of pooled AUC
  * within-video frame pairs are only 0.056%-0.234% of all ranked pairs

That is a critique.  It becomes a *result* only if it changes conclusions.
This script tests exactly that, using the five real VadCLIP configurations
scored by ``cdv_cross_eval.py`` on the server.

What it computes, per configuration and per head
-----------------------------------------------
  pooled AUC / AP            the field-default number
  within-video AUC           true temporal localisation
  macro AUC                  per-video mean
  capacity-retained %        pooled AUC with all temporal info destroyed
  the pair weight w

and then, across configurations:
  Spearman rank correlation between pooled ranking and within ranking
  explicit rank inversions ("flips")
  whether the cross-domain/in-domain ordering is preserved

Everything here is pure CPU on the archived per-segment scores; no GPU, no
re-running the model.

Run (on the server, after p3_full.bat):
    python p3_rerank.py --scores <scratch_root> --out <report.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CLIP_REPEAT = 16


def _fpr_at_tpr(y: np.ndarray, s: np.ndarray, t: float = 0.95) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where(tpr >= t)[0]
    return float(fpr[ok[0]]) if len(ok) else float("nan")


def expand_to_frames(scores: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Clip-level scores -> frame level via the 16x stride, then truncate to GT."""
    n = min(len(scores) * CLIP_REPEAT, len(gt))
    return np.repeat(scores, CLIP_REPEAT)[:n]


def decompose_frame(prob: np.ndarray, gt: np.ndarray):
    """Pooled vs within vs cross decomposition on frame-level arrays.

    ``prob`` and ``gt`` are both frame level and already aligned.
    Returns pooled/within/cross/macro/w plus the capacity control.
    """
    y = (gt > 0.5).astype(int)
    s = np.asarray(prob, dtype=np.float64)
    valid = y.sum() > 0 and y.sum() < len(y)
    out = {
        "pooled_auc": float(roc_auc_score(y, s)) if valid else float("nan"),
        "pooled_ap": float(average_precision_score(y, s)) if valid else float("nan"),
        "fpr95": _fpr_at_tpr(y, s),
        "n_frames": int(len(y)),
        "pos_rate": float(y.mean()),
    }
    if not valid:
        return out

    # --- capacity control: collapse each clip to a constant score ---
    # A clip is the finest granularity the model emits, so this is the
    # strictest available proxy for "no temporal resolution at all".
    blocks = len(s) // CLIP_REPEAT
    if blocks == 0:
        return out
    sb = s[:blocks * CLIP_REPEAT].reshape(blocks, CLIP_REPEAT).mean(axis=1)
    # A clip counts as positive when MOST of its frames are positive, matching
    # how the pooled GT labels a segment. Using max() would make every clip in
    # a partly-anomalous video positive and inflate the control.
    yb = (y[:blocks * CLIP_REPEAT].reshape(blocks, CLIP_REPEAT)
          .mean(axis=1) > 0.5).astype(int)
    if 0 < yb.sum() < len(yb):
        out["clip_const_auc"] = float(roc_auc_score(yb, sb))
        out["capacity_retained"] = (out["clip_const_auc"] / out["pooled_auc"]
                                    if out["pooled_auc"] else float("nan"))
    else:
        out["clip_const_auc"] = float("nan")
        out["capacity_retained"] = float("nan")
    return out


def within_video_from_clips(p1: np.ndarray, p2: np.ndarray,
                            gt: np.ndarray, lengths: List[int]):
    """Within-video AUC using true video boundaries from the lengths list.

    Index bookkeeping (this is the subtle part):
      * ``p``  is indexed by CLIP  -> offset advances by n
      * ``gt`` is indexed by FRAME -> video k starts at frame off*n*16
    Keeping one offset in clip units and deriving the frame offset keeps the
    two granularities from silently desynchronising.
    """
    res = {}
    for name, p in (("head1", p1), ("head2", p2)):
        aucs, wsum, wtot, pairs, npos_tot, nneg_tot = [], 0.0, 0.0, 0, 0.0, 0.0
        off = 0                                   # offset in CLIPS
        for n in lengths:
            n = int(n)
            if n <= 0:
                continue
            if off + n > len(p):
                break
            f0 = off * CLIP_REPEAT                # offset in FRAMES
            f1 = (off + n) * CLIP_REPEAT
            y = (gt[f0:f1:CLIP_REPEAT] > 0.5).astype(int)
            s = np.asarray(p[off:off + n], dtype=np.float64)
            if len(y) != len(s):
                raise ValueError(
                    f"clip/GT mismatch at clip offset {off}: "
                    f"gt={len(y)} scores={len(s)}")
            npos_tot += float(y.sum())
            nneg_tot += float(len(y) - y.sum())
            if y.sum() == 0 or y.sum() == len(y):
                off += n
                continue
            a = float(roc_auc_score(y, s))
            aucs.append(a)
            w = float(y.sum()) * float(len(y) - y.sum())
            wsum += a * w
            wtot += w
            pairs += int(y.sum() * (len(y) - y.sum()))
            off += n
        res[name] = {
            "within_auc": (wsum / wtot) if wtot else float("nan"),
            "macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "within_pair_weight": (pairs / (npos_tot * nneg_tot)
                                   if npos_tot > 0 and nneg_tot > 0
                                   else float("nan")),
            "n_videos_scored": len(aucs),
        }
    return res


def spearman(a: List[float], b: List[float]) -> float:
    from scipy.stats import spearmanr
    ok = [i for i in range(len(a)) if np.isfinite(a[i]) and np.isfinite(b[i])]
    if len(ok) < 3:
        return float("nan")
    return float(spearmanr([a[i] for i in ok], [b[i] for i in ok]).statistic)


def ranks(vals: List[float]) -> List[float]:
    """1 = best (highest value). NaNs last."""
    order = sorted(range(len(vals)), key=lambda i: (
        -vals[i] if np.isfinite(vals[i]) else float("inf")))
    r = [0.0] * len(vals)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True,
                    help="root dir containing <TAG>/cdv_scores_<src>2<tgt>.npz")
    ap.add_argument("--lengths", required=True,
                    help="JSON map TAG -> per-video clip lengths")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.lengths) as f:
        lengths_map = json.load(f)

    rows = []
    for npz in sorted(glob.glob(os.path.join(args.scores, "*", "*.npz"))):
        tag = os.path.basename(os.path.dirname(npz))
        if tag not in lengths_map:
            print(f"[skip] no lengths for {tag}")
            continue
        z = np.load(npz)
        p1, p2, gt = z["prob1"], z["prob2"], z["gt"]
        lengths = lengths_map[tag]

        f1 = expand_to_frames(p1, gt)
        f2 = expand_to_frames(p2, gt)
        row = {"tag": tag, "n_clips": int(len(p1)),
               "head1": decompose_frame(f1, gt),
               "head2": decompose_frame(f2, gt)}
        wv = within_video_from_clips(p1, p2, gt, lengths)
        row["head1"].update(wv["head1"])
        row["head2"].update(wv["head2"])
        rows.append(row)
        print(f"[ok] {tag:18s} pooled2={row['head2']['pooled_auc']:.4f} "
              f"within2={row['head2']['within_auc']:.4f} "
              f"cap={row['head2'].get('capacity_retained', float('nan')):.3f}")

    # ---------------- ranking comparison ----------------
    summary = {"configs": rows}
    for head in ("head1", "head2"):
        pooled = [r[head]["pooled_auc"] for r in rows]
        within = [r[head]["within_auc"] for r in rows]
        rp, rw = ranks(pooled), ranks(within)
        flips = [(rows[i]["tag"], rp[i], rw[i]) for i in range(len(rows))
                 if rp[i] != rw[i]]
        summary[f"ranking_{head}"] = {
            "tags": [r["tag"] for r in rows],
            "pooled": pooled,
            "within": within,
            "rank_pooled": rp,
            "rank_within": rw,
            "spearman": spearman(pooled, within),
            "n_flips": len(flips),
            "flips": flips,
        }
        print(f"\n=== {head} ranking ===")
        print(f"  spearman(pooled, within) = {summary[f'ranking_{head}']['spearman']:.4f}")
        print(f"  rank flips: {len(flips)}")
        for t, a, b in flips:
            print(f"    {t:18s} pooled#{a:.0f} -> within#{b:.0f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
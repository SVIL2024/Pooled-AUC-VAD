"""P-3 FINAL: same-target ranking comparison + video-level capacity control.

Two corrections to the first P-3 pass, both forced by the audit:

1. The sibling project's WV_AUC (0.7349) differs from ours (0.8244) even though
   our POOLED numbers match theirs bit-for-bit.  Frame-vs-clip granularity
   explains only 0.0006 of the 0.09 gap (see p3_granularity.log), so the two
   use different definitions of "within-video".  We therefore report our own
   definition explicitly and do NOT compare across projects.

2. Only configurations evaluated on the SAME target are comparable at all.
   Pooled AUC measured on UCF and on XD are different quantities; ranking
   across targets is exactly the error this project is auditing.  The flip
   analysis is therefore restricted to within-target groups.

Capacity control is recomputed at the VIDEO level (each video collapsed to its
mean score), which is the quantity that matters: it is what the pooled metric
would award to a scorer with no temporal resolution whatsoever.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np
from sklearn.metrics import roc_auc_score

R = 16


def analyse(p, gt, lengths):
    """All quantities for one (config, head), from one pair of arrays."""
    per_video_pool, per_video_const = [], []
    within_num = within_den = 0.0
    npos_tot = nneg_tot = 0.0
    off = 0
    for n in lengths:
        n = int(n)
        if n <= 0 or off + n > len(p):
            break
        f0, f1 = off * R, (off + n) * R
        y_clip = (gt[f0:f1:R] > 0.5).astype(int)
        s = np.asarray(p[off:off + n], dtype=np.float64)
        npos_tot += float(y_clip.sum())
        nneg_tot += float(len(y_clip) - y_clip.sum())
        if 0 < y_clip.sum() < len(y_clip):
            a = float(roc_auc_score(y_clip, s))
            w = float(y_clip.sum()) * float(len(y_clip) - y_clip.sum())
            within_num += a * w
            within_den += w
        # video-level entries for the capacity control
        y_vid = 1.0 if gt[f0:f1].max() > 0.5 else 0.0
        per_video_pool.append(float(s.mean()))
        per_video_const.append(y_vid)
        off += n

    n = min(len(p) * R, len(gt))
    s_fr = np.repeat(np.asarray(p, dtype=np.float64), R)[:n]
    y_fr = (gt[:n] > 0.5).astype(int)
    pooled = float(roc_auc_score(y_fr, s_fr))

    yv = np.asarray(per_video_const)
    sv = np.asarray(per_video_pool)
    cap_auc = float(roc_auc_score(yv, sv)) if 0 < yv.sum() < len(yv) else float("nan")

    return {
        "pooled_auc": pooled,
        "within_auc": (within_num / within_den) if within_den else float("nan"),
        "video_const_auc": cap_auc,
        "capacity_retained": (cap_auc / pooled) if pooled else float("nan"),
        "within_pair_weight": ((within_den) / (npos_tot * nneg_tot)
                               if npos_tot > 0 and nneg_tot > 0 else float("nan")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.lengths) as f:
        tlen = json.load(f)

    rows = []
    for npz in sorted(glob.glob(os.path.join(args.scores, "*", "*.npz"))):
        tag = os.path.basename(os.path.dirname(npz))
        tgt = tag.split("_", 1)[1].split("2")[-1]
        z = np.load(npz)
        row = {"tag": tag, "target": tgt,
               "head1": analyse(z["prob1"], z["gt"], tlen[tag]),
               "head2": analyse(z["prob2"], z["gt"], tlen[tag])}
        rows.append(row)

    out = {"configs": rows, "flips": {}}

    print("=" * 88)
    print(f"{'config':<20}{'tgt':<5}{'pooled':>9}{'within':>9}{'videoConst':>12}{'retained':>10}{'w':>10}")
    print("-" * 88)
    for r in rows:
        h = r["head2"]
        print(f"{r['tag']:<20}{r['target']:<5}{h['pooled_auc']:>9.4f}"
              f"{h['within_auc']:>9.4f}{h['video_const_auc']:>12.4f}"
              f"{h['capacity_retained']:>10.3f}{h['within_pair_weight']*100:>9.3f}%")

    # ---- within-target pairwise reversals ----
    for head in ("head1", "head2"):
        flips = []
        for a, b in combinations(rows, 2):
            if a["target"] != b["target"]:
                continue
            pa = a[head]["pooled_auc"] - b[head]["pooled_auc"]
            wa = a[head]["within_auc"] - b[head]["within_auc"]
            if np.isfinite(pa) and np.isfinite(wa) and np.sign(pa) != np.sign(wa):
                winner_pool = a["tag"] if pa > 0 else b["tag"]
                winner_within = a["tag"] if wa > 0 else b["tag"]
                flips.append({
                    "pair": [a["tag"], b["tag"]], "target": a["target"],
                    "pooled_winner": winner_pool, "within_winner": winner_within,
                    "pooled_margin": abs(pa), "within_margin": abs(wa),
                })
        out["flips"][head] = flips
        print(f"\n=== {head}: same-target rank reversals ===")
        if not flips:
            print("  none")
        for f in flips:
            print(f"  {f['target']}: {f['pooled_winner']} wins on pooled "
                  f"(+{f['pooled_margin']:.4f})  but  {f['within_winner']} wins "
                  f"on within (+{f['within_margin']:.4f})")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
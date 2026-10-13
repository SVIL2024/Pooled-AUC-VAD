"""s6_sufficiency: can the per-video offset ALONE produce the headline number?

The weak link this closes
-------------------------
The corrected claim is "pooled AUC is no better than its per-video offset
everywhere; what varies across domains is whether that offset is aligned with
the video label -- aligned gives 0.85-0.95, misaligned gives 0.40".  That rested
on two domains, one aligned (UCF/XD) and one misaligned (SHT).  Two points is a
correlation, and a reviewer would rightly press on it.

Why permutation does NOT fix this
---------------------------------
The obvious intervention -- permute video scores across videos -- has no clean
invariant.  After permuting, video i holds s_pi(i) with labels y_i, so its
within-video score-to-label relationship is destroyed too; AUC_within moves as
well as AUC_cross.  Nothing separates the two effects.

What does: a SUFFICIENCY construction
--------------------------------------
Start from the de-offsetted scores, whose within-video quality is fixed BY
CONSTRUCTION (de-offset cannot change within-video ranking; verified exactly in
s3).  Then add a per-video constant and nothing else:

    aligned     +delta to every frame of anomalous videos, 0 to normal
    anti-aligned -delta to anomalous, 0 to normal
    random      +/-delta drawn independently per video

If the aligned offset alone drives pooled AUC from the de-offsetted level back to
the reported 0.85-0.95, then the offset is SUFFICIENT for the headline number --
the within-video signal is not what produces it.  And if adding an ALIGNED offset
to the ShanghaiTech scores lifts them from 0.45 to comparable levels, then the
SHT collapse is misalignment of the offset, not a missing capability.  That is
the constructive, within-domain version of the claim, and it is tested on every
domain including SHT.

No invariant is needed for this design: the within-video score structure is
identical across the three arms by construction, so any difference in pooled AUC
is attributable to the offset and nothing else.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

REPEAT = 16
DELTAS = (0.25, 0.5, 1.0, 2.0, 4.0)


def load(path, lens_npz=None):
    z = np.load(path, allow_pickle=True)
    lens = (np.asarray(z["lens"], dtype=np.int64) if "lens" in z.files else
            np.asarray(np.load(lens_npz, allow_pickle=True)["lens"],
                      dtype=np.int64))
    return (np.asarray(z["prob2"], float), np.asarray(z["gt"]), lens)


def clip_labels(gt, n):
    return np.array([(gt[i * REPEAT:(i + 1) * REPEAT].mean() > 0.5)
                     for i in range(n)], dtype=bool)


def deoffset(s, lens):
    offs = np.concatenate([[0], np.cumsum(lens)])
    out = np.array(s, copy=True)
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        out[lo:hi] = out[lo:hi] - out[lo:hi].mean()
    return out


def add_offset(base, lens, vfrac, delta, mode, rng):
    offs = np.concatenate([[0], np.cumsum(lens)])
    out = np.array(base, copy=True)
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        if mode == "aligned":
            d = delta if vfrac[i] > 0 else 0.0
        elif mode == "anti":
            d = -delta if vfrac[i] > 0 else 0.0
        else:                                   # random, sign not label-linked
            d = float(rng.uniform(-delta, delta))
        out[lo:hi] = out[lo:hi] + d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--sht", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    LENS_FILE = {"xd": "classprobs_ucf2xd.npz", "ucf": "classprobs_xd2ucf.npz"}
    jobs = []
    for cfg, tgt in (("A_ucf2xd", "xd"), ("C_xd2xd", "xd"), ("B_xd2ucf", "ucf"),
                     ("D_ucf2ucf", "ucf")):
        f = glob.glob(os.path.join(args.p3, cfg, "cdv_scores_*.npz"))
        if f:
            jobs.append((cfg, f[0], os.path.join(args.lens, LENS_FILE[tgt])))
    for src in ("ucf", "xd"):
        f = os.path.join(args.sht, f"cdv_scores_{src}2sht.npz")
        if os.path.exists(f):
            jobs.append((f"{src}2sht", f, None))

    rows = {}
    for tag, path, lf in jobs:
        s, gt, lens = load(path, lf)
        y = clip_labels(gt, len(s))
        offs = np.concatenate([[0], np.cumsum(lens)])
        vfrac = np.array([y[int(offs[i]):int(offs[i + 1])].mean()
                          for i in range(len(lens))])
        base = deoffset(s, lens)
        auc_orig = float(roc_auc_score(y, s))
        auc_base = float(roc_auc_score(y, base))

        print("=" * 100)
        print(f"{tag}   original AUC {auc_orig:.4f}   de-offsetted AUC "
              f"{auc_base:.4f}")
        print("=" * 100)
        print(f"  {'delta':>7}{'aligned':>12}{'anti-aligned':>14}"
              f"{'random':>10}")
        print("  " + "-" * 43)
        rec = {"orig": auc_orig, "deoff": auc_base, "arms": {}}
        for d in DELTAS:
            rng = np.random.RandomState(args.seed)
            a_al = float(roc_auc_score(
                y, add_offset(base, lens, vfrac, d, "aligned", rng)))
            a_an = float(roc_auc_score(
                y, add_offset(base, lens, vfrac, d, "anti", rng)))
            rng = np.random.RandomState(args.seed)
            a_rd = float(roc_auc_score(
                y, add_offset(base, lens, vfrac, d, "rand", rng)))
            rec["arms"][str(d)] = {"aligned": a_al, "anti": a_an,
                                   "random": a_rd}
            print(f"  {d:>7.2f}{a_al:>12.4f}{a_an:>14.4f}{a_rd:>10.4f}")
        rows[tag] = rec
        print()

    print("=" * 100)
    print("SUFFICIENCY -- aligned offset alone, from the de-offsetted baseline")
    print("=" * 100)
    print(f"  {'config':<14}{'de-off AUC':>12}{'+aligned d=1':>14}"
          f"{'recovered%':>13}{'original':>10}")
    print("  " + "-" * 63)
    for tag, r in rows.items():
        d1 = r["arms"]["1.0"]["aligned"]
        rec_pct = 100 * (d1 - r["deoff"]) / max(r["orig"] - r["deoff"], 1e-9)
        print(f"  {tag:<14}{r['deoff']:>12.4f}{d1:>14.4f}"
              f"{rec_pct:>12.1f}%{r['orig']:>10.4f}")
    print()
    print("  If recovery is near 100% on every domain -- including SHT -- the")
    print("  per-video offset is SUFFICIENT to produce the reported pooled AUC,")
    print("  and the domain-to-domain difference is alignment, not capability.")

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
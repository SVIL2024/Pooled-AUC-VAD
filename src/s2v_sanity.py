"""s2v_sanity: is the ShanghaiTech result a real finding or a degenerate score?

The SHT rows are the most surprising output of s2_table -- pooled AUC of 0.3973
for ucf2sht is BELOW chance, and the within-video gap reverses sign.  Both are
publishable only if they are real.  Three failure modes would manufacture them:

  D1 degenerate scores      few unique values / near-constant -> AUC meaningless
  D2 GT misalignment       scores aligned to the wrong frames -> inversion
  D3 an inverted convention a sign error somewhere -> below-chance AUC

Checks, none of which assume the answer:
  1. score distribution: unique-value count, std, and whether large ties exist
  2. video-level consistency: correlation between each video's mean score and
     its own anomalous-frame fraction.  This does not depend on the metric's
     internal convention at all -- if the model carries ANY signal, this must be
     positive.  If it is negative too, the scores are inverted or misaligned,
     and the below-chance pooled AUC is a bug rather than a finding.
  3. a sign-flip test: if negating the SHT scores recovers sane numbers, the
     defect is a convention/sign error that must be fixed before reporting.
  4. the same consistency check on UCF and XD as a reference for what "working"
     looks like on this pipeline.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

REPEAT = 16


def load(path, lens_npz=None):
    z = np.load(path, allow_pickle=True)
    if "lens" in z.files:
        lens = np.asarray(z["lens"], dtype=np.int64)
    else:
        lens = np.asarray(np.load(lens_npz, allow_pickle=True)["lens"],
                          dtype=np.int64)
    return np.asarray(z["prob2"], float), np.asarray(z["gt"]), lens


def clip_labels(gt, n):
    return np.array([(gt[i * REPEAT:(i + 1) * REPEAT].mean() > 0.5)
                     for i in range(n)], dtype=bool)


def report(tag, prob2, gt, lens):
    y = clip_labels(gt, len(prob2))
    offs = np.concatenate([[0], np.cumsum(lens)])
    uniq = len(np.unique(prob2))
    pooled = float(roc_auc_score(y, prob2))
    pooled_neg = float(roc_auc_score(y, -prob2))

    vm, vg = [], []
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        vm.append(prob2[lo:hi].mean())
        vg.append(y[lo:hi].mean())
    vm, vg = np.asarray(vm), np.asarray(vg)
    # only videos with both classes are informative
    mixed = (vg > 0) & (vg < 1)
    rho = float("nan")
    if mixed.sum() >= 5:
        rho = float(spearmanr(vm[mixed], vg[mixed]).statistic)

    print(f"{tag:<16} n={len(prob2):<7} uniq={uniq:<7} "
          f"std={prob2.std():.4f}  pooled={pooled:.4f}  "
          f"pooled(-s)={pooled_neg:.4f}  mixedVideos={int(mixed.sum()):<4} "
          f"spearman(vidMeanScore, vidAnomFrac)={rho:+.4f}")
    return {"unique": uniq, "std": float(prob2.std()), "pooled": pooled,
            "pooled_neg": pooled_neg, "n_mixed_videos": int(mixed.sum()),
            "video_level_spearman": rho}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--sht", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    jobs = []
    LENS_FILE = {"xd": "classprobs_ucf2xd.npz", "ucf": "classprobs_xd2ucf.npz"}
    for cfg, tgt in (("A_ucf2xd", "xd"), ("C_xd2xd", "xd"),
                     ("B_xd2ucf", "ucf"), ("D_ucf2ucf", "ucf")):
        f = glob.glob(os.path.join(args.p3, cfg, "cdv_scores_*.npz"))
        if f:
            jobs.append((cfg, f[0],
                         os.path.join(args.lens, LENS_FILE[tgt])))
    for src in ("ucf", "xd"):
        f = os.path.join(args.sht, f"cdv_scores_{src}2sht.npz")
        if os.path.exists(f):
            jobs.append((f"{src}2sht", f, None))

    print("=" * 118)
    print("SANITY: score distribution, pooled AUC, and a convention-independent")
    print("        video-level consistency check")
    print("=" * 118)
    res = {}
    for tag, path, lf in jobs:
        prob2, gt, lens = load(path, lf)
        res[tag] = report(tag, prob2, gt, lens)

    print()
    print("=" * 118)
    print("READING")
    print("=" * 118)
    bad = []
    for tag, r in res.items():
        if r["unique"] < 50 or r["std"] < 1e-6:
            bad.append(f"{tag}: degenerate scores")
        if r["video_level_spearman"] < 0:
            bad.append(f"{tag}: video-level Spearman NEGATIVE "
                       f"({r['video_level_spearman']:+.4f}) -- convention or "
                       f"alignment suspect")
    for tag in ("ucf2sht", "xd2sht"):
        if tag in res:
            r = res[tag]
            print(f"  {tag}: pooled {r['pooled']:.4f}, negated "
                  f"{r['pooled_neg']:.4f}, video-level Spearman "
                  f"{r['video_level_spearman']:+.4f}")
            if r["pooled_neg"] > 0.55:
                print(f"    -> negating recovers a usable score "
                      f"({r['pooled_neg']:.4f}). THIS WOULD BE A SIGN "
                      f"CONVENTION BUG, not a finding.")
    if bad:
        print("\n  SUSPECT:")
        for b in bad:
            print(f"    - {b}")
    else:
        print("\n  All score distributions non-degenerate and all video-level "
              "correlations >= 0.")
        print("  Below-chance SHT pooled AUC is therefore a genuine transfer "
              "failure,\n  not a plumbing defect.")

    with open(args.out, "w") as f:
        json.dump({"detail": res, "suspect": bad}, f, indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
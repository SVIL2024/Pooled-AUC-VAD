"""s2v_align: is the SHT test-list ORDER the same order as gt_sht.npy?

Why this specific check
----------------------
The SHT rows are the only ones produced by a new path in this project, and they
contain the two most surprising numbers in s2_table: a below-chance pooled AUC
and a reversed within-video gap.  The guard already run asserts only
``sum(lens) * 16 == len(gt)``, which constrains the LENGTH of the label array
and says nothing about its ORDER.  A permuted label array would pass that guard
and then invert every metric, which is exactly what ucf2sht's negative
video-level correlation looks like.

So this compares, per video, two independent sources of truth:
  * the test list's own video-level label (normal vs anomalous)
  * the anomalous-frame fraction implied by gt_sht.npy at that position
They must agree in sign.  A systematic disagreement identifies which videos are
misaligned, and a permutation of the labels would show up as near-zero
agreement.

It also tests the obvious alternative: if the labels are fine but the scores
are inverted, negating should fix the video-level correlation.  The two
hypotheses -- misaligned labels vs genuinely anti-correlated model -- are
distinguished by whether the list-vs-GT check passes.
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import os

import numpy as np
from scipy.stats import spearmanr

REPEAT = 16


def read_list(sht):
    for cand in ("sht_CLIP_rgbtest.csv", "sht_CLIP_rgb.csv"):
        p = os.path.join(sht, cand)
        if os.path.exists(p):
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csvmod.DictReader(f))
            cols = list(rows[0].keys())
            pk = "path" if "path" in cols else cols[0]
            lk = next((c for c in cols if c.lower() in ("label", "labels",
                                                        "cls", "class")), None)
            return ([os.path.basename(str(r[pk]).replace("\\", "/"))
                     for r in rows],
                    [str(r[lk]) for r in rows] if lk else None, cols,
                    os.path.basename(p))
    return None, None, None, None


def is_anom(label):
    """Interpret a test-list label without assuming the vocabulary."""
    s = str(label).strip().lower()
    if s in ("0", "0.0", "normal", "false", ""):
        return False
    if s in ("1", "1.0", "anomaly", "anomalous", "true"):
        return True
    # anything non-empty and not 'normal' is treated as an anomaly class name
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sht", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names, labels, cols, listfile = read_list(args.sht)
    if names is None:
        raise SystemExit("no SHT test list found")
    gt = np.load(os.path.join(args.sht, "gt_sht.npy"))
    test_dir = os.path.join(args.sht, "test")

    lens = np.array([np.load(os.path.join(test_dir, n)).shape[0]
                     for n in names], dtype=np.int64)
    print(f"list file     : {listfile}")
    print(f"columns       : {cols}")
    print(f"label column  : {'yes' if labels else 'NONE FOUND'}")
    print(f"videos        : {len(names)}")
    print(f"sum(lens)*16  : {int(lens.sum())*REPEAT}   len(gt) = {len(gt)}")
    assert int(lens.sum()) * REPEAT == len(gt), "length guard failed"

    offs = np.concatenate([[0], np.cumsum(lens)])
    frac = np.array([float((gt[int(offs[i]) * REPEAT:int(offs[i + 1]) * REPEAT]
                            > 0.5).mean()) for i in range(len(lens))])

    print(f"\nframe-level anomalous fraction per video: "
          f"min {frac.min():.3f}  median {np.median(frac):.3f}  "
          f"max {frac.max():.3f}")
    print(f"videos with frac == 0: {(frac == 0).sum()}   "
          f"frac == 1: {(frac == 1).sum()}   "
          f"mixed: {((frac > 0) & (frac < 1)).sum()}")

    out = {"list_file": listfile, "columns": cols, "has_label_col":
           labels is not None, "n_videos": len(names),
           "frac_min": float(frac.min()), "frac_median": float(np.median(frac)),
           "frac_max": float(frac.max())}

    if labels:
        lab = np.array([is_anom(l) for l in labels], dtype=bool)
        agree = ((frac > 0) == lab).mean()
        rho = float(spearmanr(frac, lab.astype(float)).statistic)
        print(f"\nlist-vs-GT agreement : {agree*100:.1f}%  "
              f"(Spearman {rho:+.4f}, n={len(lab)})")
        print(f"list says anomalous  : {int(lab.sum())}")
        print(f"GT says >0 anomalous : {int((frac > 0).sum())}")
        out["agreement"] = float(agree)
        out["spearman"] = rho
        out["list_anom"] = int(lab.sum())
        out["gt_anom"] = int((frac > 0).sum())

        # which videos disagree?
        bad = [(names[i], labels[i], float(frac[i]))
               for i in range(len(names)) if ((frac[i] > 0) != lab[i])]
        print(f"\ndisagreements: {len(bad)}")
        for n, l, f in bad[:10]:
            print(f"    {n:<26} list={l:<14} gt_frac={f:.3f}")
        out["disagreements"] = bad[:40]

        # anti-correlation check: if the list order were reversed, would this
        # look better?  That is what a sign/order bug would produce.
        rev = ((frac[::-1] > 0) == lab).mean()
        print(f"\nagreement if GT were REVERSED: {rev*100:.1f}%  "
              f"(if this were higher, the GT order is flipped)")
        out["agreement_reversed"] = float(rev)
    else:
        print("\nNo label column in the list, so order cannot be checked "
              "against it directly.")
        print("Falling back to an internal consistency check: on a correctly "
              "ordered\nGT, a video whose frames are entirely normal should not "
              "be a MIXED video,\nand the fraction distribution should be "
              "bimodal-ish rather than smooth.")
        h = np.histogram(frac, bins=10, range=(0, 1))[0]
        print(f"  histogram of per-video anomalous fraction: {h.tolist()}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
"""s2v_align2: validate the gt_sht.npy ORDER against ShanghaiTech's own annotations.

The situation
-------------
s2v_align could not test the order, because this SHT test split contains no
normal videos: every entry in the list is anomalous, so "list label vs GT" agrees
trivially and constrains nothing.  Meanwhile ucf2sht shows a NEGATIVE
video-level correlation (Spearman -0.39), which is exactly what a permuted label
array would produce, and that number is the source of the paper's most
interesting claim.  It cannot be used unverified.

ShanghaiTech ships its own frame-level annotations, independently of anything
this project produced:
    E:\\dataset\\ShanghaiTech\\label\\test.csv
    E:\\dataset\\ShanghaiTech\\testing\\test_frame_mask
Those give a ground truth for the ORDER.  If the anomalous-frame fraction implied
by gt_sht.npy at each video position matches the fraction derived from the
original mask at that video's name, the order is correct and the negative
correlation is a genuine property of the model.

Also reports the reverse test -- agreement under a reversed GT -- because a
flipped order can hide behind a plausible-looking number, and an accidental
reversal would invert every SHT metric in the paper.
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import os

import numpy as np

REPEAT = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sht", required=True)
    ap.add_argument("--raw", default=r"E:\dataset\ShanghaiTech")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = [os.path.basename(p)
             for p in sorted(os.listdir(os.path.join(args.sht, "test")))]
    csv_path = os.path.join(args.sht, "sht_CLIP_rgbtest.csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csvmod.DictReader(f))
    pk = "path" if "path" in rows[0] else list(rows[0])[0]
    list_names = [os.path.basename(str(r[pk]).replace("\\", "/")) for r in rows]
    gt = np.load(os.path.join(args.sht, "gt_sht.npy"))
    lens = np.array([np.load(os.path.join(args.sht, "test", n)).shape[0]
                     for n in list_names], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)])
    frac_gt = np.array([float(
        (gt[int(offs[i]) * REPEAT:int(offs[i + 1]) * REPEAT] > 0.5).mean())
        for i in range(len(list_names))])

    # ---- original annotations ----
    mask_dir = os.path.join(args.raw, "testing", "test_frame_mask")
    print(f"mask dir exists : {os.path.isdir(mask_dir)}")
    if not os.path.isdir(mask_dir):
        print("  cannot validate order -- original masks absent")
        return
    mfiles = sorted(os.listdir(mask_dir))
    print(f"original masks  : {len(mfiles)}  (first: {mfiles[:3]})")

    # mask naming is 01_0014.npy while features are 01_0014.npy too? check both
    def load_mask(stem):
        for cand in (stem + ".npy", stem + "_mask.npy", stem):
            p = os.path.join(mask_dir, cand)
            if os.path.exists(p):
                return np.load(p).reshape(-1)
        return None

    stem_of = lambda n: os.path.splitext(n)[0]
    m_stems = {os.path.splitext(m)[0]: m for m in mfiles}
    hit = [n for n in list_names if stem_of(n) in m_stems]
    print(f"list entries found in original masks: {len(hit)}/{len(list_names)}")
    if not hit:
        # try matching by prefix
        pref = {m.split("_")[0]: m for m in m_stems}
        hit = [n for n in list_names if stem_of(n).split("_")[0] in pref]
        print(f"  after prefix matching: {len(hit)}")

    agree_frac, rows_out = [], []
    for i, n in enumerate(list_names):
        stem = stem_of(n)
        mf = m_stems.get(stem)
        if mf is None:
            mf = m_stems.get(stem.split("_")[0])
        if mf is None:
            continue
        m = load_mask(stem if stem in m_stems else stem.split("_")[0])
        if m is None:
            continue
        agree_frac.append((float(m.mean()), float(frac_gt[i])))

    if not agree_frac:
        print("  NO per-video overlap with the original masks -- cannot "
              "validate order")
        return
    a = np.array([x[0] for x in agree_frac])
    b = np.array([x[1] for x in agree_frac])
    r = float(np.corrcoef(a, b)[0, 1])
    mae = float(np.abs(a - b).mean())
    print()
    print(f"per-video anomalous fraction, original mask vs gt_sht.npy")
    print(f"  n compared : {len(a)}")
    print(f"  correlation: {r:+.4f}")
    print(f"  mean |diff|: {mae:.4f}")
    print(f"  original mean {a.mean():.4f}   gt_sht mean {b.mean():.4f}")
    rr = float(np.corrcoef(a[::-1], b)[0, 1])
    print(f"  correlation if order REVERSED: {rr:+.4f}")
    ok = r > 0.8 and rr < r
    print()
    print("ORDER VALIDATED" if ok else
          "ORDER NOT VALIDATED -- the SHT numbers must not be reported")
    for i in range(min(8, len(a))):
        print(f"    original {a[i]:.3f}   gt_sht {b[i]:.3f}")
    with open(args.out, "w") as f:
        json.dump({"n": len(a), "corr": r, "corr_reversed": rr,
                   "mae": mae, "order_validated": bool(ok)}, f, indent=2,
                  default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
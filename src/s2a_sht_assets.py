"""s2a_sht_assets: can SHT produce a (scores, gt, lens) triple consistent with
the existing metric machinery?

Why this gate exists
--------------------
The metrics in this project were verified bit-exactly against the archived UCF
and XD numbers, and they assume a specific clip grid:
    scores : one float per clip, in test-list order
    gt     : one label per FRAME (16 per clip), same order
    lens   : clips per video, summing to len(scores)
ShanghaiTech ships its own test list and its own GT npy, and the project archive
records a protocol difference for SHT (13.7-13).  If SHT's clips are not 16
frames each, or if its GT is already clip-level rather than frame-level, then
pooling SHT into the same analysis would silently change the metric definition
-- the exact defect class this project has been auditing all along.

So nothing is assumed.  This reports the actual shapes, dtypes, value ranges,
and the implied alignment, and states whether SHT can be added under the SAME
definition as UCF/XD or needs a separate path.
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sht", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    S = args.sht
    res = {}
    ok = True

    print("=" * 88)
    print("A. directory")
    print("=" * 88)
    for sub in sorted(os.listdir(S)):
        p = os.path.join(S, sub)
        if os.path.isdir(p):
            print(f"  DIR  {sub:<22} {len(os.listdir(p))} files")
        else:
            print(f"  FILE {sub:<22} {os.path.getsize(p)/1e3:.1f} KB")
    test_dir = os.path.join(S, "test")
    train_dir = os.path.join(S, "train")
    test_files = sorted(os.listdir(test_dir)) if os.path.isdir(test_dir) else []
    print(f"\n  test clips: {len(test_files)}  first: {test_files[:3]}")

    print()
    print("=" * 88)
    print("B. test list")
    print("=" * 88)
    lst = None
    for cand in ("sht_CLIP_rgbtest.csv", "sht_CLIP_rgb.csv"):
        p = os.path.join(S, cand)
        if os.path.exists(p):
            lst = p
            break
    if lst is None:
        print("  NO TEST LIST FOUND")
        ok = False
    else:
        with open(lst, newline="", encoding="utf-8") as f:
            rows = list(csvmod.DictReader(f))
        print(f"  {os.path.basename(lst)}: {len(rows)} rows")
        print(f"  columns: {list(rows[0].keys())}")
        for r in rows[:4]:
            print(f"      {dict(r)}")
        lens = [int(float(r.get("_n", r.get("n", 0)) or 0)) for r in rows]
        print(f"  _n / n column: present={'_n' in rows[0] or 'n' in rows[0]}"
              f"  sum={sum(lens)}  nonzero={sum(1 for v in lens if v>0)}")
        res["list"] = {"path": os.path.basename(lst), "rows": len(rows),
                       "columns": list(rows[0].keys()),
                       "n_col_sum": int(sum(lens))}
        if sum(lens) == 0:
            print("  NOTE: no clip-count column; lengths must come from the "
                  "feature files")

    print()
    print("=" * 88)
    print("C. feature files -- shape and clip length")
    print("=" * 88)
    shapes = []
    for fn in test_files[:6]:
        a = np.load(os.path.join(test_dir, fn))
        shapes.append(a.shape)
        print(f"  {fn:<26} {a.shape} {a.dtype}  "
              f"finite={bool(np.isfinite(a).all())}")
    lens = np.array([s[0] for s in shapes], dtype=np.int64)
    print(f"\n  clip lengths (first 6): {lens.tolist()}")
    print(f"  dim consistency: {sorted({s[1] for s in shapes})}")

    print()
    print("=" * 88)
    print("D. GT")
    print("=" * 88)
    gtp = os.path.join(S, "gt_sht.npy")
    if os.path.exists(gtp):
        gt = np.load(gtp)
        print(f"  gt_sht.npy  shape={gt.shape} dtype={gt.dtype}")
        u, c = np.unique(gt, return_counts=True)
        print(f"  values={u.tolist()}  counts={c.tolist()}")
        print(f"  positive rate (elementwise) = {float((gt > 0.5).mean()):.4f}")

        # the decisive alignment question
        n_test_clips = sum(int(np.load(os.path.join(test_dir, f)).shape[0])
                           for f in test_files)
        print(f"\n  ALIGNMENT")
        print(f"    total test clips from feature files : {n_test_clips}")
        print(f"    gt length                          : {gt.shape[0]}")
        ratio = gt.shape[0] / max(n_test_clips, 1)
        print(f"    ratio gt/clips                    : {ratio:.4f}")
        if abs(ratio - 1.0) < 1e-6:
            print("    -> GT is CLIP-LEVEL (1 label per clip).  "
                  "Metric code that assumes\n       16 labels per clip must "
                  "be adapted, or GT expanded.")
            ok = ok and False
        elif abs(ratio - 16.0) < 1e-6:
            print("    -> GT is FRAME-LEVEL at 16 frames per clip, matching "
                  "UCF/XD.")
        else:
            print(f"    -> UNRECOGNISED ratio.  Cannot assume the UCF/XD "
                  "definition.")
            ok = False
        res["gt"] = {"shape": list(gt.shape), "dtype": str(gt.dtype),
                     "values": u.tolist(),
                     "positive_rate": float((gt > 0.5).mean()),
                     "ratio_to_clips": ratio}
    else:
        print("  NO gt_sht.npy")
        ok = False

    print()
    print("=" * 88)
    print(f"SHT GATE: {'PASS -- same metric definition as UCF/XD' if ok else 'NEEDS A SEPARATE PATH'}")
    print("=" * 88)
    res["_gate"] = ok
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
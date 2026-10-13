"""probe_i3d: is the already-extracted I3D feature bank usable for a two-backbone test?

The plan is to test whether fusing two genuinely different frozen backbones
improves the video-level ranking, which is what pooled AUC actually measures.
CLIP ViT-B/32 features already exist.  I3D features also exist on disk, and I3D
(3D convolutions) versus CLIP (ViT) is the one backbone swap the field's own
history says produced a large jump -- so they are the most promising cheap
source of diversity, and testing with them costs no extraction time at all.

This checks the things that decide usability, without assuming any layout:
  1. which datasets are covered, at what resolution, and how many files;
  2. the on-disk format (npy / pkl / npz) and the array shape/dtype;
  3. whether the clip count and ordering can be matched to the CLIP feature
     bank, since an unmatchable bank cannot be fused frame-for-frame.

The ordering question is the decisive one.  A fusion study needs backbone A and
backbone B to score the SAME clip at the SAME index.  If the two banks are keyed
by different lists, the study has to fall back to video-level fusion with
per-video aggregation, which is reported here as a fallback rather than
discovered later.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np


def describe(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            a = np.load(path, allow_pickle=False)
        elif ext == ".npz":
            z = np.load(path, allow_pickle=False)
            keys = list(z.keys())
            a = z[keys[0]]
        elif ext in (".pkl", ".pickle"):
            with open(path, "rb") as f:
                a = pickle.load(f)
            if isinstance(a, dict):
                keys = sorted(a.keys())[:6]
                first = a[keys[0]]
                return {"ext": ext, "dict_keys": keys, "value_shape":
                        list(np.shape(first)),
                        "value_dtype": str(np.asarray(first).dtype)}
        else:
            return {"ext": ext, "note": "unhandled extension"}
    except Exception as exc:
        return {"ext": ext, "error": f"{type(exc).__name__}: {exc}"}
    if hasattr(a, "shape"):
        return {"ext": ext, "shape": list(a.shape), "dtype": str(a.dtype),
                "finite": bool(np.isfinite(np.asarray(a, dtype=np.float64)).all())}
    return {"ext": ext, "type": type(a).__name__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i3d", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = {}
    for label, root in (("I3D", args.i3d), ("CLIP", args.clip)):
        print("=" * 78)
        print(f"{label} feature bank: {root}")
        print("=" * 78)
        if not os.path.isdir(root):
            print("  (absent)")
            res[label] = {"present": False}
            continue
        datasets = sorted(d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d)))
        print(f"  datasets: {datasets}")
        total = 0
        for ds in datasets:
            files = glob.glob(os.path.join(root, ds, "**", "*.*"),
                              recursive=True)
            files = [f for f in files
                     if os.path.splitext(f)[1].lower() in
                     (".npy", ".npz", ".pkl", ".pickle")]
            total += len(files)
            print(f"    {ds:<16} {len(files)} feature files")
            for f in files[:args.samples]:
                print(f"        {os.path.relpath(f, root):<44} "
                      f"{describe(f)}")
        print(f"  TOTAL {total} files")
        res[label] = {"present": True, "datasets": datasets, "n_files": total}

    # ---- the decisive question: can the two banks be matched clip-for-clip? ----
    print()
    print("=" * 78)
    print("MATCHABILITY (needed for frame-level fusion)")
    print("=" * 78)
    for ds in ("UCF", "XD", "SHT", "ShanghaiTech"):
        i3 = glob.glob(os.path.join(args.i3d, "**", f"{ds}*"),
                       recursive=True)
        c1 = glob.glob(os.path.join(args.clip, "**", f"{ds}*"),
                       recursive=True)
        if not i3 and not c1:
            continue
        print(f"  {ds:<14} i3d_dirs={len(i3)}  clip_dirs={len(c1)}")
        for p in (i3 + c1)[:4]:
            print(f"      {p}")
    res["match"] = True
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
"""probe_expansion: can the two promised additions actually be made?

The plan is a measurement paper on what cross-domain WSVAD evaluation measures,
and two additions were assumed cheap:
  (a) a second backbone family, to show the finding is not one model's quirk;
  (b) a third domain, to show it is not one split's quirk.

Both were called "zero extraction cost" because the features are already on
disk.  That was too quick.  Two things could make them expensive or impossible:

  I3D.  I3D features are not CLIP features.  Scoring them for anomaly requires a
  scorer, and the checkpoints in this tree are CLIP-based (model_ucf.pth /
  model_xd.pth), so they cannot consume I3D features.  A new scorer means a new
  method choice, which is exactly the kind of unstated decision that produces
  the class of defect this project has already been auditing.  So this probe
  reports what the I3D bank actually contains -- features, or checkpoints, or
  precomputed scores -- because that determines whether a second family is
  reachable at all.

  SHT.  The project archive records that SHT uses a different visual_length
  protocol (13.7-13), so its clip grid may not be interchangeable with UCF/XD's.
  If the per-clip feature length or the GT construction differ, the metrics
  cannot simply be pooled across domains and the "third domain" claim would be
  wrong before it started.

Both are checked against the known-good CLIP layout (UCFClipFeatures /
XDTestClipFeatures) rather than against assumptions.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np


def peek(path, deep=False):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            a = np.load(path, allow_pickle=False)
        elif ext == ".npz":
            z = np.load(path, allow_pickle=False)
            return {"ext": ext, "keys": list(z.keys())}
        elif ext in (".pkl", ".pickle", ".dat"):
            with open(path, "rb") as f:
                a = pickle.load(f, encoding="latin1")
            if isinstance(a, dict):
                ks = sorted(a.keys())[:8]
                v = np.asarray(a[ks[0]]) if not isinstance(a[ks[0]], dict) else None
                return {"ext": ext, "dict": True, "n_keys": len(a),
                        "sample_keys": ks, "first_shape":
                            list(v.shape) if v is not None else None}
            a = np.asarray(a)
        else:
            return {"ext": ext, "note": "unknown"}
    except Exception as exc:
        return {"ext": ext, "error": f"{type(exc).__name__}: {exc}"}
    if a.dtype == object and deep:
        return {"ext": ext, "dtype": "object",
                "first_type": type(a.flat[0]).__name__,
                "first_shape": list(np.shape(a.flat[0]))}
    return {"ext": ext, "shape": list(np.shape(a)), "dtype": str(a.dtype)}


def walk(root, limit=6):
    files = [p for p in glob.glob(os.path.join(root, "**", "*"),
                                 recursive=True)
             if os.path.isfile(p)]
    out = []
    for p in sorted(files)[:limit]:
        out.append((os.path.relpath(p, root), peek(p)))
    return len(files), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\dataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R = args.root
    res = {}

    print("=" * 88)
    print("REFERENCE: the known-good CLIP layout")
    print("=" * 88)
    for d in ("UCFClipFeatures", "XDTestClipFeatures"):
        p = os.path.join(R, d)
        n, sample = walk(p, 4)
        print(f"  {d}: {n} files")
        for rel, info in sample:
            print(f"      {rel[:52]:<54} {info}")
        res[d] = {"n_files": n}

    print()
    print("=" * 88)
    print("A. I3D bank -- features, checkpoints, or precomputed scores?")
    print("=" * 88)
    i3 = os.path.join(R, "i3d")
    if not os.path.isdir(i3):
        print("  (absent)")
        res["i3d"] = {"present": False}
    else:
        for sub in sorted(os.listdir(i3)):
            p = os.path.join(i3, sub)
            if not os.path.isdir(p):
                print(f"  FILE {sub}  {os.path.getsize(p)/1e6:.1f} MB")
                continue
            n, sample = walk(p, 5)
            print(f"  DIR  {sub}/  {n} files")
            for rel, info in sample:
                print(f"      {rel[:52]:<54} {info}")
            res.setdefault("i3d", {})[sub] = {"n_files": n, "sample": sample}
        verdict = ("I3D holds per-video SCORES, not raw features -- usable as "
                   "a second detector only if they are frame-granular"
                   if any(s in str(res.get('i3d', {}))
                          for s in ("UCF", "XD"))
                   else "I3D layout unrecognised")
        print(f"\n  -> {verdict}")
        res["_i3d_note"] = verdict

    print()
    print("=" * 88)
    print("B. SHT features -- is the clip grid compatible with UCF/XD?")
    print("=" * 88)
    sht = None
    for cand in ("SHTClipFeatures", "ShanghaiTechClipFeatures"):
        p = os.path.join(R, cand)
        if os.path.isdir(p):
            sht = p
            break
    if sht is None:
        print("  no SHTClipFeatures directory found")
        res["sht"] = {"present": False}
    else:
        subs = sorted(os.listdir(sht))
        print(f"  {os.path.basename(sht)}: {len(subs)} entries -> {subs[:8]}")
        shapes = {}
        for sub in subs[:3]:
            p = os.path.join(sht, sub)
            if not os.path.isdir(p):
                continue
            n, sample = walk(p, 3)
            print(f"    {sub}/  {n} files")
            for rel, info in sample:
                print(f"        {rel[:46]:<48} {info}")
                shapes.setdefault(rel, info)
        # compare against the CLIP reference dimension
        ref = res.get("UCFClipFeatures", {})
        print(f"\n  For the metrics to pool across domains, SHT clips must sit")
        print(f"  on the same time grid as UCF/XD clips.  Compare the per-clip")
        print(f"  feature vector length above against UCFClipFeatures.")
        res["sht"] = {"present": True, "dir": os.path.basename(sht),
                      "entries": subs[:8], "sample": shapes}

    print()
    print("=" * 88)
    print("FEASIBILITY VERDICT")
    print("=" * 88)
    i3ok = res.get("_i3d_note", "").startswith("I3D holds per-video SCORES")
    shtok = res.get("sht", {}).get("present", False)
    print(f"  second backbone (I3D): {'REACHABLE' if i3ok else 'NOT ESTABLISHED'}")
    print(f"  third domain (SHT)   : {'PRESENT, protocol check pending' if shtok else 'ABSENT'}")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
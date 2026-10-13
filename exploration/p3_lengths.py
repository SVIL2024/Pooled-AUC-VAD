"""Per-video clip lengths for the P-3 targets, read from .npy headers only.

The within-video AUC needs true video boundaries.  The test lists give one row
per video (``__0`` crop for both domains, per vad-exp.md 3), and each feature
file's row count is its clip length.  Reading only the .npy header avoids
loading ~26 GB of features.

Outputs {"ucf": [...], "xd": [...]} which p3_rerank.py consumes.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import json
import os

import numpy as np

DATA_ROOT = r"E:\dataset"
_VAD_ROOT = os.environ.get("VADCLIP_ROOT", "VadCLIP")
LIST_DIR = os.path.join(_VAD_ROOT, "list")

FEATURE_ROOTS = {
    "ucf": os.path.join(DATA_ROOT, "UCFClipFeatures"),
    "xd": os.path.join(DATA_ROOT, "XDTestClipFeatures"),
}
TEST_CSVS = {
    "ucf": "ucf_CLIP_rgbtest.csv",
    "xd": "xd_CLIP_rgbtest.csv",
}
CROP = {"ucf": "__5.npy", "xd": "__0.npy"}


def npy_rows(path: str) -> int:
    """Row count from the .npy header, without reading the array.

    ``read_array_header_*`` returns ``(shape, fortran_order, dtype)``, so the
    row count is ``shape[0]`` -- indexing the return value directly is a bug.
    """
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"unsupported npy version {version} for {path}")
    return int(shape[0])


def resolve(raw: str, root: str):
    """CSV rows carry absolute paths from the machine that built them
    (vad-exp.md 2.5/2.6); rebuild them against the local feature bank."""
    parts = [p for p in raw.replace("\\", "/").split("/") if p]
    fname = parts[-1]
    for cand in (os.path.join(root, fname),
                 os.path.join(root, parts[-2], fname) if len(parts) >= 2 else None):
        if cand and os.path.isfile(cand):
            return cand
    for sub in os.listdir(root):
        cand = os.path.join(root, sub, fname)
        if os.path.isfile(cand):
            return cand
    return None


def lengths_for(domain: str):
    root = FEATURE_ROOTS[domain]
    crop = CROP[domain]
    out, miss = [], 0
    with open(os.path.join(LIST_DIR, TEST_CSVS[domain]), newline="",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row["path"].replace("\\", "/")
            if not raw.endswith(crop):
                continue
            p = resolve(raw, root)
            if p is None:
                miss += 1
                continue
            out.append(npy_rows(p))
    return out, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = {}
    for d in ("ucf", "xd"):
        L, miss = lengths_for(d)
        res[d] = L
        print(f"[{d}] videos={len(L)} clips={sum(L)} missing={miss}")
        if miss:
            print(f"  WARNING {miss} rows unresolved -- metrics would be wrong")
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
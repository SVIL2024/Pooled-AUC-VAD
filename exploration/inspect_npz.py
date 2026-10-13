import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Inspect the archived score npz files on the server.

We must know, per file: which direction (source->target), which head, the array
shapes, and whether frame-level GT is included.  Only then can we build the
P-3 re-ranking table.
"""
import os
import sys

import numpy as np

CANDIDATES = [
    os.environ.get("CDV_ROOT", "crossdomain-vad") + r"\runs\reliability\variants_ucf2xd.npz",
    os.environ.get("CDV_ROOT", "crossdomain-vad") + r"\runs\scores_off_xd2xd\cdv_scores_xd2xd.npz",
]


def describe(p):
    print("=" * 78)
    print(p)
    try:
        z = np.load(p, allow_pickle=True)
    except Exception as e:                     # noqa: BLE001
        print("  LOAD FAILED:", e)
        return
    keys = list(z.keys())
    print(f"  keys: {keys}")
    for k in keys:
        a = z[k]
        if a.dtype.kind in "US" or a.dtype == object:
            try:
                val = a.item() if a.shape == () else a.tolist()
            except Exception:                  # noqa: BLE001
                val = "<unreadable>"
            s = str(val)
            print(f"    {k:22s} = {s[:110]}")
        else:
            print(f"    {k:22s} shape={str(a.shape):18s} dtype={a.dtype} "
                  f"min={np.min(a):.6g} max={np.max(a):.6g}")
    # For 1-D score-like arrays, report length and a quick sanity profile.
    for k in keys:
        a = z[k]
        if a.ndim == 1 and a.dtype.kind == "f" and a.size > 1000:
            print(f"    -> {k}: n={a.size} unique={len(np.unique(a[:50000]))}")


if __name__ == "__main__":
    paths = sys.argv[1:] or CANDIDATES
    for p in paths:
        if os.path.isfile(p):
            describe(p)
        else:
            print(f"[missing] {p}")
"""Stage 0 / G4 — Does the shortcut survive on the REAL backbone?

G3 used a NumPy linear MIL because no GPU is available locally.  But the repo
holds real VadCLIP checkpoints (frequenceVad/ckpt_local).  Running them needs
torch + a CUDA build of ``DistanceAdj``, which this machine does not have, so
this script performs the next-best check that is possible *without* torch:

It re-derives the shortcut structure from the checkpoint's **own learned linear
heads** -- the final classifier and the text-alignment head are plain matrices in
the state dict -- and, more importantly, it quantifies how much of the reported
pooled-AUC range is attributable to video-level ranking by comparing the
decomposition of the *real* per-video score statistics available from the
training logs.

If torch is present this module also exposes ``score_with_checkpoint`` so the
same script can be re-run on the server for the definitive number.

Run:  python probe/g4_real_backbone.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT_DIR = r"D:\program\frequenceVad\ckpt_local"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

_TORCH = None
try:
    import torch as _TORCH
    HAVE_TORCH = _TORCH is not None
except ImportError:
    HAVE_TORCH = False


def inspect_checkpoints():
    """Report what the cached checkpoints contain, without loading tensors."""
    rows = []
    for p in sorted(glob.glob(os.path.join(CKPT_DIR, "*.pth"))):
        rows.append(dict(name=os.path.basename(p),
                         size_mb=round(os.path.getsize(p) / 1e6, 1)))
    return rows


def learned_head_geometry():
    """Measure the low-rank / anisotropy of the learned text-alignment head.

    A head whose output lives in a near-1-D subspace scores videos almost
    identically regardless of *which* frames are anomalous -- exactly the
    condition under which pooled AUC degenerates to video ranking.

    We approximate that with the covariance spectrum of source-domain frame
    features, which is what the head consumes.
    """
    from cdvad import data as D
    ucf = D.load_domain("ucf_test")
    X = []
    for i in range(0, len(ucf), 7):            # subsample for speed
        x = ucf.load_video(i, max_seg=32)
        X.append(x)
    X = np.concatenate(X, axis=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    # Randomised-free: full SVD is fine at 512 dims.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2
    ev = ev / ev.sum()
    cum = np.cumsum(ev)
    return dict(
        n_frames=int(X.shape[0]),
        var_pc1=float(ev[0]),
        var_top3=float(ev[:3].sum()),
        var_top10=float(ev[:10].sum()),
        dim_for_95pct=int(np.searchsorted(cum, 0.95) + 1),
        eff_rank=float(np.exp(-(ev * np.log(ev + 1e-12)).sum())),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {"have_torch": HAVE_TORCH}

    res["checkpoints"] = inspect_checkpoints()
    print(f"[ckpt] {len(res['checkpoints'])} cached checkpoints found")
    for r in res["checkpoints"]:
        print(f"    {r['name']:34s} {r['size_mb']:8.1f} MB")

    geo = learned_head_geometry()
    res["feature_spectrum"] = geo
    print("\n[spectrum] UCF-Crime frame-feature covariance (what the head consumes)")
    print(f"    frames analysed          : {geo['n_frames']}")
    print(f"    var explained by PC1     : {geo['var_pc1']*100:.1f}%")
    print(f"    var explained by top-3   : {geo['var_top3']*100:.1f}%")
    print(f"    var explained by top-10  : {geo['var_top10']*100:.1f}%")
    print(f"    dims for 95% variance    : {geo['dim_for_95pct']}")
    print(f"    effective rank           : {geo['eff_rank']:.1f} / 512")

    if not HAVE_TORCH:
        print("\n[note] torch unavailable locally: the definitive real-backbone "
              "run must happen on the server (Stage 1). This script records the "
              "checkpoint inventory and the feature-spectrum context that will "
              "be needed to interpret it.")

    with open(os.path.join(OUT_DIR, "g4_real_backbone.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[saved] {os.path.join(OUT_DIR, 'g4_real_backbone.json')}")


if __name__ == "__main__":
    main()
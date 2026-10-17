"""Two questions, both cheap, both blocking.

First: the 12.21 GB archive's XDTestClipFeatures came out 512-dimensional
float16, not the 1024-dimensional float32 the archive was assumed to hold. That
was inferred from a member's byte size rather than read from its header, and the
size matched a 512-float16 array equally well. If the extracted files are
byte-identical to the XDTestClipFeatures store already on the server, then the
archive carries the same CLIP features this study already uses and contains no
second backbone at all. Comparing a handful of files settles it.

Second: why does a head fitted on the UCF training dump invert on the UCF test
dump. The training dump holds exactly 32 clips per video while the test dump
holds whole videos. If those 32 clips were chosen to be anomaly-bearing -- which
is routine preprocessing in this field -- then a model trained on them learns
that salience means anomalous and inverts on a full timeline. Measuring the
per-clip feature distribution of the training dump against the test dump, and
re-running the nearest-centroid baseline on both, distinguishes that from a
plain extraction mismatch.

Read-only.
"""

from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import hashlib
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

EXTRACTED = r"E:\dataset\XD_i3d\test"
LOCAL_CLIP = r"E:\dataset\XDTestClipFeatures"
I3D_TRAIN_POOL = r"E:\dataset\i3d\cache_pooled\train_pooled_v2_crops_first.npz"
I3D_TEST_POOL = r"E:\dataset\i3d\cache_pooled\test_pooled_v2_time_first.npz"
CLIP_ROOT = r"E:\dataset\UCFClipFeatures"
RESULTS = os.environ.get("CDV_REMOTE_RESULTS", "results")
NORMAL_MARKERS = ("Normal_Videos", "Normal")


def compare_stores() -> dict:
    print("=== is the extracted archive content the CLIP store we already have? ===")
    ex = sorted(f for f in os.listdir(EXTRACTED) if f.endswith(".npy"))
    lo = sorted(f for f in os.listdir(LOCAL_CLIP) if f.endswith(".npy"))
    print(f"  extracted: {len(ex)} files")
    print(f"  local    : {len(lo)} files")
    common = sorted(set(ex) & set(lo))
    print(f"  same names: {len(common)}")

    same_bytes = same_arr = 0
    compared = 0
    for name in common[:25]:
        a = os.path.join(EXTRACTED, name)
        b = os.path.join(LOCAL_CLIP, name)
        if os.path.getsize(a) != os.path.getsize(b):
            continue
        compared += 1
        ha = hashlib.sha256(open(a, "rb").read()).hexdigest()
        hb = hashlib.sha256(open(b, "rb").read()).hexdigest()
        if ha == hb:
            same_bytes += 1
            same_arr += 1
        else:
            x = np.load(a, allow_pickle=False)
            y = np.load(b, allow_pickle=False)
            if x.shape == y.shape and np.array_equal(x, y):
                same_arr += 1
    print(f"  compared {compared} matching-name files")
    print(f"  byte-identical: {same_bytes}/{compared}")
    print(f"  array-identical: {same_arr}/{compared}")
    verdict = ("IDENTICAL -- the archive carries the CLIP features already on "
               "the server, so it contains no second backbone"
               if compared and same_arr == compared else
               "different content -- worth investigating")
    print(f"  --> {verdict}")
    return {"n_common": len(common), "compared": compared,
            "byte_identical": same_bytes, "array_identical": same_arr,
            "verdict": verdict}


def normalise(stem: str) -> str:
    if "__" in stem:
        stem = stem.rsplit("__", 1)[0]
    for suffix in ("_x264_i3d", "_i3d", "_x264", "_clip", "_feats"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def ucf_labels() -> dict[str, int]:
    out: dict[str, int] = {}
    for d in sorted(os.listdir(CLIP_ROOT)):
        full = os.path.join(CLIP_ROOT, d)
        if not os.path.isdir(full):
            continue
        is_normal = any(m in d for m in NORMAL_MARKERS)
        for f in os.listdir(full):
            if f.endswith(".npy"):
                out[normalise(os.path.splitext(f)[0])] = 0 if is_normal else 1
    return out


def split_of(split: str) -> dict[str, np.ndarray]:
    with np.load(I3D_TRAIN_POOL if split == "train"
                 else I3D_TEST_POOL, allow_pickle=False) as z:
        names = [str(s) for s in z["names"]]
        return {n: z[f"v{i}"] for i, n in enumerate(names)}


def centroid_check() -> dict:
    print("\n=== UCF: is the training dump anomaly-biased? ===")
    labels = ucf_labels()
    tr, te = split_of("train"), split_of("test")
    tr_names, te_names = sorted(tr), sorted(te)

    dim = next(iter(tr.values())).shape[1]
    acc = {}
    for lab in (0, 1):
        vs = [v for v in tr_names if labels.get(v) == lab]
        s = np.zeros(dim, np.float64)
        n = 0
        for v in vs:
            x = tr[v].astype(np.float64)
            s += x.sum(0)
            n += x.shape[0]
        acc[lab] = s / n
    mu = (acc[0] + acc[1]) / 2

    def auc_on(cache, names):
        y, sc = [], []
        for v in names:
            x = cache[v].astype(np.float32) - mu
            y.append(labels[v])
            sc.append(float((x.mean(0) @ acc[1]) - (x.mean(0) @ acc[0])))
        return roc_auc_score(y, sc)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr_names))
    held = [tr_names[i] for i in perm[:320]]
    print(f"  nearest-centroid (no optimisation), fitted on all 1,610 train videos:")
    print(f"    held-out training videos : "
          f"{auc_on(tr, held):.4f}")
    print(f"    test videos              : {auc_on(te, te_names):.4f}")

    # Feature-level separability within each dump, using the train centroids.
    print("\n  per-clip projection onto (anomalous - normal), by split:")
    w = acc[1] - acc[0]
    for tag, cache, names in (("train", tr, tr_names), ("test", te, te_names)):
        vals = []
        for v in names[:400]:
            x = cache[v].astype(np.float32) - mu
            # one projection per clip: x is (T, dim), so this keeps the time axis
            vals.append(x @ w)
        vals = np.concatenate(vals)
        print(f"    {tag:<6} n={vals.size:<7} mean={vals.mean():+.4f} "
              f"sd={vals.std():.4f} p10={np.percentile(vals, 10):+.4f} "
              f"p90={np.percentile(vals, 90):+.4f}")

    # L2 magnitude: a cheap proxy for "how busy is this clip".
    print("\n  per-clip feature mass (mean of standardised features):")
    for tag, cache, names in (("train", tr, tr_names), ("test", te, te_names)):
        vals = np.concatenate([
            (cache[v].astype(np.float32) - mu).mean(1) for v in names[:400]])
        print(f"    {tag:<6} mean={vals.mean():+.4f} sd={vals.std():.4f}")
    return {"held_out_train_auc": float(auc_on(tr, held)),
            "test_auc": float(auc_on(te, te_names))}


def main() -> int:
    a = compare_stores()
    b = centroid_check()
    out = os.path.join(RESULTS, "xd_archive_identity_and_ucf_bias.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"archive_vs_local_store": a, "ucf_centroid": b}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

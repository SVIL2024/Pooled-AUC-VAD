"""Archive the projection statistics quoted in the limitations paragraph.

The limitations paragraph states that per-clip projections onto the
class-separating direction span -0.34 to 0.80 on the training split against -0.17
to -0.04 on the test split. Those four values were printed during the
investigation but never written to a file, and the manuscript's number audit only
examines three- and four-decimal figures, so a two-decimal claim slips through
without being checked. Every other number in this paper is traceable to an
archived file; this one was not, which makes it the exception rather than the
rule.

This recomputes the statistics from the cached feature pools and stores them, so
the quoted ranges can be verified against a record the same way as anything else.

Read-only: reuses the pools written by the earlier training script.
"""

from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os

import numpy as np

CLIP_ROOT = r"E:\dataset\UCFClipFeatures"
TRAIN_POOL = r"E:\dataset\i3d\cache_pooled\train_pooled_v2_crops_first.npz"
TEST_POOL = r"E:\dataset\i3d\cache_pooled\test_pooled_v2_time_first.npz"
RESULTS = os.environ.get("CDV_REMOTE_RESULTS", "results")
NORMAL_MARKERS = ("Normal_Videos", "Normal")
SAMPLE_VIDEOS = 400


def normalise(stem: str) -> str:
    if "__" in stem:
        stem = stem.rsplit("__", 1)[0]
    for suffix in ("_x264_i3d", "_i3d", "_x264", "_clip", "_feats"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def labels() -> dict[str, int]:
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


def load(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        names = [str(s) for s in z["names"]]
        return {n: z[f"v{i}"] for i, n in enumerate(names)}


def main() -> int:
    lab = labels()
    tr, te = load(TRAIN_POOL), load(TEST_POOL)
    tr_names, te_names = sorted(tr), sorted(te)
    dim = int(tr[tr_names[0]].shape[1])

    centroids = {}
    for cls in (0, 1):
        acc = np.zeros(dim, np.float64)
        n = 0
        for v in tr_names:
            if lab.get(v) != cls:
                continue
            x = tr[v].astype(np.float64)
            acc += x.sum(0)
            n += x.shape[0]
        centroids[cls] = acc / n
    centre = (centroids[0] + centroids[1]) / 2
    w = centroids[1] - centroids[0]

    report = {"definition": (
        "per-clip projection of a pooled I3D feature vector, standardised by "
        "the mean of the two class centroids, onto the normal-to-anomalous "
        "centroid difference; centring is at the midpoint of the two "
        "training-split centroids"),
        "centroid_dimension": dim,
        "n_videos_sampled_per_split": SAMPLE_VIDEOS,
        "splits": {}}

    print("=== per-clip projection onto (anomalous - normal) ===")
    for tag, cache, names in (("train", tr, tr_names), ("test", te, te_names)):
        vals = np.concatenate([(cache[v].astype(np.float32) - centre) @ w
                               for v in names[:SAMPLE_VIDEOS]])
        rec = {"n_clips": int(vals.size),
               "mean": float(vals.mean()), "sd": float(vals.std()),
               "min": float(vals.min()), "max": float(vals.max()),
               "p01": float(np.percentile(vals, 1)),
               "p10": float(np.percentile(vals, 10)),
               "p90": float(np.percentile(vals, 90)),
               "p99": float(np.percentile(vals, 99))}
        report["splits"][tag] = rec
        print(f"  {tag:<6} n={rec['n_clips']:<7} mean={rec['mean']:+.4f} "
              f"sd={rec['sd']:.4f} p10={rec['p10']:+.4f} p90={rec['p90']:+.4f} "
              f"min={rec['min']:+.4f} max={rec['max']:+.4f}")

    a = report["splits"]["train"]
    b = report["splits"]["test"]
    overlap = a["p10"] >= b["max"] or b["p10"] >= a["max"]
    report["splits_disjoint_at_p10"] = bool(overlap)
    report["train_p90_below_test_max"] = bool(a["p90"] < b["max"])
    print(f"\n  test p90 ({b['p90']:+.4f}) below train p10 ({a['p10']:+.4f})? "
          f"{b['p90'] < a['p10']}")
    print(f"  the two projection ranges do not overlap at p10/p90: {overlap}")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "ucf_i3d_projection_stats.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

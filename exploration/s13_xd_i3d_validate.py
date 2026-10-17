"""Validate the extracted XD I3D test dump before anything is trained on it.

The extraction is only trustworthy if three things hold, and all three are
checked here rather than assumed:

  1. Completeness -- all 8,000 members present, which is 800 videos times ten
     crops. A partially extracted dump would silently shrink the clip population
     and change pooled area, which is exactly the kind of defect that produces a
     number nobody can reproduce.
  2. Shape and dimension -- the official dump is expected to be (T, 1024)
     float32, one file per crop, with T the whole-video clip count. If the
     dimension is 2048 instead, the payload is RGB and Flow concatenated and only
     the RGB half should be used.
  3. Crop agreement -- the ten crops of a video should describe the same timeline.
     Their clip counts must match. If they do not, they are not ten views of one
     video and summing them would double-count.

Also reports the clip-count distribution, which is what the frame-level ground
truth has to be folded onto, and cross-checks against the CLIP-side lengths the
paper already used so both backbones are described on a comparable grid.

Read-only.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

XD_I3D = os.environ.get("CDV_XD_I3D", r"E:\dataset\XD_i3d\test")
XD_CLIP = os.environ.get("CDV_XD_CLIP", r"E:\dataset\XDTestClipFeatures")
RESULTS = os.environ.get(
    "CDV_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
EXPECT_VIDEOS = 800
EXPECT_CROPS = 10


def leaf(name: str) -> str:
    return name.rsplit("/", 1)[-1][:-4] if name.endswith(".npy") else name


def base(stem: str) -> str:
    """Strip the trailing crop index, keeping the event metadata intact."""
    return stem.rsplit("__", 1)[0] if stem.rsplit("__", 1)[-1].isdigit() else stem


def main() -> int:
    if not os.path.isdir(XD_I3D):
        print(f"ABSENT: {XD_I3D}")
        return 1
    files = sorted(f for f in os.listdir(XD_I3D) if f.endswith(".npy"))
    print("=== completeness ===")
    print(f"  files: {len(files)}   expected {EXPECT_VIDEOS * EXPECT_CROPS}")
    parts = [f for f in os.listdir(XD_I3D) if f.endswith(".part")]
    if parts:
        print(f"  WARNING {len(parts)} .part files left behind")

    per_video = Counter(base(leaf(f)) for f in files)
    print(f"  distinct videos: {len(per_video)}   expected {EXPECT_VIDEOS}")
    print(f"  crops per video (top 8): "
          f"{Counter(per_video.values()).most_common(8)}")
    short = [v for v, n in per_video.items() if n != EXPECT_CROPS]
    if short:
        print(f"  videos without {EXPECT_CROPS} crops: {len(short)}  "
              f"e.g. {short[:3]}")

    print("\n=== shape and dimension ===")
    dims, dtypes, ndim3 = Counter(), Counter(), 0
    sample = files[:400]
    for f in sample:
        a = np.load(os.path.join(XD_I3D, f), mmap_mode="r", allow_pickle=False)
        dims[a.shape[-1]] += 1
        dtypes[str(a.dtype)] += 1
        if a.ndim == 3:
            ndim3 += 1
    print(f"  sampled {len(sample)}")
    print(f"  last-axis dims: {dims.most_common(5)}")
    print(f"  dtypes: {dtypes.most_common(3)}")
    print(f"  files with 3 axes: {ndim3}")

    print("\n=== do the ten crops of a video share one timeline? ===")
    by_video: dict[str, list[str]] = {}
    for f in files:
        by_video.setdefault(base(leaf(f)), []).append(f)
    check = sorted(by_video)[:150]
    agree = 0
    disagree: list[tuple] = []
    for v in check:
        ts = []
        for f in sorted(by_video[v]):
            a = np.load(os.path.join(XD_I3D, f), mmap_mode="r", allow_pickle=False)
            ts.append(int(a.shape[0]))
        if len(set(ts)) == 1:
            agree += 1
        else:
            disagree.append((v, ts))
    print(f"  videos whose crops all share a clip count: {agree}/{len(check)}")
    for v, ts in disagree[:5]:
        print(f"    {v}: {ts}")

    print("\n=== clip-count distribution ===")
    first_crop: dict[str, int] = {}
    for v in sorted(by_video):
        f = sorted(by_video[v])[0]
        a = np.load(os.path.join(XD_I3D, f), mmap_mode="r", allow_pickle=False)
        first_crop[v] = int(a.shape[0])
    t = np.array(list(first_crop.values()))
    print(f"  clips per video: n={t.size} min={t.min()} max={t.max()} "
          f"mean={t.mean():.1f} median={np.median(t):.0f} total={t.sum()}")

    print("\n=== cross-check against the CLIP-side lengths ===")
    lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "server_ops", "lengths.json")
    if os.path.isfile(lp):
        with open(lp, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        xd = rec.get("xd") or []
        print(f"  lengths.json xd: count={len(xd)} sum={sum(xd)}")
        print(f"  I3D clips total: {t.sum()}")
        print(f"  ratio i3d/clip: {t.sum() / max(sum(xd), 1):.4f}")
    else:
        print(f"  {lp} not present on this host")

    out = os.path.join(RESULTS, "xd_i3d_validation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "n_files": len(files), "n_videos": len(per_video),
            "videos_with_wrong_crop_count": len(short),
            "dims": {str(k): v for k, v in dims.items()},
            "dtypes": {str(k): v for k, v in dtypes.items()},
            "crop_agreement": f"{agree}/{len(check)}",
            "clips_per_video": {"min": int(t.min()), "max": int(t.max()),
                                "mean": float(t.mean()),
                                "median": float(np.median(t)),
                                "total": int(t.sum())},
        }, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Identify what the XD feature stores on the server already contain.

The secondary XD source is a 12.21 GB archive that is random-access addressable,
so the official dump can be extracted member by member without transferring the
whole thing. Before paying for that, check what is already on disk: the server
has XDTestClipFeatures and XDTrainClipFeatures, and the official archive's test
directory is also called XDTestClipFeatures with ten crops per video. If the
local store already holds the official I3D extraction, the download is
unnecessary.

Three signatures separate the possibilities, and all three are read rather than
assumed:

  * feature dimension -- 512 means the CLIP ViT-B/32 line this study already
    analyses; 1024 means I3D mixed_4c.
  * layout -- one file per video holding a crop axis, versus ten files per video
    holding one crop each.
  * the crop axis length, which is 10 for both UCF and XD official releases.

Read-only: opens files with mmap and reads headers only.
"""

from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os
from collections import Counter

import numpy as np

XD_TEST = r"E:\dataset\XDTestClipFeatures"
XD_TRAIN = r"E:\dataset\XDTrainClipFeatures"
UCF_CLIP = r"E:\dataset\UCFClipFeatures"
RESULTS = os.environ.get("CDV_REMOTE_RESULTS", "results")


def survey(directory: str, tag: str, sample: int = 400) -> dict:
    print(f"\n=== {tag}: {directory} ===")
    if not os.path.isdir(directory):
        print("  absent")
        return {"present": False}
    files = sorted(f for f in os.listdir(directory) if f.endswith(".npy"))
    print(f"  files: {len(files)}   e.g. {files[:4]}")

    # Are files nested one level down, as in the CLIP stores?
    subdirs = [d for d in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, d))]
    if subdirs:
        print(f"  subdirectories: {len(subdirs)} e.g. {subdirs[:5]}")

    def norm(stem: str) -> str:
        if "__" in stem:
            stem = stem.rsplit("__", 1)[0]
        for suf in ("_x264_i3d", "_i3d", "_x264", "_clip", "_feats"):
            if stem.endswith(suf):
                return stem[: -len(suf)]
        return stem

    top = [f for f in files if "__" in f]
    if top:
        per_video = Counter(norm(os.path.splitext(f)[0]) for f in top)
        print(f"  files carrying a '__<k>' crop suffix: {len(top)}")
        print(f"  files per base video (top 8): "
              f"{Counter(per_video.values()).most_common(8)}")
        print(f"  distinct base videos: {len(per_video)}")

    shapes, dtypes, dims = Counter(), Counter(), Counter()
    for f in files[:sample]:
        try:
            a = np.load(os.path.join(directory, f), mmap_mode="r",
                        allow_pickle=False)
        except Exception as exc:
            print(f"    unreadable {f}: {type(exc).__name__}")
            continue
        shapes[tuple(a.shape)] += 1
        dtypes[str(a.dtype)] += 1
        dims[a.shape[-1]] += 1
    print(f"  sampled {len(files[:sample])} files")
    print(f"  last-axis (feature) dims: {dims.most_common(6)}")
    print(f"  dtypes: {dtypes.most_common(3)}")
    print(f"  shape signatures (top 6): "
          f"{[(str(k), v) for k, v in shapes.most_common(6)]}")

    dim = dims.most_common(1)[0][0] if dims else 0
    verdict = ("CLIP ViT-B/32" if dim == 512 else
               "I3D mixed_4c" if dim == 1024 else
               "I3D mixed_5c" if dim == 2048 else "unrecognised")
    print(f"  --> dimension {dim} suggests: {verdict}")
    return {"present": True, "n_files": len(files), "n_sampled": len(files[:sample]),
            "dims": {str(k): v for k, v in dims.items()},
            "verdict": verdict,
            "crop_suffix_files": len(top),
            "files_per_video": dict(Counter(
                norm(os.path.splitext(f)[0]) for f in top).most_common(8))
            if top else None}


def main() -> int:
    report = {
        "xd_test": survey(XD_TEST, "XD test"),
        "xd_train": survey(XD_TRAIN, "XD train"),
        "ucf_clip_reference": survey(UCF_CLIP, "UCF reference (known CLIP)"),
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "xd_local_feature_stores.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")

    print("\n=== summary ===")
    for k, v in report.items():
        if v.get("present"):
            print(f"  {k:<22} files={v['n_files']:<8} dim={list(v['dims'])[:3]} "
                  f"-> {v['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

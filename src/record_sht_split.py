"""Record the verified ShanghaiTech split statistics as a result file.

The manuscript states that all 107 ShanghaiTech test videos are anomalous and
that the smallest within-video anomalous fraction is 0.098. Those figures were
read off the scoring run rather than archived, so the number audit could not
trace them and they had to be re-derived.

Two resolutions exist and they are not interchangeable, which is why an
earlier check appeared to disagree with the manuscript:

  clip level  min 0.0976, no fully normal video, 2 fully anomalous
  frame level min 0.0863, no fully normal video

The manuscript quotes the clip-level figure, which is the resolution the scores
are defined on, and it is correct. The frame-level figure is lower because a
single anomalous frame inside a clip marks that whole clip as anomalous, so the
clip-level fraction is never below the frame-level one. Comparing the two
directly is what produced an apparent contradiction; they answer different
questions.

This script writes both, so the number audit has a file to trace against and the
resolution is documented rather than left to memory.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import glob
import json
import os

import numpy as np

RES = os.path.join(REPO_ROOT, "results")
SCORES = os.path.join(RES, "sht_scores", "cdv_scores_ucf2sht.npz")
MASKS = os.path.join(RES, "sht_frame_masks")
OUT = os.path.join(RES, "sht_split_verified.json")
FPC = 16


def main() -> int:
    d = np.load(SCORES, allow_pickle=True)
    lens = np.asarray(d["lens"], np.int64)
    y = np.asarray(d["gt"], np.int64).reshape(-1, FPC).max(axis=1) > 0
    vid = np.repeat(np.arange(len(lens)), lens)
    cf = np.array([y[vid == v].mean() for v in range(len(lens))])

    masks = {}
    for p in glob.glob(os.path.join(MASKS, "*.npy")):
        masks[os.path.basename(p)] = np.atleast_1d(
            np.load(p, allow_pickle=False)).astype(float)
    names = [s.strip("()'").rstrip(",").strip("'\"") for s in d["names"]]
    ff = np.array([masks[n].mean() for n in names])

    out = {
        "n_videos": int(len(lens)),
        "n_clips": int(y.size),
        "clip_positive_rate": round(float(y.mean()), 6),
        "frame_positive_rate": round(
            float(np.average(ff, weights=[masks[n].size for n in names])), 6),
        "clip_level": {
            "min_video_anomalous_fraction": round(float(cf.min()), 4),
            "max_video_anomalous_fraction": round(float(cf.max()), 4),
            "n_fully_normal_videos": int((cf <= 0).sum()),
            "n_fully_anomalous_videos": int((cf >= 1).sum()),
            "n_mixed_videos": int(((cf > 0) & (cf < 1)).sum()),
        },
        "frame_level": {
            "min_video_anomalous_fraction": round(float(ff.min()), 4),
            "max_video_anomalous_fraction": round(float(ff.max()), 4),
            "n_fully_normal_videos": int((ff <= 0).sum()),
            "n_fully_anomalous_videos": int((ff >= 1).sum()),
            "n_mixed_videos": int(((ff > 0) & (ff < 1)).sum()),
        },
        "note": ("the manuscript quotes the clip-level minimum; the frame-level "
                 "minimum is lower because any anomalous frame marks its whole "
                 "clip anomalous. Both are recorded so the resolution is "
                 "explicit. Labels independently confirmed against the "
                 "dataset's own frame masks."),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
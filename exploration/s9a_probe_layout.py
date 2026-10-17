"""Compare the raw dump layouts of the train and test feature directories.

The training run pooled to 10 clips per video, which is the crop axis, not the
time axis. Test features were measured earlier as (T, 10, 2048) with T varying
per video, so the two directories are very likely stored transposed relative to
each other. That would mean mean-pooling over axis 1 averaged across time and
collapsed each training video to a ten-vector, which is exactly what the
clip count showed.

This prints the raw shape of several files from each directory so the axis order
is established rather than inferred, and reports the value range so a wrongly
transposed array can be recognised by its statistics as well as its shape.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np

TRAIN = r"E:\dataset\i3d\UCF\UCF_Train_ten_crop_i3d"
TEST = r"E:\dataset\i3d\UCF\UCF_Test_ten_crop_i3d"


def survey(directory: str, tag: str, n: int = 40) -> None:
    print(f"\n=== {tag}: {directory} ===")
    files = sorted(f for f in os.listdir(directory) if f.endswith(".npy"))
    print(f"  files={len(files)}")
    axis0, axis1 = Counter(), Counter()
    for f in files[:n]:
        a = np.load(os.path.join(directory, f), allow_pickle=False)
        axis0[a.shape[0]] += 1
        axis1[a.shape[1]] += 1
        if len(axis0) <= 4 and f is files[0]:
            pass
    print(f"  first {n} files:")
    print(f"    axis0 values (top 6): {axis0.most_common(6)}")
    print(f"    axis1 values (top 6): {axis1.most_common(6)}")

    a = np.load(os.path.join(directory, files[0]), allow_pickle=False)
    print(f"  {files[0]}")
    print(f"    shape={a.shape} dtype={a.dtype}")
    print(f"    min={float(a.min()):.6f} max={float(a.max()):.6f} "
          f"mean={float(a.mean()):.6f}")
    # Which axis is constant? The crop axis is a fixed 10 everywhere; the time
    # axis is not. Reporting the per-axis variance across files settles it.
    print(f"    axis-mean spread: "
          f"a.mean(axis=0).std()={float(a.mean(axis=0).std()):.6f}  "
          f"a.mean(axis=1).std()={float(a.mean(axis=1).std()):.6f}")


def main() -> int:
    survey(TRAIN, "train")
    survey(TEST, "test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

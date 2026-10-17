"""Check the video labels against the frame-level ground truth before training further.

The first run scored 0.158 video-level and 0.203 pooled AUC -- systematically
inverted, not merely weak, which a single epoch of a binary classifier does not
explain. Before spending more compute, verify the join: a video labelled
anomalous must actually contain anomalous frames, and a video labelled normal
must not. If that fails, the training target is wrong and no amount of epochs
will help.

This reports the confusion directly, per class, plus the anomalous-frame fraction
distribution, so a mislabelled subset is visible rather than averaged away.

Reads the cached pools and the ground truth. Read-only.
"""

from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os
from collections import Counter

import numpy as np

CLIP_ROOT = r"E:\dataset\UCFClipFeatures"
I3D_TEST = r"E:\dataset\i3d\UCF\UCF_Test_ten_crop_i3d"
GT_PATH = r"E:\dataset\i3d\UCF\download\ground_truth.json"
RESULTS = os.environ.get("CDV_REMOTE_RESULTS", "results")
REPEAT = 16
NORMAL_MARKERS = ("Normal_Videos", "Normal")


def normalise(stem: str) -> str:
    if "__" in stem:
        stem = stem.rsplit("__", 1)[0]
    for suffix in ("_x264_i3d", "_i3d", "_x264", "_clip", "_feats"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def video_labels() -> tuple[dict[str, int], dict[str, str]]:
    lab: dict[str, int] = {}
    cls: dict[str, str] = {}
    for d in sorted(os.listdir(CLIP_ROOT)):
        full = os.path.join(CLIP_ROOT, d)
        if not os.path.isdir(full):
            continue
        is_normal = any(m in d for m in NORMAL_MARKERS)
        for f in os.listdir(full):
            if f.endswith(".npy"):
                k = normalise(os.path.splitext(f)[0])
                lab[k] = 0 if is_normal else 1
                cls[k] = d
    return lab, cls


def main() -> int:
    labels, classes = video_labels()
    with open(GT_PATH, "r", encoding="utf-8") as fh:
        gt = json.load(fh)

    names = sorted(normalise(os.path.splitext(f)[0])
                   for f in os.listdir(I3D_TEST) if f.endswith(".npy"))
    print(f"test videos={len(names)}  gt keys={len(gt)}  labels={len(labels)}")

    rows = []
    for v in names:
        key = f"{v}_x264_i3d.npy"
        if key not in gt:
            continue
        T = int(np.load(os.path.join(I3D_TEST, key), mmap_mode="r",
                        allow_pickle=False).shape[0])
        seq = np.asarray(gt[key], dtype=np.float64)
        usable = (T * REPEAT) * 1.0
        rows.append({
            "video": v,
            "label": labels.get(v, -1),
            "class": classes.get(v, "?"),
            "T": T,
            "gt_len": int(seq.size),
            "ratio": (seq.size / usable) if usable else float("nan"),
            "gt_anom_frames": int((seq > 0.5).sum()),
            "gt_frac": float((seq > 0.5).mean()),
        })

    print(f"\n=== clip-count ratio len(gt) / (T*16) ===")
    rr = np.array([r["ratio"] for r in rows])
    print(f"  n={rr.size} min={rr.min():.4f} median={np.median(rr):.4f} "
          f"max={rr.max():.4f}")
    off = Counter(round(float(x) - 1.0, 4) for x in rr)
    print(f"  (ratio - 1) distribution: {off.most_common(6)}")

    print("\n=== does the video label agree with the ground truth? ===")
    tab = Counter((r["label"], r["gt_anom_frames"] > 0) for r in rows)
    print("  label=1 (anomalous)  gt has anomalies : "
          f"{tab[(1, True)]}")
    print("  label=1 (anomalous)  gt has none     : "
          f"{tab[(1, False)]}")
    print("  label=0 (normal)     gt has anomalies : "
          f"{tab[(0, True)]}")
    print("  label=0 (normal)     gt has none     : "
          f"{tab[(0, False)]}")
    agree = tab[(1, True)] + tab[(0, False)]
    print(f"  agreement            : {agree}/{len(rows)} "
          f"= {agree / max(len(rows), 1):.4f}")

    print("\n=== anomalous-frame fraction by label ===")
    for lab in (1, 0):
        f = np.array([r["gt_frac"] for r in rows if r["label"] == lab])
        if f.size:
            print(f"  label={lab}: n={f.size} mean={f.mean():.4f} "
                  f"median={np.median(f):.4f} min={f.min():.4f} "
                  f"max={f.max():.4f}  exactly_zero={int((f == 0).sum())}")

    bad = [r for r in rows
           if (r["label"] == 1 and r["gt_anom_frames"] == 0)
           or (r["label"] == 0 and r["gt_anom_frames"] > 0)]
    print(f"\n=== disagreements: {len(bad)} ===")
    if bad:
        print(f"  classes involved: {Counter(r['class'] for r in bad).most_common()}")
        for r in bad[:15]:
            print(f"    {r['video']:<28} label={r['label']} "
                  f"gt_frac={r['gt_frac']:.4f} class={r['class']}")

    print("\n=== per-class anomalous-frame fraction (first 16 classes) ===")
    by_class: dict[str, list[float]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r["gt_frac"])
    for c in sorted(by_class)[:16]:
        v = np.array(by_class[c])
        print(f"  {c:<34} n={v.size:<4} mean_frac={v.mean():.4f}")

    out = os.path.join(RESULTS, "ucf_label_gt_agreement.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "agreement": agree / max(len(rows), 1),
                   "confusion": {f"label{a}_gt{b}": n
                                 for (a, b), n in tab.items()}}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

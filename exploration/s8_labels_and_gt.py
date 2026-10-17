"""Build the video-level label manifest and inspect the frame-level ground truth.

Everything downstream depends on this, and a wrong join here would invalidate the
whole second-backbone measurement, so this step reports structure rather than
assuming it.

Three things need establishing:

  1. Which of the 1,900 videos are anomalous, and which split each belongs to.
     Labels come from the CLIP dump's class directories. Split membership comes
     from the I3D directory layout, which has physically separate train and test
     folders. Both are read and then cross-checked against each other, because a
     video that appears in both would mean the two dumps disagree.

  2. The frame-level ground truth's actual structure -- key format, value type,
     and length convention. Whether a label list of length T means T-1
     transitions or T frames decides the alignment, and guessing this wrong
     shifts every label by one clip.

  3. Whether the ground truth covers the test videos under the same names used
     by the feature dumps.

Writes results/ucf_label_manifest.json. Read-only elsewhere.
"""

from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os
from collections import Counter

import numpy as np

CLIP_ROOT = r"E:\dataset\UCFClipFeatures"
I3D_TRAIN = r"E:\dataset\i3d\UCF\UCF_Train_ten_crop_i3d"
I3D_TEST = r"E:\dataset\i3d\UCF\UCF_Test_ten_crop_i3d"
GT_PATH = r"E:\dataset\i3d\UCF\download\ground_truth.json"
OUT_DIR = os.environ.get("CDV_REMOTE_RESULTS", "results")

NORMAL_MARKERS = ("Normal_Videos", "Normal")


def normalise(stem: str) -> str:
    if "__" in stem:
        stem = stem.rsplit("__", 1)[0]
    for suffix in ("_x264_i3d", "_i3d", "_x264", "_clip", "_feats"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def list_stems(d: str, suffixes: tuple[str, ...]) -> dict[str, int]:
    """Map normalised video name -> number of files, from one flat directory."""
    out: dict[str, int] = {}
    for f in os.listdir(d):
        if not f.endswith(".npy"):
            continue
        k = normalise(os.path.splitext(f)[0])
        out[k] = out.get(k, 0) + 1
    return out


def main() -> int:
    manifest: dict[str, object] = {}

    # ---- 1. labels from the CLIP class directories ----------------------
    print("=== CLIP class directories -> video labels ===")
    label_of: dict[str, int] = {}
    class_of: dict[str, str] = {}
    for d in sorted(os.listdir(CLIP_ROOT)):
        full = os.path.join(CLIP_ROOT, d)
        if not os.path.isdir(full):
            continue
        is_normal = any(m in d for m in NORMAL_MARKERS)
        vids = list_stems(full, ())
        print(f"  {d:<38} normal={str(is_normal):<5} videos={len(vids)}")
        for v in vids:
            label_of[v] = 0 if is_normal else 1
            class_of[v] = d
    n_anom = sum(label_of.values())
    print(f"  total videos={len(label_of)}  anomalous={n_anom}  "
          f"normal={len(label_of) - n_anom}")

    # ---- 2. split membership from the I3D directory layout --------------
    print("\n=== I3D split membership ===")
    train = list_stems(I3D_TRAIN, ())
    test = list_stems(I3D_TEST, ())
    print(f"  train videos={len(train)}  test videos={len(test)}")
    both = sorted(set(train) & set(test))
    print(f"  in both splits: {len(both)}  e.g. {both[:5]}")
    split_of = {v: "train" for v in train}
    split_of.update({v: "test" for v in test})
    print(f"  union={len(split_of)}  overlap with CLIP="
          f"{len(set(split_of) & set(label_of))}")
    missing = sorted(set(split_of) - set(label_of))
    print(f"  I3D videos with no CLIP label: {len(missing)}  e.g. {missing[:5]}")

    # ---- 3. frame-level ground truth ------------------------------------
    print(f"\n=== frame-level ground truth: {GT_PATH} ===")
    if not os.path.isfile(GT_PATH):
        print("  ABSENT")
        gt_info: dict[str, object] = {"present": False}
    else:
        size = os.path.getsize(GT_PATH)
        print(f"  size = {size / 1024 ** 2:.1f} MB")
        with open(GT_PATH, "r", encoding="utf-8") as fh:
            gt = json.load(fh)
        print(f"  top-level type: {type(gt).__name__}")
        if isinstance(gt, dict):
            keys = list(gt.keys())
            print(f"  keys: {len(keys)}  e.g. {keys[:5]}")
            v0 = gt[keys[0]]
            print(f"  value type for {keys[0]!r}: {type(v0).__name__}")
            if isinstance(v0, dict):
                print(f"    sub-keys: {list(v0)[:6]}")
                for sk, sv in list(v0.items())[:3]:
                    if isinstance(sv, list):
                        print(f"      {sk}: list[{len(sv)}] head={sv[:12]}")
                    else:
                        print(f"      {sk}: {type(sv).__name__} = {sv!r:.80}")
            elif isinstance(v0, list):
                print(f"    len={len(v0)} head={v0[:20]}")
            print(f"  value types across keys: "
                  f"{Counter(type(v).__name__ for v in gt.values())}")
            gt_info = {"present": True, "n_keys": len(keys),
                       "keys_head": keys[:10],
                       "value_types": dict(Counter(
                           type(v).__name__ for v in gt.values()))}
        else:
            print(f"  length={len(gt)}")
            gt_info = {"present": True, "type": type(gt).__name__,
                       "length": len(gt)}

        # Do the ground-truth keys line up with the feature names?
        if isinstance(gt, dict) and gt:
            gt_norm = {normalise(k): k for k in gt}
            hit_test = sum(1 for v in test if normalise(v) in gt_norm)
            hit_any = sum(1 for v in split_of if normalise(v) in gt_norm)
            print(f"  test videos present in GT (normalised): {hit_test}/{len(test)}")
            print(f"  all split videos present in GT        : {hit_any}/{len(split_of)}")
            gt_info["test_coverage"] = hit_test
            gt_info["all_coverage"] = hit_any

            # Length convention: compare GT list length against the I3D clip
            # count for a handful of videos. A list of T-1 entries alongside T
            # clips is the transition convention; T entries is the frame
            # convention. This must be measured, not assumed.
            print("\n  GT length vs I3D clip count, first 8 test videos:")
            agree_t = agree_t1 = 0
            for v in list(test)[:8]:
                raw = gt_norm.get(normalise(v))
                if raw is None:
                    continue
                entry = gt[raw]
                seq = None
                if isinstance(entry, list):
                    seq = entry
                elif isinstance(entry, dict):
                    for sk, sv in entry.items():
                        if isinstance(sv, list) and len(sv) > 1:
                            seq = sv
                            break
                if seq is None:
                    continue
                T = int(np.load(os.path.join(I3D_TEST, f"{v}_x264_i3d.npy"),
                                mmap_mode="r", allow_pickle=False).shape[0])
                d0 = len(seq) - T
                agree_t += d0 == 0
                agree_t1 += d0 == -1
                print(f"    {v:<22} I3D T={T:<5} GT len={len(seq):<5} "
                      f"diff={d0}")
            print(f"  matches T: {agree_t}/8   matches T-1: {agree_t1}/8")

    manifest["labels"] = {
        "n_videos": len(label_of), "n_anomalous": n_anom,
        "class_dirs": sorted(set(class_of.values())),
    }
    manifest["split"] = {
        "n_train": len(train), "n_test": len(test), "overlap": len(both),
        "n_unlabelled": len(missing), "unlabelled_examples": missing[:20],
    }
    manifest["ground_truth"] = gt_info
    manifest["per_video"] = {
        v: {"label": label_of.get(v, -1), "class": class_of.get(v, "?"),
            "split": split_of.get(v, "missing")}
        for v in sorted(set(label_of) | set(split_of))
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "ucf_label_manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {out}")

    # Console digest of the per-split label balance, which is what pooled AUC
    # depends on most.
    print("\n=== label balance per split ===")
    for sp in ("train", "test"):
        vs = [v for v in split_of if split_of[v] == sp and v in label_of]
        a = sum(label_of[v] for v in vs)
        print(f"  {sp:<6} videos={len(vs):<5} anomalous={a:<5} "
              f"normal={len(vs) - a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

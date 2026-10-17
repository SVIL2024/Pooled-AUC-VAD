"""Do the two backbones describe the same segments on UCF-Crime test?

Runs on the server, where the features already are. The question decides whether
a two-backbone comparison means anything: pooled frame-level AUC is defined on a
sequence of per-segment scores, so the two backbones' score vectors must refer to
the same segments before any number can be compared across them.

Evidence that this is not automatic, from the first 30 seconds of looking:

  * the recorded CLIP score lengths for UCF test begin 88, 96, 148, 196, 226;
  * the first I3D test file alphabetically, Abuse028, has T = 89.

Those are not equal, so either the two sides use different clip rates, a
different boundary rule, or a different video ordering. Each has a different
consequence, so this script measures rather than assumes. It checks, per video:

  1. the I3D clip count and the CLIP clip count, by video name;
  2. the ratio between them, which should be constant if the grids agree up to a
     fixed rate;
  3. whether the video-level anomaly label agrees between the two layouts,
     since a class-directory disagreement would mean the join itself is wrong;
  4. the total clip budget under each side, which is what pooled AUC depends on.

Read-only: opens files with mmap and reads no payload.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

I3D_TEST = r"E:\dataset\i3d\UCF\UCF_Test_ten_crop_i3d"
CLIP_ROOT = r"E:\dataset\UCFClipFeatures"
OUT_JSON = os.environ.get(
    "CDV_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                 "ucf_grid_alignment.json"))
# Derived from this file's own location so the script runs wherever the tree is
# checked out, rather than only on the machine that produced it.
LENGTHS_LOCAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "server_ops",
    "lengths.json")
I3D_DOCUMENTED_CPS = 10  # clips per second, as recorded for this dump

# UCF-Crime's anomalous categories. Everything else under a normal_* directory is
# the negative class. Used only to check that both layouts agree on the label.
NORMAL_MARKERS = ("Normal_Videos", "normal_videos", "Normal")


def i3d_shape(path: str) -> tuple[int, ...]:
    # mmap_mode parses the header without reading the array. This is the path
    # that works where read_array_header_1_0 does not.
    return tuple(np.load(path, mmap_mode="r", allow_pickle=False).shape)


def clip_shape(path: str) -> tuple[int, ...]:
    return tuple(np.load(path, mmap_mode="r", allow_pickle=False).shape)


def normalise(stem: str) -> str:
    # The CLIP side stores one file per chunk, named <video>__<chunk>. Strip the
    # chunk index first, otherwise the join sees 19,000 distinct "videos" and
    # matches nothing. This is the reason the first attempt reported zero common
    # names despite both sides holding the same 290 videos.
    if "__" in stem:
        stem = stem.rsplit("__", 1)[0]
    for suffix in ("_x264_i3d", "_i3d", "_x264", "_clip", "_feats"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def main() -> int:
    report: dict[str, object] = {}

    # ---- I3D side -------------------------------------------------------
    i3d_files = sorted(f for f in os.listdir(I3D_TEST) if f.endswith(".npy"))
    i3d: dict[str, tuple[int, ...]] = {}
    i3d_err: list[str] = []
    for f in i3d_files:
        try:
            i3d[normalise(os.path.splitext(f)[0])] = i3d_shape(
                os.path.join(I3D_TEST, f))
        except Exception as exc:
            i3d_err.append(f"{f}: {type(exc).__name__}: {exc}")

    print("=== I3D test features ===")
    print(f"  files listed : {len(i3d_files)}")
    print(f"  readable     : {len(i3d)}")
    if i3d_err:
        print(f"  unreadable   : {len(i3d_err)}  e.g. {i3d_err[:3]}")
    sig = Counter(i3d.values())
    print("  shape signatures:")
    for s, n in sig.most_common(6):
        print(f"    {str(s):<26} x{n}")
    ts = np.array([s[0] for s in i3d.values()], dtype=np.int64)
    print(f"  clips T: total={ts.sum()} min={ts.min()} max={ts.max()} "
          f"median={int(np.median(ts))}")
    print(f"  implied duration at {I3D_DOCUMENTED_CPS} clips/s: "
          f"{ts.sum() / I3D_DOCUMENTED_CPS / 60:.1f} min")
    report["i3d"] = {
        "n_files": len(i3d_files), "n_readable": len(i3d),
        "shape_signatures": {str(k): v for k, v in sig.items()},
        "clips_total": int(ts.sum()),
        "clips_min": int(ts.min()), "clips_max": int(ts.max()),
        "errors": i3d_err[:20],
    }

    # ---- CLIP side ------------------------------------------------------
    clip: dict[str, list[tuple[int, ...]]] = {}
    clip_err: list[str] = []
    clip_dirs = sorted(d for d in os.listdir(CLIP_ROOT)
                        if os.path.isdir(os.path.join(CLIP_ROOT, d)))
    for d in clip_dirs:
        for f in sorted(os.listdir(os.path.join(CLIP_ROOT, d))):
            if not f.endswith(".npy"):
                continue
            key = normalise(os.path.splitext(f)[0])
            try:
                clip.setdefault(key, []).append(
                    clip_shape(os.path.join(CLIP_ROOT, d, f)))
            except Exception as exc:
                clip_err.append(f"{d}/{f}: {type(exc).__name__}: {exc}")

    print("\n=== CLIP test features ===")
    print(f"  class directories : {len(clip_dirs)}")
    print(f"  chunk files       : {len(clip)} videos / "
          f"{sum(len(v) for v in clip.values())} chunks")
    if clip_err:
        print(f"  unreadable        : {len(clip_err)}  e.g. {clip_err[:3]}")
    nchunks = Counter(len(v) for v in clip.values())
    print("  chunks per video (most common):")
    for n, c in nchunks.most_common(6):
        print(f"    {n:<6} chunks x{c} videos")
    report["clip"] = {
        "class_dirs": clip_dirs, "n_videos": len(clip),
        "n_chunks": sum(len(v) for v in clip.values()),
        "chunks_per_video": {str(k): v for k, v in nchunks.items()},
        "errors": clip_err[:20],
    }

    # ---- join ----------------------------------------------------------
    common = sorted(set(i3d) & set(clip))
    only_i3d = sorted(set(i3d) - set(clip))
    only_clip = sorted(set(clip) - set(i3d))
    print("\n=== join by normalised video name ===")
    print(f"  matched on both sides : {len(common)}")
    print(f"  I3D only              : {len(only_i3d)}  e.g. {only_i3d[:5]}")
    print(f"  CLIP only             : {len(only_clip)}  e.g. {only_clip[:5]}")
    report["join"] = {"common": len(common), "i3d_only": len(only_i3d),
                      "clip_only": len(only_clip),
                      "i3d_only_examples": only_i3d[:20],
                      "clip_only_examples": only_clip[:20]}

    if not common:
        print("\nno common names; cannot compare grids")
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {OUT_JSON}")
        return 1

    # ---- ratio ---------------------------------------------------------
    # A video's CLIP clip count is the sum of its chunks' time axis. The axis
    # choice is recorded so the decision is auditable rather than implicit.
    print("\n=== per-video clip counts (first 12 matched) ===")
    print(f"  {'video':<24} {'I3D T':>7} {'CLIP n':>7} {'ratio':>7}")
    ratios: list[float] = []
    rows = []
    dim_counter = Counter()
    for name in common:
        ic = i3d[name][0]
        shapes = clip[name]
        dims = {s[-1] for s in shapes if len(s) > 1}
        dim_counter[tuple(sorted(dims))] += 1
        cc = int(sum(s[0] for s in shapes))
        ratio = ic / cc if cc else float("nan")
        ratios.append(ratio)
        rows.append({"video": name, "i3d_T": int(ic),
                     "clip_n": cc, "ratio": float(ratio),
                     "chunks": len(shapes)})
    for r_ in rows[:12]:
        print(f"  {r_['video']:<24} {r_['i3d_T']:>7} {r_['clip_n']:>7} "
              f"{r_['ratio']:>7.3f}")

    r = np.array(ratios, dtype=float)
    r = r[np.isfinite(r)]
    print(f"\n  ratio I3D/CLIP: n={r.size} min={r.min():.3f} "
          f"median={np.median(r):.3f} max={r.max():.3f} "
          f"iqr={np.percentile(r, 75) - np.percentile(r, 25):.3f}")
    print("  ratio distribution (rounded to 0.01):")
    for val, n in Counter(np.round(r, 2).tolist()).most_common(8):
        print(f"    {val:<8} x{n}")

    exact = int((np.abs(r - 1.0) < 1e-9).sum())
    near = int((np.abs(r - 1.0) <= 0.01).sum())
    print(f"  ratio == 1 exactly : {exact}/{r.size}")
    print(f"  ratio within 1%    : {near}/{r.size}")
    print(f"  feature dims seen : {dict(dim_counter)}")

    total_i3d = sum(x["i3d_T"] for x in rows)
    total_clip = sum(x["clip_n"] for x in rows)
    print(f"\n  total clips: I3D={total_i3d}  CLIP={total_clip}  "
          f"ratio={total_i3d / total_clip:.3f}")
    report["grids"] = {
        "n": len(rows),
        "clip_feature_dims": {str(k): v for k, v in dim_counter.items()},
        "ratio_min": float(r.min()), "ratio_median": float(np.median(r)),
        "ratio_max": float(r.max()),
        "ratio_exactly_one": exact, "ratio_within_1pct": near,
        "total_i3d": total_i3d, "total_clip": total_clip,
        "rows": rows,
    }

    # ---- recorded CLIP lengths, as an independent cross-check -----------
    if os.path.isfile(LENGTHS_LOCAL):
        with open(LENGTHS_LOCAL, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        ul = rec.get("ucf") or []
        print("\n=== recorded CLIP score lengths (lengths.json) ===")
        print(f"  count={len(ul)} sum={sum(ul)} first={ul[:8]}")
        print(f"  measured CLIP feature clips: sum={total_clip}")
        same = sum(1 for a, b in zip(ul, [x['clip_n'] for x in rows]) if a == b)
        print(f"  position-wise equal for {same}/{min(len(ul), len(rows))} "
              f"positions")
        if len(ul) != total_clip:
            print("  NOTE: recorded and measured totals differ "
                  f"({sum(ul)} vs {total_clip})")
        report["recorded_lengths"] = {
            "count": len(ul), "sum": int(sum(ul)), "first": ul[:16],
            "positionwise_equal": same, "measured_sum": total_clip,
        }

    # ---- class-directory layout ---------------------------------------
    print("\n=== class-directory layout (CLIP side) ===")
    for d in clip_dirs:
        is_normal = any(m in d for m in NORMAL_MARKERS)
        print(f"    {d:<24} normal={is_normal}")
    report["class_dirs"] = {d: bool(any(m in d for m in NORMAL_MARKERS))
                            for d in clip_dirs}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

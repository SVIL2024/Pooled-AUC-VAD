"""Cross-check the two backbones' temporal grids against the paper's own record.

The server-side probe established that the I3D dump carries one time step per
second while the CLIP dump carries ten, so a two-backbone number is only
computable after resampling onto one grid. This decides which grid to resample
onto, by comparing the paper's recorded score lengths against both feature sets.

It also isolates the videos whose ratio departs from the modal 0.100. Those are
not noise to be averaged away: a video where the two dumps cover different spans
of footage cannot contribute a paired comparison, and silently pooling it in
would corrupt every aggregate reported later.

Reads only the archived probe output and the recorded lengths. No network.
"""

from __future__ import annotations

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ALIGN = os.path.join(ROOT, "results", "ucf_grid_alignment.json")
LENGTHS = os.path.join(ROOT, "server_ops", "lengths.json")
OUT = os.path.join(ROOT, "results", "ucf_grid_verdict.json")


def main() -> int:
    with open(ALIGN, "r", encoding="utf-8") as fh:
        al = json.load(fh)
    with open(LENGTHS, "r", encoding="utf-8") as fh:
        rec = json.load(fh)

    rows = al["grids"]["rows"]
    print(f"=== joined videos: {len(rows)} ===")
    print(f"  total I3D clips : {al['grids']['total_i3d']}")
    print(f"  total CLIP clips: {al['grids']['total_clip']}")
    print(f"  CLIP feature dim: {list(al['grids']['clip_feature_dims'])}")

    # ---- which grid does the paper actually score on? ------------------
    # lengths.json is an ordered list, so test the orderings rather than
    # assuming the file order matches the alphabetical join order.
    ul = rec["ucf"]
    print("\n=== paper's recorded score lengths for UCF test ===")
    print(f"  count={len(ul)} sum={sum(ul)} first={ul[:10]}")

    names_sorted = [r["video"] for r in rows]
    i3d_by = {r["video"]: r["i3d_T"] for r in rows}
    clip_by = {r["video"]: r["clip_n"] for r in rows}

    print("\n=== do the recorded lengths match the alphabetical join order? ===")
    if len(ul) == len(names_sorted):
        diff_i3d = [ul[k] - i3d_by[names_sorted[k]] for k in range(len(ul))]
        diff_clip = [ul[k] - clip_by[names_sorted[k]] / 10.0
                     for k in range(len(ul))]
        c_i3d = Counter(diff_i3d)
        c_clip = Counter(round(d, 3) for d in diff_clip)
        print(f"  recorded - I3D_T      : {c_i3d.most_common(6)}")
        print(f"  recorded - CLIP_n/10  : {c_clip.most_common(6)}")
    else:
        print(f"  length mismatch: recorded {len(ul)} vs joined {len(names_sorted)}")

    # ---- outliers ------------------------------------------------------
    print("\n=== videos whose I3D:CLIP ratio departs from 0.100 ===")
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(round(r["ratio"], 2)), []).append(r)
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        g = groups[key]
        names = [x["video"] for x in g]
        print(f"\n  ratio {key}  n={len(g)}")
        print(f"    classes: {Counter(n.split('0')[0].rstrip('0123456789') for n in names).most_common()}")
        print(f"    e.g. {names[:8]}")
        if len(g) <= 60:
            print(f"    all: {names}")

    # ---- a defensible subset ------------------------------------------
    # Keep only videos where the two dumps agree on coverage. The tolerance is
    # half a time step at the coarser of the two grids, which is the largest
    # disagreement that can be attributed to the boundary convention alone.
    print("\n=== paired subset where the grids agree ===")
    for tol in (0.0, 0.005, 0.01, 0.05):
        keep = [r for r in rows if abs(r["ratio"] - 0.100) <= tol]
        tot_i = sum(r["i3d_T"] for r in keep)
        tot_c = sum(r["clip_n"] for r in keep)
        print(f"  |ratio-0.100| <= {tol:<6} : {len(keep):>3}/290 videos, "
              f"I3D clips={tot_i}, CLIP clips={tot_c}, "
              f"ratio={tot_i / tot_c:.4f}")

    keep = [r for r in rows if abs(r["ratio"] - 0.100) <= 0.01]
    verdict = {
        "joined": len(rows),
        "modal_ratio": 0.100,
        "ratios": {k: len(v) for k, v in groups.items()},
        "paired_tolerance_0.01": {
            "n": len(keep),
            "total_i3d_clips": sum(r["i3d_T"] for r in keep),
            "total_clip_clips": sum(r["clip_n"] for r in keep),
            "videos": [r["video"] for r in keep],
        },
        "excluded": [{"video": r["video"], "ratio": r["ratio"],
                      "i3d_T": r["i3d_T"], "clip_n": r["clip_n"]}
                     for r in rows if abs(r["ratio"] - 0.100) > 0.01],
        "i3d_clips_per_second": 1,
        "clip_clips_per_second": 10,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

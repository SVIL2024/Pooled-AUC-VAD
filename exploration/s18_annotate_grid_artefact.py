"""Annotate the raw grid-alignment record with the artefact it contains.

This file is a raw probe output and its numbers stay as they were recorded. What
changed is how they read: clip_n is the sum over all ten spatial crops of a
video, so the ratio against the I3D clip count sits near 0.1 by construction and
must not be read as a clip-rate difference. The ten crops share one temporal
axis -- verified 1900/1900 -- and against a single crop the same comparison gives
1.0038 with an offset of at most one clip.

The numbers are left untouched because editing an archived measurement to make it
agree with a later one is the failure mode this project keeps auditing for. A
pointer is added instead, so the record is not silently misleading.
"""

from __future__ import annotations

import json
import os

TARGETS = [
    r"D:\program\vad-cross-domain\results\second_backbone\ucf_grid_alignment.json",
    r"D:\program\vad-cross-domain\results\ucf_grid_alignment.json",
]

CAVEAT = {
    "artefact_in_this_file": (
        "clip_n in grids.rows is the SUM over all ten spatial crops of a "
        "video, and i3d_T is a single crop's clip count. The ratio near 0.1 "
        "and the 51-video exclusion derived from it are both artefacts of that "
        "sum. The ten files <video>__0..__9 are ten spatial crops of one clip "
        "sequence and share a single temporal axis."),
    "correct_comparison": (
        "Compare i3d_T against ONE crop's row count. Result: 290/290 videos "
        "join, 1900/1900 crops agree on clip count, offset (I3D - CLIP) is 1 "
        "for 266 videos and 0 for 24, total clips 69,634 against 69,368, ratio "
        "1.003835. There is no clip-rate difference between the two dumps and no "
        "video needs excluding for a grid mismatch."),
    "superseded_by": "results/ucf_grid_verdict.json",
    "values_left_unchanged": (
        "Deliberately. This file is a raw record; editing it to agree with a "
        "later analysis would destroy the audit trail."),
    "produced_by": "ops/s7_ucf_grid_check.py (superseded by "
                   "ops/s17_ucf_grid_corrected.py)",
}


def main() -> int:
    for path in TARGETS:
        if not os.path.isfile(path):
            print(f"  absent, skipped: {path}")
            continue
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if "artefact_in_this_file" in d:
            print(f"  already annotated: {os.path.basename(path)}")
            continue
        # Insert the caveat first so a reader sees it before the numbers.
        ordered = dict(CAVEAT)
        ordered.update(d)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            json.dump(ordered, fh, indent=2)
        print(f"  annotated: {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

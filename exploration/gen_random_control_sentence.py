"""Generate the random-control sentence from the archive, as text.

Written after two successive manual transcriptions of these ranges turned out to
be wrong. Four numbers per configuration, two configurations each time, and the
paper had to be corrected twice. The sentence is now produced from
results/s6_sufficiency.json so it cannot drift from the measurement.

Prints a LaTeX fragment ready to paste, and checks it against the manuscript so
a stale sentence is caught rather than shipped.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os
import re

RES = os.path.join(REPO_ROOT, "results")
TEX = os.environ.get("POOLED_TEX", r"D:\paper\2026\Pooled-Frame-Level-AUC-Measures\Pooled Frame-Level AUC Measures.tex")
DELTAS = ["0.25", "0.5", "1.0", "2.0", "4.0"]
# the four single-checkpoint UCF-Crime / XD-Violence configurations, in the
# order Table 1 lists them
ORDER = ["A_ucf2xd", "C_xd2xd", "B_xd2ucf", "D_ucf2ucf"]
WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def main() -> int:
    s = json.load(open(os.path.join(RES, "s6_sufficiency.json"), encoding="utf-8"))

    spans = []
    for i, k in enumerate(ORDER, 1):
        vals = [round(float(s[k]["arms"][d]["random"]), 4) for d in DELTAS]
        spans.append((min(vals), max(vals)))

    clause = ", ".join(f"{lo:.4f} to {hi:.4f} on the {WORDS[i]} configuration"
                       for i, (lo, hi) in enumerate(spans, 1))
    sentence = (
        "stays in a narrow band far below the aligned values throughout: "
        f"across the sweep it spans {clause}. The\nrecovery is specific to "
        "alignment between the offset and the video label."
    )

    print("=== generated sentence ===")
    print(sentence)

    # Verify by value, not by matching the generated string. Matching whole
    # sentences is fragile because line breaks in the .tex differ from the
    # wrapped output here, and an earlier version of this check reported a
    # disagreement that was only a line-wrap difference.
    try:
        tex = open(TEX, encoding="utf-8").read()
    except OSError as exc:
        print(f"\ncould not read the manuscript: {exc}")
        return 1

    print()
    ok = True
    for i, (lo, hi) in enumerate(spans, 1):
        pair = f"{lo:.4f} to {hi:.4f} on the {WORDS[i]} configuration"
        good = pair in re.sub(r"\s+", " ", tex)
        ok &= good
        print(f"  {'ok ' if good else 'MISSING'}  {pair}")

    if ok:
        print("\nMANUSCRIPT AGREES with the archive on all four ranges")
        return 0
    print("\nMANUSCRIPT DISAGREES -- replace the stale ranges with the ones above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
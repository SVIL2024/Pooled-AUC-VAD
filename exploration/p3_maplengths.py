"""Map each P-3 config tag to its target domain's clip-length list.

Each score archive only records a flat score vector, but the within-video AUC
needs true video boundaries, which come from the target's test list.  The tag
names encode the target (``*_ucf2xd`` -> target ``xd``), so the mapping is
derived from the tag rather than hardcoded.
"""
from __future__ import annotations

import argparse
import json
import os
import re

# A config dir is named like "<TAG>_<src>2<tgt>" (our own naming); the target
# is the part after the '2'.  Fall back to the npz filename when needed.
PAT_DIR = re.compile(r"_(\w+)2(\w+)$")
PAT_NPZ = re.compile(r"cdv_scores_(\w+)2(\w+)\.npz$")


def target_of(d: str, f: str):
    m = PAT_NPZ.search(f) or PAT_DIR.search(d)
    return m.group(2) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.lengths) as f:
        by_domain = json.load(f)

    out, unresolved = {}, []
    for d in sorted(os.listdir(args.root)):
        p = os.path.join(args.root, d)
        if not os.path.isdir(p):
            continue
        npz = [x for x in os.listdir(p) if x.endswith(".npz")]
        if not npz:
            continue
        tgt = target_of(d, npz[0])
        if tgt is None or tgt not in by_domain:
            unresolved.append(d)
            continue
        out[d] = by_domain[tgt]
        print(f"[map] {d:20s} -> target={tgt} videos={len(out[d])}")

    if unresolved:
        print(f"[warn] could not resolve target for: {unresolved}")
    if not out:
        raise SystemExit("no config mapped -- check the tag naming")

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
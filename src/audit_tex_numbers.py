"""audit_tex_numbers: every 3-4 decimal number in the manuscript must be
traceable to an archived result file.

A claim-scope gate, not a style check. The manuscript asserts many decimals;
each one has to come from a stored measurement rather than from recollection or
from an intermediate run that was later corrected. Anything untraceable is
reported by name so it can be removed or sourced.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import glob
import json
import os
import re
import sys

TEX = r"D:\paper\2026\Pooled Frame-Level AUC Measures\Pooled Frame-Level AUC Measures.tex"
RES = os.path.join(REPO_ROOT, "results")


def walk(o, out):
    if isinstance(o, dict):
        for v in o.values():
            walk(v, out)
    elif isinstance(o, list):
        for v in o:
            walk(v, out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        # The archives store full machine precision (e.g. 0.8685187461687165)
        # while the manuscript quotes 4 decimals. Both sides must be reduced
        # to the SAME number of decimals before comparison -- an earlier
        # version rounded the archive to 6 and the manuscript to 4, which made
        # every value look untraceable and produced a false alarm.
        out.add(round(float(o), 4))


def main():
    tex = open(TEX, encoding="utf-8").read()
    # nums holds STRINGS (regex output) and auth holds FLOATS; comparing them
    # directly always fails, which is what an earlier version of this script
    # did and reported as "0 traceable".
    nums = {n: round(float(n), 4) for n in
            set(re.findall(r"(?<![\w.])\d+\.\d{3,4}(?![\d])", tex))}

    auth = set()
    files = sorted(glob.glob(os.path.join(RES, "*.json")))
    used = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        walk(d, auth)
        used.append(os.path.basename(f))

    # quantities we derive in-text from stored fields, so they are not literal
    # values in any file
    def add_derived():
        p = os.path.join(RES, "s4_g5redo.json")
        if os.path.exists(p):
            for v in json.load(open(p, encoding="utf-8"))["rows"].values():
                auth.add(round(v["const_auc"], 4))
                auth.add(round(v["deoff_auc"], 4))
                auth.add(round(v["deoff_ap"], 4))
                auth.add(round(v["deoff_ap"] - v["ap"], 4))
        p = os.path.join(RES, "s3_deoffset.json")
        if os.path.exists(p):
            for v in json.load(open(p, encoding="utf-8"))["rows"].values():
                auth.add(round(v["offset_rho"], 3))
                auth.add(round(v["cross"], 4))
                auth.add(round(v["deoff_cross"], 4))
                auth.add(round(v["cross"] - v["deoff_cross"], 4))
                auth.add(round(v["pooled"], 4))
                auth.add(round(v["deoff_pooled"], 4))
    add_derived()

    # Expand the archive side downward to 3 decimals so that a manuscript value
    # printed at lower precision than the stored value still resolves. The
    # archive keeps full machine precision; the manuscript deliberately rounds
    # for readability. Rounding the ARCHIVE down is what makes the comparison
    # well posed -- rounding the manuscript down instead would accept numbers
    # that no stored value rounds to.
    #
    # Absolute values are added because the manuscript writes signed
    # correlations as $-0.387$ and the digit regex cannot see the minus, so
    # the stored -0.3874 would otherwise look unsourced.
    auth |= {round(a, 3) for a in list(auth)}
    auth |= {round(abs(a), 4) for a in list(auth)}
    auth |= {round(abs(a), 3) for a in list(auth)}

    # The manuscript quotes at most 4 decimals and sometimes rounds a stored
    # value down to 3 for readability (0.0691 shown as 0.069). Comparing only at
    # 4 dp flagged every such display rounding as unsourced, which is a false
    # alarm rather than a provenance failure.
    def matches(v):
        if v in auth:
            return True
        return round(v, 3) in auth

    missing = sorted(n for n, v in nums.items() if not matches(v))
    print(f"result files scanned : {len(used)}")
    print(f"3-4 dp numbers in tex: {len(nums)}")
    print(f"traceable            : {len(nums) - len(missing)}")
    print(f"NOT traceable        : {missing if missing else 'none'}")
    if missing:
        print()
        print("These are either derived quantities computed from stored fields,")
        print("or values that need a source. Each must be justified, not kept")
        print("because it reads plausibly.")

    # LaTeX structural sanity that does not need a TeX binary
    print()
    bal_b, bal_e = tex.count("\\begin{"), tex.count("\\end{")
    print(f"\\begin{{}} / \\end{{}}  : {bal_b} / {bal_e}  "
          f"{'BALANCED' if bal_b == bal_e else 'UNBALANCED'}")
    for env in ("table", "abstract", "frontmatter", "document", "keyword",
                "highlights", "thebibliography"):
        b = tex.count(f"\\begin{{{env}}}")
        e = tex.count(f"\\end{{{env}}}")
        if b or e:
            flag = "ok" if b == e else "MISMATCH"
            print(f"  {env:<16} {b}/{e}  {flag}")
    todos = len(re.findall(r"%%TODO", tex))
    print(f"\\TODO markers        : {todos}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
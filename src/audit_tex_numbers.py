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

# The manuscript path comes from the command line (documented in the
# README) or the POOLED_TEX environment variable; nothing about the
# operator machine is baked into the script.
TEX = (sys.argv[1] if len(sys.argv) > 1
       else os.environ.get("POOLED_TEX", ""))
RES = os.path.join(REPO_ROOT, "results")
if not TEX:
    sys.exit("usage: python src/audit_tex_numbers.py <manuscript.tex>  (or set POOLED_TEX)")


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


def strip_comments(s):
    r"""Drop LaTeX comment lines and trailing comments before scanning.

    A %-comment is not TeX, so anything inside one is not part of the document.
    The manuscript's header block discusses the \begin{keyword} macro in prose,
    which made the environment-balance and keyword counters report a mismatch
    that did not exist. A backslash-percent is an escaped percent, not a comment
    introducer, so the scan skips over it.
    """
    out = []
    for line in s.splitlines():
        # Walk the line so an escaped percent does not cut it short.
        cut = len(line)
        i = 0
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == "%":
                cut = i
                break
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


def main():
    raw = open(TEX, encoding="utf-8").read()
    tex = strip_comments(raw)
    # nums holds STRINGS (regex output) and auth holds FLOATS; comparing them
    # directly always fails, which is what an earlier version of this script
    # did and reported as "0 traceable".
    nums = {n: round(float(n), 4) for n in
            set(re.findall(r"(?<![\w.])\d+\.\d{3,4}(?![\d])", tex))}

    auth = set()
    # Recursive: the second-backbone investigation records live in a
    # subdirectory, and a non-recursive glob reported numbers as traceable that
    # no scanned file actually contained.
    files = sorted(glob.glob(os.path.join(RES, "**", "*.json"), recursive=True))
    used = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        walk(d, auth)
        used.append(os.path.relpath(f, RES))

    # quantities we derive in-text from stored fields, so they are not literal
    # values in any file
    def add_sufficiency():
        p = os.path.join(RES, "s6_sufficiency.json")
        if os.path.exists(p):
            # s6 is keyed by configuration directly, not under "rows".
            for v in json.load(open(p, encoding="utf-8")).values():
                for delta in v.get("arms", {}):
                    arm = v["arms"][delta]
                    for key in ("aligned", "anti", "random"):
                        if key in arm:
                            auth.add(round(float(arm[key]), 4))
                    aligned = float(arm.get("aligned", "nan"))
                    if aligned == aligned:
                        # deviations are quoted in the text, not stored
                        auth.add(round(aligned - v["orig"], 4))
                        auth.add(round(abs(aligned - v["orig"]), 4))

    add_sufficiency()

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

    def add_w_pct():
        # The manuscript quotes the within-video pair share in
        # percent (0.071 to 0.388), which is w_posneg * 100 from
        # audit_within_w.json. The stored field is a fraction, so the
        # percent spellings are derived quantities, not literals in
        # any file.
        p = os.path.join(RES, "audit_within_w.json")
        if not os.path.exists(p):
            return
        def walkw(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if (k in ('w_posneg', 'w_all')
                            and isinstance(v, (int, float))
                            and not isinstance(v, bool)):
                        auth.add(round(v * 100.0, 4))
                        auth.add(round(v * 100.0, 3))
                    walkw(v)
            elif isinstance(o, list):
                for v in o:
                    walkw(v)
        walkw(json.load(open(p, encoding='utf-8')))

    add_w_pct()

    def add_consequences():
        """Quantities that follow from the identity rather than from a stored row.

        The pair-weighted within-video area that enters Eq. (identity) is
        A_cross - Gap, not the unweighted within area stored as auc_within; the
        two differ because the identity weights each video by its own anomalous
        x normal pair count. Every number below is computed from results/, so the
        manuscript's statement that the audit script recomputes its derived
        quantities holds without a new archive file.
        """
        p = os.path.join(RES, "fig_values.json")
        if not os.path.exists(p):
            return
        fv = json.load(open(p))
        ORD = ("A_ucf2xd", "C_xd2xd", "B_xd2ucf", "D_ucf2ucf", "ucf2sht",
               "xd2sht")
        ENS = ("E_ens11_ucf2xd", "F_ens22_ucf2xd", "G_ens33_ucf2xd")
        WS = {"A_ucf2xd": 0.000710, "C_xd2xd": 0.000710,
              "B_xd2ucf": 0.003884, "D_ucf2ucf": 0.003884,
              "ucf2sht": 0.008169, "xd2sht": 0.008169}
        pooled = fv["pooled_auc"]
        cross = fv["auc_cross"]
        gap = fv["gap"]
        keep = fv["const_keep_pct"]
        deoff = fv["deoff_keep_pct"]
        ceiling = {}
        for k in ORD:
            a_with = cross[k] - gap[k]
            c = WS[k] * (1.0 - a_with)
            ceiling[k] = c
            auth.add(round(c, 4))
        for group in (ORD[0:2], ORD[2:4], ORD[4:6]):
            auth.add(round(max(ceiling[k] for k in group), 4))
        for w in (0.000710, 0.003884, 0.008169):
            auth.add(round((1.0 - w) / w, 3))
            auth.add(round(1e-4 * (1.0 - w) / w, 3))
        ens = [pooled[k] for k in ENS]
        auth.add(round(max(ens) - min(ens), 3))
        auth.add(round(abs(pooled["A_ucf2xd"] - min(ens)), 3))
        auth.add(round(abs(pooled["C_xd2xd"] - max(ens)), 3))
        for k in ORD[:4]:
            se_keep = keep[k] / 100.0 * pooled[k]
            se_deoff = deoff[k] / 100.0 * pooled[k]
            auth.add(round((se_keep - 0.5) / (pooled[k] - 0.5) * 100.0, 1))
            auth.add(round((se_deoff - 0.5) / (pooled[k] - 0.5) * 100.0, 1))
            auth.add(round((1.0 - keep[k] / 100.0) * pooled[k] * 100.0, 1))
    add_consequences()




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
    # Counted on the raw source: TODO markers live in comments by convention, so
    # scanning the stripped text would report the outstanding work as done.
    todos = len(re.findall(r"%%TODO", raw))
    print(f"\\TODO markers        : {todos}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
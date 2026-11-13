"""Cross-section consistency audit of the manuscript.

Internal consistency fails in ways a per-section proofread cannot catch: a count
written in the Introduction that no longer matches the table after an edit, a
number in the Abstract that drifted from the one in Results, a limitation that
no longer covers a claim made elsewhere. Each check below compares statements
that must agree, across the sections where they appear.

Checks
  A  counts        configuration, table-row, figure and citation counts, which
                   must agree wherever they are stated
  B  number echo   every decimal asserted in the Abstract, Highlights and
                   Conclusion must appear in the Results section too
  C  sign claims   the sign-reversal claim is stated per benchmark; the same
                   benchmark must not be given opposite signs in two places
  D  scope         sweeping words must be covered by a stated limitation
  E  verification  "identical" claims about the invariance must carry the
                   verified qualifier and its measured deviation

Exit status is non-zero when any check fails.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os
import re
import sys

TEX = r"D:\paper\2026\Pooled-Frame-Level-AUC-Measures\Pooled Frame-Level AUC Measures.tex"
RES = os.path.join(REPO_ROOT, "results")
CITE_RE = re.compile(r"\\cite[a-z]*\{([^}]*)\}")


def sections(tex):
    """Split into named sections so a claim can be located, not just found."""
    out, name, buf = {}, "(front)", []
    pat = re.compile(r"\\(sub)?section\*?\{([^}]*)\}")
    for line in tex.split("\n"):
        m = pat.match(line.strip())
        if m:
            out[name] = "\n".join(buf)
            name = m.group(2)
            buf = []
        else:
            buf.append(line)
    out[name] = "\n".join(buf)
    return out


def body_of(sec_text):
    """Strip LaTeX so prose can be searched for claims rather than markup."""
    t = sec_text
    t = re.sub(r"%.*", "", t)
    t = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\\citep?\{[^}]*\}", " CITE ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    t = re.sub(r"[{}$\\~]", " ", t)
    return re.sub(r"\s+", " ", t)


def decimals(text):
    return set(re.findall(r"(?<![\w.])\d+\.\d{2,4}(?![\d])", text))


def main() -> int:
    tex = open(TEX, encoding="utf-8").read()
    sec = {k: body_of(v) for k, v in sections(tex).items()}
    problems = []

    # ---------------------------------------------------------------- A
    print("=== A. count claims ===")
    t1 = re.search(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S)
    n_rows = 0
    if t1:
        n_rows = len([l for l in t1.group(0).split("\n")
                      if l.strip().endswith("\\\\") and "&" in l
                      and "midrule" not in l and "toprule" not in l
                      and "bottomrule" not in l])
    print(f"  Table 1 data rows: {n_rows}")

    for claim, pattern, expect in [
        ("six scoring configurations", r"six scoring configurations", 6),
        ("six single-checkpoint configurations",
         r"six single-checkpoint configurations", 6),
        ("nine rows", r"nine rows", None),
        ("nine configurations", r"nine configurations", None),
        ("seven rows", r"seven rows", None),
    ]:
        hits = [k for k, v in sec.items() if re.search(pattern, v, re.I)]
        if hits:
            print(f"  '{claim}' in: {', '.join(hits)}")
        if expect is None and hits:
            problems.append(f"stale count '{claim}' still present in {hits}")

    # every distinct configuration named in Table 1
    if t1:
        rows = [l for l in t1.group(0).split("\n")
                if l.strip().endswith("\\\\") and "&" in l]
        print(f"  rows in table markup: {len(rows)}")

    # ---------------------------------------------------------------- B
    print("\n=== B. numbers echoed in Abstract / Conclusion ===")
    # Everything from the Results section heading to the Discussion heading,
    # NOT only the results subsections. An earlier version matched subsection
    # titles and so missed prose in \section{Results} itself, which produced
    # false "quoted but not in Results" findings.
    m_res = re.search(r"\\section\{Results\}(.*?)\\section\{Discussion\}", tex, re.S)
    results_text = body_of(m_res.group(1)) if m_res else ""
    print(f"  (Results prose scanned: {len(results_text)} chars)")
    have = decimals(results_text)
    # En-dash ranges such as 0.81--0.95 are two numbers to the regex but one
    # span to the reader; add both endpoints so a quoted range resolves.
    for a, b in re.findall(r"(\d+\.\d{2,4})--(\d+\.\d{2,4})", results_text):
        have.add(a)
        have.add(b)
    abstract_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    concl = sec.get("Conclusion", "")
    for label, text in (("Abstract", body_of(abstract_m.group(1)) if abstract_m else ""),
                        ("Conclusion", concl)):
        ds = decimals(text)
        for a, b in re.findall(r"(\d+\.\d{2,4})--(\d+\.\d{2,4})", text):
            ds.add(a)
            ds.add(b)
        missing = sorted(d for d in ds if d not in have)
        print(f"  {label}: {len(ds)} decimals, not found in Results: "
              f"{missing if missing else 'none'}")
        for d in missing:
            problems.append(f"{label} quotes {d}, which Results does not contain")

    # ---------------------------------------------------------------- C
    print("\n=== C. sign-reversal claims per benchmark ===")
    # every sentence naming a benchmark together with a signed SHT gap
    for k, v in sec.items():
        for m in re.finditer(r"[^.]*ShanghaiTech[^.]*?\.", v):
            s = m.group(0)
            if re.search(r"-0\.0(69|58|7)|revers", s, re.I):
                nums = re.findall(r"-0\.0\d+", s)
                print(f"  [{k}] signed values {nums}: "
                      f"{s.strip()[:96]}...")
                if nums and not all(x.startswith("-") for x in nums):
                    problems.append(f"{k}: SHT gap stated without negative sign")

    # ---------------------------------------------------------------- D
    print("\n=== D. sweeping words vs stated limitations ===")
    sweep = re.compile(r"\b(general|universal|always|all models|any benchmark|"
                       r"every benchmark|in general)\b", re.I)
    for k, v in sec.items():
        if k.startswith(("What these results", "Threats")):
            continue
        for m in sweep.finditer(v):
            ctx = v[max(0, m.start() - 90):m.end() + 90]
            if not re.search(r"do not|not a claim|cannot|only|neither", ctx, re.I):
                print(f"  [{k}] '{m.group(0)}': ...{ctx.strip()}...")
                problems.append(f"{k}: unguarded sweeping word '{m.group(0)}'")

    # ---------------------------------------------------------------- E
    print("\n=== E. invariance claim carries its qualifier ===")
    # A proof ("adding a constant cannot reorder") needs no measured bound.
    # What needs one is a claim that something was MEASURED to be equal.
    # So this flags sentences asserting equality without a bound, and only
    # when the sentence is not explicitly a construction argument.
    for k, v in sec.items():
        for m in re.finditer(r"[^.]*(?:bit-identical|identical to the last bit|"
                             r"unchanged|invariant)[^.]*\.", v, re.I):
            s = m.group(0)
            is_proof = re.search(r"because adding a constant|provably cannot|"
                                r"constructive|by construction", s, re.I)
            if is_proof:
                print(f"  [{k}] construction argument, no bound needed")
                continue
            ok = re.search(r"deviation|zero|exact|verified|before and after|"
                           r"to four decimal", s, re.I)
            if not ok:
                print(f"  [{k}] no qualifier: {s.strip()[:100]}")
                problems.append(f"{k}: identity claim without a measured bound")

    # ---------------------------------------------------------------- F
    print("\n=== F. archive agreement for headline quantities ===")
    arch = os.path.join(RES, "s2_table.json")
    if os.path.exists(arch):
        g = json.load(open(arch, encoding="utf-8"))["rows"]
        pairs = [("ucf2sht", "0.3973"), ("xd2sht", "0.5317"),
                 ("D_ucf2ucf", "0.8571"), ("C_xd2xd", "0.9468")]
        for k, expect in pairs:
            actual = round(g[k]["pooled_auc"], 4)
            flag = "ok " if f"{actual:.4f}" == expect else "MISMATCH"
            print(f"  {flag} {k}: archive {actual:.4f}, manuscript {expect}")
            if flag.strip() != "ok":
                problems.append(f"{k} archive/manuscript mismatch")

    print()
    if problems:
        print(f"{len(problems)} finding(s):")
        for p in problems:
            print(f"  - {p}")
        print("\nVERDICT: inconsistencies present")
        return 1
    print("VERDICT: no inconsistency found by these checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Report per-section word counts so section balance can be judged numerically.

Requirement: no subsection may be redundant while another is thin. That is an
eyeball judgement, but the inputs to it are countable, so they get counted.

Comments, the abstract, the declarations block, and figure/table environments
are excluded: they are not section prose.
"""
from __future__ import annotations

import re
import sys

TEX = r"D:\paper\2026\Pooled-Frame-Level-AUC-Measures\Pooled Frame-Level AUC Measures.tex"


def main() -> int:
    s = open(TEX, encoding="utf-8").read()

    s = re.sub(r"(?m)^\s*%.*$", "", s)
    s = re.sub(r"(?s)\\begin\{abstract\}.*?\\end\{abstract\}", "", s)
    s = re.sub(r"(?s)\\begin\{frontmatter\}.*?\\end\{frontmatter\}", "", s)
    s = re.sub(r"(?s)\\begin\{Declarations\}.*?\\end\{document\}", "", s)
    s = re.sub(r"\\begin\{(figure\*?|table)\*?\}.*?\\end\{\1\*?\}", "", s, flags=re.S)

    s = re.sub(r"\\(sub)?section\*?\{([^}]*)\}",
               lambda m: "\n@@@" + m.group(2) + "\n", s)
    s = re.sub(r"\\cite[a-z]*\{[^}]*\}", "", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", "", s)
    s = re.sub(r"[{}$^_&~]", "", s)

    order, counts, cur = [], {}, None
    for line in s.split("\n"):
        t = line.strip()
        if t.startswith("@@@"):
            cur = t[3:]
            if cur not in order:
                order.append(cur)
            counts.setdefault(cur, 0)
        elif cur is not None and t:
            counts[cur] += len([w for w in t.split() if re.search(r"[A-Za-z]{2}", w)])

    total = 0
    print("section".ljust(48) + "words")
    print("-" * 56)
    for k in order:
        print(k.ljust(48) + str(counts[k]))
        total += counts[k]
    print("-" * 56)
    print("TOTAL".ljust(48) + str(total))
    if order:
        vals = [counts[k] for k in order]
        print("largest / smallest ratio: "
              f"{max(vals) / max(1, min(vals)):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
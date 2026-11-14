"""Check the manuscript against Elsevier's Highlights requirement.

Elsevier's Guide for Authors requires a Highlights file of 3 to 5 bullets, each
at most 85 characters including spaces. This counts them, because a bullet one
character over is silently truncated or rejected by the submission system.

Reads the bullets from the .tex so the check cannot drift from what is actually
submitted.
"""
from __future__ import annotations

import re
import os
import sys

TEX = os.environ.get("POOLED_TEX", r"D:\paper\2026\Pooled-Frame-Level-AUC-Measures\Pooled Frame-Level AUC Measures.tex")
LIMIT = 85
MIN_BULLETS, MAX_BULLETS = 3, 5


def strip_tex(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)      # drop control words
    s = s.replace("\\%", "%").replace("--", "-")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("~", " ").replace("\\,", " ").replace("\\ ", " ")
    return " ".join(s.split())


def main() -> int:
    s = open(TEX, encoding="utf-8").read()
    m = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", s, re.S)
    if m is None:
        print("NO highlights ENVIRONMENT FOUND")
        return 1
    bullets = [strip_tex(b) for b in re.findall(r"\\item\s+(.*)", m.group(1))]

    ok = True
    print(f"highlights found : {len(bullets)}  "
          f"(Elsevier requires {MIN_BULLETS}-{MAX_BULLETS})")
    if not MIN_BULLETS <= len(bullets) <= MAX_BULLETS:
        ok = False
        print("  FAIL: wrong number of bullets")

    for i, b in enumerate(bullets, 1):
        n = len(b)
        flag = "ok  " if n <= LIMIT else "OVER"
        if n > LIMIT:
            ok = False
        print(f"  {i}. {n:>3}/{LIMIT} chars  {flag}  {b}")

    # The abstract has its own cap on this journal.
    a = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    if a:
        words = len([w for w in strip_tex(a.group(1)).split()
                     if re.search(r"[A-Za-z0-9]", w)])
        print(f"abstract words   : {words}  (Elsevier cap 250)")
        if words > 250:
            ok = False
            print("  FAIL: abstract over the 250-word cap")

    k = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", s, re.S)
    if k:
        nkw = len([x for x in k.group(1).split("\\sep") if x.strip()])
        print(f"keywords         : {nkw}  (Elsevier range 1-7)")
        if not 1 <= nkw <= 7:
            ok = False
            print("  FAIL: keyword count outside 1-7")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
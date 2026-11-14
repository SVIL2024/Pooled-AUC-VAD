"""Escape underscores that appear inside BibTeX note fields.

The archives under results/ and the entries added from publisher records both
carry explanatory `note = {...}` text, and some of those notes quote a DOI or an
identifier verbatim. BibTeX copies note text into the .bbl without any
processing, so an underscore reaches LaTeX as a raw subscript character and the
build dies with "Missing $ inserted" at the bibliography.

Fields such as `doi` and `url` are deliberately left alone: elsarticle-num routes
those through the \\path and \\href commands, which already treat their contents
verbatim, so escaping them would print a literal backslash.

Run once after editing references.bib. It is idempotent.
"""
from __future__ import annotations

import re
import os
import sys

BIB = os.environ.get("POOLED_BIB", r"D:\paper\2026\Pooled-Frame-Level-AUC-Measures\references.bib")

# A note field, possibly spanning lines, up to its balanced closing brace.
NOTE_RE = re.compile(r"note\s*=\s*\{", re.DOTALL)
ESCAPED_RE = re.compile(r"\\_")


def find_close(s: str, open_idx: int) -> int:
    """Index of the brace closing the one at open_idx, ignoring escaped ones."""
    depth = 0
    i = open_idx
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] in "{}_%&#$":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced note field")


def main() -> int:
    src = open(BIB, encoding="utf-8").read()
    out, pos, changed = [], 0, 0

    while True:
        m = NOTE_RE.search(src, pos)
        if m is None:
            out.append(src[pos:])
            break
        brace = m.end() - 1
        close = find_close(src, brace)
        body = src[brace + 1:close]
        if ESCAPED_RE.search(body):
            new = body
        else:
            new = re.sub(r"(?<!\\)_", r"\\_", body)
        if new != body:
            changed += 1
        out.append(src[pos:brace + 1])
        out.append(new)
        pos = close

    result = "".join(out)
    if result != src:
        open(BIB, "w", encoding="utf-8").write(result)
    print(f"note fields rewritten: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
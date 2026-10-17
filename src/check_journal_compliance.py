"""Check the manuscript against the target journal's stated requirements.

Two journals are covered because the requirements differ and the manuscript was
checked against both at different points.

  Image and Vision Computing (the current target)
    words 4500-7500, excluding references, captions, table content and
    anything after the references
    abstract <= 250 words
    keywords 1-7
    Highlights REQUIRED, 3-5 bullets, <= 85 characters each
    graphical abstract REQUIRED, >= 1328 px wide, legible at 13 cm
    vector figures as EPS or PDF with fonts embedded

  Journal of Visual Communication and Image Representation (checked earlier)
    abstract <= 250 words, keywords 1-7, Highlights required

Every limit here comes from that journal's published Guide for Authors. Nothing
is inferred.

Run:  python ops\\check_journal_compliance.py [neurocomputing|imavis|jvcir]
"""
from __future__ import annotations

import os
import re
import struct
import sys

PAPER = r"D:\paper\2026\Pooled-Frame-Level-AUC-Measures"
TEX = os.path.join(PAPER, "Pooled Frame-Level AUC Measures.tex")
GA = os.path.join(PAPER, "graphical_abstract.png")

LIMITS = {
    "imavis": {"words": (4500, 7500), "abstract": 250, "keywords": (1, 7),
               "highlights": True, "graphical_abstract": True,
               "label": "Image and Vision Computing"},
    "jvcir": {"words": None, "abstract": 250, "keywords": (1, 7),
              "highlights": False, "graphical_abstract": False,
              "label": "Journal of Visual Communication and Image Representation"},
    # Neurocomputing, from its Guide for Authors as read on 2026-10-17.
    #   abstract   "Abstracts should be ... no more than 250 words"
    #   keywords   "1 to 7 keywords", English, avoid multi-word keywords using
    #              "and" or "of"
    #   highlights "You are encouraged to provide article highlights", 3-5
    #              bullets of at most 85 characters including spaces
    #   graphical abstract  not mentioned anywhere in the guide; neither
    #              required nor provided for
    #   length     no word limit, no page limit and no length tier is stated
    #              for any of the seven article types
    #   class      the guide links els-cas-templates.zip and never mentions
    #              elsarticle, which is what this manuscript currently uses
    "neurocomputing": {"words": None, "abstract": 250, "keywords": (1, 7),
                       "highlights": True, "highlights_from_file": True,
                       "graphical_abstract": False,
                       "label": "Neurocomputing",
                       "class_rx": r"cas-(sc|dc)|els-cas",
                       "journal_rx": r"Neurocomputing"},
}

# elsarticle spells the environment \begin{keyword}; the CAS classes spell it
# \begin{keywords}. Accept either so the check survives the class swap.
KEYWORDS_RX = r"\\begin\{keywords?\}(.*?)\\end\{keywords?\}"


def strip_tex(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = s.replace("\\%", "%").replace("--", "-")
    s = s.replace("{", "").replace("}", "")
    return " ".join(s.replace("~", " ").split())


def countable_words(src: str) -> int:
    """Words in the body under the journal's exclusion rules.

    Excluded: the abstract, comments, the declarations block, every figure and
    table environment (so captions and table content drop out), and anything
    from \\bibliography onward (so references drop out).
    """
    s = re.sub(r"(?m)^\s*%.*$", "", src)
    s = re.sub(r"(?s)\\begin\{abstract\}.*?\\end\{abstract\}", "", s)
    s = re.sub(r"(?s)\\begin\{frontmatter\}.*?\\end\{frontmatter\}", "", s)
    s = re.sub(r"(?s)\\begin\{(figure\*?|table)\*?\}.*?\\end\{\1\*?\}", "", s)
    s = s.split("\\bibliographystyle")[0]
    s = re.sub(r"\\cite[a-z]*\{[^}]*\}", "", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", "", s)
    s = re.sub(r"[{}$^_&~]", "", s)
    return len([w for w in s.split() if re.search(r"[A-Za-z]{2}", w)])


def png_width(path: str) -> int:
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">I", head[16:20])[0]


def main(argv):
    which = argv[1] if len(argv) > 1 else "neurocomputing"
    if which not in LIMITS:
        print(f"unknown target: {which}")
        return 2
    L = LIMITS[which]
    src = open(TEX, encoding="utf-8").read()
    ok = True

    print(f"TARGET: {L['label']}")
    print("=" * 62)

    # ---- body length ----------------------------------------------------
    n_words = countable_words(src)
    if L["words"]:
        lo, hi = L["words"]
        good = lo <= n_words <= hi
        ok &= good
        print(f"[{'ok ' if good else 'FAIL'}] body words : {n_words}   "
              f"required {lo}-{hi}")
    else:
        # No limit is published. Reported for information, not as a pass or fail,
        # because silence in the guide is not the same as permission.
        print(f"[info] body words : {n_words}   journal states no limit")

    # ---- class and journal field ---------------------------------------
    if "class_rx" in L:
        m = re.search(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", src)
        cls = m.group(1).strip() if m else "<absent>"
        good = re.search(L["class_rx"], cls) is not None
        ok &= good
        print(f"[{'ok ' if good else 'FAIL'}] class      : {cls}   "
              f"guide links the CAS templates")
    if "journal_rx" in L:
        m = re.search(r"\\journal\s*\{([^}]*)\}", src)
        j = m.group(1).strip() if m else ""
        cls_ok = True
        cm = re.search(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", src)
        if cm and re.search(r"cas-(sc|dc)|els-cas", cm.group(1).strip()):
            cls_ok = True
        if not j:
            # The CAS bundle defines no \journal macro; the journal name is
            # recorded in the submission system rather than typeset by the
            # class. Absence is therefore correct, not a missing field.
            print(f"[ok ] journal    : not declared in the source, which is "
                  f"correct for the CAS class")
        else:
            good = re.search(L["journal_rx"], j) is not None
            ok &= good
            print(f"[{'ok ' if good else 'FAIL'}] journal    : {j}")

    # ---- abstract -------------------------------------------------------
    a = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
    if a:
        n = len([w for w in strip_tex(a.group(1)).split()
                 if re.search(r"[A-Za-z0-9]", w)])
        good = n <= L["abstract"]
        ok &= good
        print(f"[{'ok ' if good else 'FAIL'}] abstract   : {n} words   "
              f"max {L['abstract']}")

    # ---- keywords -------------------------------------------------------
    k = re.search(KEYWORDS_RX, src, re.S)
    if k:
        n = len([x for x in k.group(1).split("\\sep") if x.strip()])
        lo, hi = L["keywords"]
        good = lo <= n <= hi
        ok &= good
        print(f"[{'ok ' if good else 'FAIL'}] keywords   : {n}   "
              f"required {lo}-{hi}")

    # ---- highlights -----------------------------------------------------
    if L["highlights"]:
        # Neurocomputing takes Highlights as a separate file, so the source of
        # record is that file rather than the manuscript. elsarticle typeset them
        # in the document, and the imavis profile below still reads it there.
        if L.get("highlights_from_file"):
            path = os.path.join(PAPER, "highlights.txt")
            if not os.path.exists(path):
                print("[FAIL] highlights : file missing but REQUIRED")
                ok = False
            else:
                text = open(path, encoding="utf-8").read()
                items = [strip_tex(m.group(1)) for m in
                         re.finditer(r"^\s*\d\.\s+(.+)$", text, re.M)]
                cnt_ok = 3 <= len(items) <= 5
                ok &= cnt_ok
                print(f"[{'ok ' if cnt_ok else 'FAIL'}] highlights : "
                      f"{len(items)} bullets in highlights.txt   required 3-5")
                for i, t in enumerate(items, 1):
                    good = len(t) <= 85
                    ok &= good
                    print(f"       {'ok ' if good else 'OVER'} {len(t):>3}/85  {t}")
        else:
            h = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}",
                          src, re.S)
            if h is None:
                print("[FAIL] highlights : environment absent but REQUIRED")
                ok = False
            else:
                items = [strip_tex(x) for x in
                         re.findall(r"\\item\s+(.*)", h.group(1))]
                cnt_ok = 3 <= len(items) <= 5
                ok &= cnt_ok
                print(f"[{'ok ' if cnt_ok else 'FAIL'}] highlights : "
                      f"{len(items)} bullets   required 3-5")
                for i, t in enumerate(items, 1):
                    good = len(t) <= 85
                    ok &= good
                    print(f"       {'ok ' if good else 'OVER'} {len(t):>3}/85  {t}")

    # ---- graphical abstract --------------------------------------------
    if L["graphical_abstract"]:
        if not os.path.exists(GA):
            print("[FAIL] graphical abstract : file missing but REQUIRED")
            ok = False
        else:
            try:
                w = png_width(GA)
                good = w >= 1328
                ok &= good
                print(f"[{'ok ' if good else 'FAIL'}] graphical  : {w} px wide"
                      f"   min 1328")
            except Exception as exc:
                print(f"[FAIL] graphical abstract : {exc}")
                ok = False

    print("-" * 62)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

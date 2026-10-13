"""Scan a tree for credentials and other things that must not be published.

Run before every release. A public repository cannot be un-published, so this
is the last gate before anything leaves the machine.

What is searched for:
  passwords, API keys, tokens, private keys, connection strings
  the analysis host's address and its account name
  absolute paths that reveal the operator's directory layout
  credentials written into .bat launchers, which is where they usually land

Exit status is non-zero when anything is found, so it can gate a commit.
"""
from __future__ import annotations

import os
import re
import sys

# (label, compiled pattern, files to skip)
TEXT_EXT = {".py", ".bat", ".md", ".txt", ".json", ".yml", ".yaml",
            ".cfg", ".ini", ".toml", ".sh", ".ps1"}

RULES = [
    # `PASS:` in these scripts is a verdict label, not a credential, so the
    # value must look like something assignable to a variable: a quoted string,
    # an identifier, or a call. A bare `{` means the author is opening a dict.
    ("password assignment",
     re.compile(r"(?i)\b(pass(word|wd)?|pwd)\b\s*[:=]\s*"
                r"(?!\{)\s*['\"]?[A-Za-z_][\w.()\[\]'\"]*")),
    ("secret/token key",
     re.compile(r"(?i)\b(secret|token|api[_-]?key|access[_-]?key)\b\s*[:=]\s*"
                r"['\"]?[A-Za-z0-9/+_-]{8,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("hardcoded ip with port", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}\b")),
    ("bare private ipv4", re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("operator directory",
     re.compile(r"[A-Za-z]:\\+(program|Users)\\[A-Za-z0-9_.-]+", re.I)),
]

SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".venv", "venv"}


def iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in TEXT_EXT:
                yield os.path.join(dirpath, fn)


def main(argv) -> int:
    root = argv[1] if len(argv) > 1 else "."
    hits = []
    n = 0
    for path in iter_files(root):
        n += 1
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for label, rx in RULES:
                m = rx.search(line)
                if not m:
                    continue
                frag = m.group(0)
                # An env-var read is not a leaked credential.
                if re.search(r"(getenv|environ|ENV\[)", line) and \
                        re.search(r"(pass|pwd|secret|token)", line, re.I):
                    continue
                hits.append((os.path.relpath(path, root), i, label, frag))

    print(f"scanned {n} text files under {root}")
    if not hits:
        print("RESULT: clean -- nothing that must not be published")
        return 0
    print(f"\n{len(hits)} finding(s):\n")
    for p, i, label, frag in hits:
        shown = frag if len(frag) <= 70 else frag[:70] + "..."
        print(f"  {p}:{i}  [{label}]  {shown}")
    print("\nRESULT: BLOCKED -- review every line above before publishing")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
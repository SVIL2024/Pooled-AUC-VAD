"""probe_env: what is actually available for the training-free / frozen-foundation-model line?

Decision gate, not science.  The 2025-2026 VAD frontier (LAVAD, VADTree, PRISM,
Cog-VADU) is built on frozen foundation models, and the LLM variants need an
inference endpoint.  Before committing to that direction we need to know which
of these are actually reachable on this box, because the answer determines
whether the pivot is viable at all or whether only the CLIP-only variants are.

Checked
  A. torch / transformers / open_clip / sentence-transformers availability
  B. any locally cached model weights, and whether HF cache is populated
  C. outbound network access (pip / huggingface), which decides whether a
     pretrained checkpoint can be fetched at all
  D. dataset directories present, since a third domain is needed for any
     genuinely cross-domain experiment
  E. disk headroom

Prints a capability matrix and a one-line recommendation.  No experiment here.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import importlib
import json
import os
import socket
import sys


def has(mod):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        return True, v
    except Exception as exc:
        return False, type(exc).__name__


def main():
    print("=" * 78)
    print("A. PYTHON PACKAGES")
    print("=" * 78)
    for mod in ("torch", "torchvision", "transformers", "open_clip",
                "sentence_transformers", "accelerate", "timm", "ftfy",
                "regex", "sklearn", "scipy", "PIL"):
        ok, v = has(mod)
        print(f"  {mod:<24} {'YES ' + str(v) if ok else 'no  (' + v + ')'}")

    print()
    print("=" * 78)
    print("B. CACHED MODEL WEIGHTS")
    print("=" * 78)
    roots = [os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface"),
             os.path.expanduser("~/.cache/torch"), r"E:\models",
             os.environ.get("VADCLIP_ROOT", "VadCLIP") + r"\data"]
    for r in roots:
        if not r:
            continue
        if os.path.isdir(r):
            entries = []
            for dirpath, _dirnames, filenames in os.walk(r):
                for fn in filenames:
                    if fn.endswith((".pth", ".pt", ".bin", ".safetensors")):
                        full = os.path.join(dirpath, fn)
                        try:
                            sz = os.path.getsize(full) / 1e6
                        except OSError:
                            sz = -1
                        entries.append((sz, full))
            entries.sort(reverse=True)
            print(f"  {r}: {len(entries)} weight file(s)")
            for sz, full in entries[:12]:
                print(f"      {sz:9.1f} MB  {full}")
        else:
            print(f"  {r}: (absent)")

    print()
    print("=" * 78)
    print("C. NETWORK")
    print("=" * 78)
    for host, port in (("huggingface.co", 443), ("pypi.org", 443),
                       ("github.com", 443)):
        s = socket.socket()
        s.settimeout(6)
        try:
            s.connect((host, port))
            print(f"  {host}:{port}  REACHABLE")
        except Exception as exc:
            print(f"  {host}:{port}  no  ({type(exc).__name__})")
        finally:
            s.close()

    print()
    print("=" * 78)
    print("D. DATASETS")
    print("=" * 78)
    for root in (r"E:\dataset", os.environ.get("VADCLIP_ROOT", "VadCLIP") + r"\list"):
        if os.path.isdir(root):
            subs = sorted(os.listdir(root))
            print(f"  {root}: {len(subs)} entries")
            for s in subs[:40]:
                p = os.path.join(root, s)
                mark = "/" if os.path.isdir(p) else " "
                n = ""
                if os.path.isdir(p):
                    try:
                        n = f"  ({len(os.listdir(p))} files)"
                    except OSError:
                        pass
                print(f"     {mark} {s}{n}")
        else:
            print(f"  {root}: (absent)")

    print()
    print("=" * 78)
    print("E. DISK / GPU")
    print("=" * 78)
    try:
        import torch
        print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  gpu  {torch.cuda.get_device_name(0)}  "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    except Exception as exc:
        print(f"  torch unavailable: {exc}")
    for drive in ("C:\\", "E:\\"):
        try:
            import shutil
            free = shutil.disk_usage(drive)
            print(f"  {drive} free {free.free/1e9:.1f} GB / "
                  f"{free.total/1e9:.1f} GB")
        except Exception as exc:
            print(f"  {drive}: {exc}")

    print()
    print(f"python {sys.version.split()[0]}  executable {sys.executable}")


if __name__ == "__main__":
    main()
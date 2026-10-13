"""probe_net: outbound reachability and the dataset inventory.

Split out from probe_env because the socket probes were the part that hung, and
they are the part that decides whether the pivot is feasible.  Without a network
no new pretrained checkpoint can be fetched, and the 2026 training-free line
(which leans on frozen SigLIP / DINOv2 / LLM checkpoints) is closed off; with a
network it is merely an install problem.
"""
from __future__ import annotations

import os
import socket
import sys


def probe(host, port=443, timeout=5):
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return "REACHABLE"
    except Exception as exc:
        return f"no ({type(exc).__name__})"
    finally:
        s.close()


def main():
    print("=== NETWORK ===")
    for host, port in (("huggingface.co", 443), ("pypi.org", 443),
                       ("github.com", 443), ("8.8.8.8", 53)):
        print(f"  {host}:{port:<5} {probe(host, port)}")

    print()
    print("=== DNS ===")
    for host in ("huggingface.co", "pypi.org"):
        try:
            print(f"  {host} -> {socket.gethostbyname(host)}")
        except Exception as exc:
            print(f"  {host} -> {type(exc).__name__}")

    print()
    print("=== DATASET ROOT E:\\dataset ===")
    root = r"E:\dataset"
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                try:
                    subs = os.listdir(p)
                    inner = ""
                    if subs:
                        first = os.path.join(p, sorted(subs)[0])
                        if os.path.isdir(first):
                            inner = f" -> {sorted(subs)[0]}({len(os.listdir(first))})"
                    print(f"  DIR  {name:<34} {len(subs)} entries{inner}")
                except OSError as exc:
                    print(f"  DIR  {name:<34} <{exc}>")
            else:
                try:
                    print(f"  FILE {name:<34} {os.path.getsize(p)/1e6:.1f} MB")
                except OSError:
                    print(f"  FILE {name}")
    else:
        print("  (absent)")

    print()
    print("=== OTHER DISKS ===")
    for r in (r"E:\program", r"E:\dataset\vad", r"D:\dataset"):
        print(f"  {r}: {'present' if os.path.isdir(r) else 'absent'}")


if __name__ == "__main__":
    sys.exit(main())
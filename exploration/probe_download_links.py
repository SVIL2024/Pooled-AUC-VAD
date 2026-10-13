"""probe_download_links: follow the project pages to the actual archives.

The user is right that the project homepages are the place to look.  XD-Violence
has a reachable project page; UCF-Crime's crcv.ucf.edu page failed outright.
This script pulls the links out of whichever pages answer, classifies them by
host, and then actually probes each host -- because the decisive fact is not
"is there a link" but "is that host reachable from this box".

It also reports the separation that matters for the study itself: the proposed
scene-matched attribution needs RESOLUTION and CODEC per video.  That is
metadata obtainable from a small sidecar, a dataset card, or the competitor's
released scene-factor annotations, long before committing to any amount of
video download.  So any archive here is recorded with its size, and the
metadata-only route is scored separately.
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (probe)"}
TIMEOUT = 20

PAGES = [
    ("XD-Violence", "https://roc-ng.github.io/XD-Violence/"),
    ("UCF-Crime (official)", "https://www.crcv.ucf.edu/projects/real-world/"),
    ("UCF-Crime (http)", "http://www.crcv.ucf.edu/projects/real-world/"),
    ("VadCLIP repo", "https://github.com/nwpu-zxr/VadCLIP"),
    ("AnomalyCLIP repo", "https://github.com/zqhang/AnomalyCLIP"),
    ("awesome VAD list", "https://github.com/vt-le/Video-Anomaly-Detection"),
]

LINK_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)


def ctx():
    return ssl.create_default_context()


def fetch(url, timeout=TIMEOUT, limit=400_000):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=ctx()) as r:
        raw = r.read(limit)
        enc = r.headers.get_content_charset() or "utf-8"
        return r.status, raw.decode(enc, "replace"), dict(r.headers)


def host_reach(url, timeout=10):
    p = urllib.parse.urlparse(url)
    host = p.hostname or ""
    port = p.port or (443 if p.scheme == "https" else 80)
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx()) as r:
            n = r.headers.get("Content-Length", "?")
        return True, f"HTTP {r.status} len={n}"
    except Exception as exc:
        return False, type(exc).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"pages": {}, "links": {}}

    print("=" * 78)
    print("A. project pages")
    print("=" * 78)
    for label, url in PAGES:
        try:
            status, html, _ = fetch(url)
        except Exception as exc:
            print(f"  {label:<24} FAIL  {type(exc).__name__}")
            res["pages"][label] = {"url": url, "ok": False,
                                   "error": type(exc).__name__}
            continue
        links = sorted(set(LINK_RE.findall(html)))
        data_like = [l for l in links
                     if re.search(r"\.(zip|tar|gz|7z|rar|csv|json|txt|md)\b",
                                  l, re.I)]
        interesting = [l for l in data_like
                       if not re.search(r"\.js$|\.css$", l, re.I)]
        print(f"  {label:<24} HTTP {status}  {len(links)} links, "
              f"{len(interesting)} data-ish")
        for l in interesting[:14]:
            print(f"        {l[:100]}")
        res["pages"][label] = {"url": url, "ok": True, "status": status,
                               "n_links": len(links),
                               "data_links": interesting[:40]}

    print()
    print("=" * 78)
    print("B. probe the hosts those links point at")
    print("=" * 78)
    hosts = {}
    for label, p in res["pages"].items():
        for l in p.get("data_links", []):
            full = urllib.parse.urljoin(p["url"], l)
            h = urllib.parse.urlparse(full).netloc
            if h and h not in hosts:
                hosts[h] = full
    for h, sample in sorted(hosts.items()):
        ok, info = host_reach(sample)
        res["links"][h] = {"reachable": ok, "info": info, "sample": sample}
        print(f"  {h:<44} {'OK ' if ok else 'NO '} {info}")

    print()
    print("=" * 78)
    print("C. bandwidth (bounded GET of a known file)")
    print("=" * 78)
    url = ("https://openaipublic.azureedge.net/clip/models/"
           "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
           "ViT-B-32.pt")
    import time
    req = urllib.request.Request(url, headers=UA)
    t0, got = time.time(), 0
    try:
        with urllib.request.urlopen(req, timeout=60, context=ctx()) as r:
            while got < (16 << 20):
                c = r.read(1 << 18)
                if not c:
                    break
                got += len(c)
        dt = time.time() - t0
        mbps = got / dt / 1e6
        print(f"  {got/1e6:.1f} MB in {dt:.1f}s -> {mbps:.1f} MB/s")
        for gb in (10, 50, 100, 200):
            print(f"    {gb:>4} GB -> ~{gb*1000/max(mbps,1e-6)/60:6.1f} min")
        res["bandwidth"] = {"MBps": mbps}
    except Exception as exc:
        print(f"  FAILED after {got} bytes: {type(exc).__name__}")
        res["bandwidth"] = {"error": type(exc).__name__}

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
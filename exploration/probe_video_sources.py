"""probe_video_sources: can the raw UCF-Crime / XD-Violence footage be obtained,
and do we actually need it?

Two reasons this is worth checking before committing to the scene-matched
cross-domain study:

1. Feasibility.  huggingface.co is blocked on this box, but hf-mirror.com
   serves the same API, so any dataset mirrored on HF is reachable.  The usual
   homes for these two datasets (Google Drive, Dropbox, OneDrive) are likely
   filtered, so HF/mirror and any CDN are what actually matter.

2. Necessity.  The proposed attribution only needs RECORDING PROPERTIES --
   resolution, codec, pixel format, bitrate, duration.  That is metadata, not
   pixels.  Downloading hundreds of gigabytes of surveillance footage to read
   ffprobe output would be absurd if the competitor's released scene-factor
   annotations, or a dataset card, or a small metadata sidecar, already carries
   the same numbers.

So this probe reports, per candidate source: reachable, whether it is video or
metadata, total size, and file count.  It also measures raw download bandwidth
on a known file so that any size estimate can be turned into a wall-clock
number rather than left abstract.

Nothing is downloaded except one small bandwidth-probe file.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

MIRROR = "https://hf-mirror.com"
TIMEOUT = 20
UA = {"User-Agent": "probe/1.0"}

DATASET_CANDIDATES = [
    "UCF-Crime", "ucfcrime", "UCFcrime", "ucf-crime",
    "XD-Violence", "XDViolence", "xd-violence", "xdviolence",
]
MODEL_CANDIDATES = [
    "auditing-wsvad-scene-annotations",
]
OTHER_SOURCES = [
    ("openaipublic CLIP (bandwidth probe)",
     "https://openaipublic.azureedge.net/clip/models/"
     "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
     "ViT-B-32.pt"),
    ("UCF official page", "https://www.crcv.ucf.edu/projects/real-world/"),
    ("XD-Violence project page", "https://roc-ng.github.io/XD-Violence/"),
    ("Google Drive (expected blocked)", "https://drive.google.com/"),
    ("Dropbox (expected blocked)", "https://www.dropbox.com/"),
]


def ctx():
    return ssl.create_default_context()


def head(url, timeout=TIMEOUT):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx()) as r:
            return True, r.headers.get("Content-Length", "?"), r.status
    except urllib.error.HTTPError as exc:
        return True, str(exc.code), exc.code
    except Exception as exc:
        return False, type(exc).__name__, None


def get_json(url, timeout=TIMEOUT):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                   timeout=timeout, context=ctx()) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def bandwidth(url, max_bytes=8 << 20):
    req = urllib.request.Request(url, headers=UA)
    t0 = time.time()
    got = 0
    try:
        with urllib.request.urlopen(req, timeout=60, context=ctx()) as r:
            while got < max_bytes:
                chunk = r.read(1 << 18)
                if not chunk:
                    break
                got += len(chunk)
        dt = time.time() - t0
        return got, dt, (got / dt / 1e6 if dt > 0 else 0.0), None
    except Exception as exc:
        return got, time.time() - t0, None, f"{type(exc).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}

    print("=" * 78)
    print("A. HF (via hf-mirror) dataset search for raw footage")
    print("=" * 78)
    found = {}
    for name in DATASET_CANDIDATES:
        url = f"{MIRROR}/api/datasets/{urllib.parse.quote(name)}"
        meta = get_json(url)
        if "__error__" in meta:
            print(f"  {name:<22} --  {meta['__error__'][:60]}")
            continue
        sibs = [s.get("rfilename", "") for s in meta.get("siblings", [])]
        n_vid = sum(1 for f in sibs
                    if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")))
        n_meta = sum(1 for f in sibs
                     if f.lower().endswith((".csv", ".json", ".txt", ".yaml")))
        found[name] = {"id": meta.get("id"), "files": len(sibs),
                       "video_files": n_vid, "meta_files": n_meta}
        print(f"  {name:<22} OK  files={len(sibs):<6} video={n_vid:<6} "
              f"meta={n_meta}")
        for f in sibs[:8]:
            print(f"        {f}")
    res["hf_datasets"] = found

    print()
    print("=" * 78)
    print("B. Other candidate sources (reachability only)")
    print("=" * 78)
    other = {}
    for label, url in OTHER_SOURCES:
        ok, info, code = head(url)
        other[label] = {"reachable": ok, "info": info, "code": code}
        print(f"  {label:<40} {'OK' if ok else 'NO'}   {info}")
    res["other"] = other

    print()
    print("=" * 78)
    print("C. Raw download bandwidth (one bounded GET)")
    print("=" * 78)
    got, dt, mbps, err = bandwidth(OTHER_SOURCES[0][1])
    if mbps is None:
        print(f"  FAILED after {got} bytes  {err}")
    else:
        print(f"  {got/1e6:.1f} MB in {dt:.1f}s  ->  {mbps:.1f} MB/s")
        for gb in (10, 50, 100, 200):
            print(f"    a {gb:>4} GB pull would take ~{gb*1000/max(mbps,1e-6)/60:6.1f} min")
    res["bandwidth"] = {"bytes": got, "seconds": dt, "MBps": mbps,
                        "error": err}

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
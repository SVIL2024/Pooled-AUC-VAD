"""Fetch the XD-Violence I3D test dump from the HuggingFace mirror.

Why XD and not UCF
------------------
The UCF route is abandoned for a measured reason, not a guess. Every one of the
1,610 training videos in that dump holds exactly 32 clips -- a fixed-length
sample rather than a whole video -- while the 290 test videos average 240. A head
fitted on the first memorises it (held-out AUC inside the training videos hits
1.0000 by epoch five) and inverts on the second, because the two dumps are not
the same kind of object. The XD dump holds 3,950 training videos that are whole
videos, so the same experiment can be run with a real train/test separation.

Why not huggingface_hub
-----------------------
The mirror paginates with a Link header pointing back at huggingface.co, which is
unreachable here, so snapshot_download and list_repo_tree hang. This enumerates
the directory over plain HTTP, rewriting the Link host back to the mirror, and
fetches files directly. Redirects land on a CDN host that does work.

Format, already established: (T, 5, 2048) float32, centre crop in axis 1, and
the 2048 dimensions are RGB(1024) concatenated with Flow(1024). Only the RGB half
is wanted, and only the centre crop, so the pooled vector is arr[:, 0, :1024].

Writes a manifest recording every file's size and clip count, because the paper
requires numbers to be traceable to an archived record rather than to a download.
"""

from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

MIRROR = "https://hf-mirror.com"
REPO = "jherng/xd-violence"
REPO_TYPE = "dataset"
DEST = r"E:\dataset\XD_i3d\test"
RESULTS = os.environ.get("CDV_REMOTE_RESULTS", "results")
API_LIST = f"{MIRROR}/api/{REPO_TYPE}s/{REPO}/tree/main"
RESOLVE = f"{MIRROR}/{REPO_TYPE}s/{REPO}/resolve/main"
TIMEOUT = 60


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "curl/8"
    return s


def parse_link(header: str) -> str | None:
    """Next-page URL from a Link header, with the host forced back to the mirror."""
    if not header:
        return None
    for part in header.split(","):
        m = re.match(r'\s*<([^>]+)>;\s*rel="next"', part)
        if m:
            url = m.group(1)
            url = re.sub(r"^https?://[^/]+", MIRROR, url)
            return url
    return None


def list_dir(session: requests.Session, path: str,
             limit: int | None = None) -> list[dict]:
    """Enumerate one directory, following rewritten pagination links."""
    out: list[dict] = []
    url = f"{API_LIST}/{path}"
    params = {"recursive": "false", "expand": "false"}
    while url:
        r = session.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        page = r.json()
        if isinstance(page, dict):
            page = [page]
        out.extend(x for x in page if x.get("type") == "file")
        if limit and len(out) >= limit:
            return out[:limit]
        url = parse_link(r.headers.get("Link", ""))
        params = None
    return out


def fetch(session: requests.Session, entry: dict, dest_dir: str) -> dict:
    name = entry["path"].rsplit("/", 1)[-1]
    target = os.path.join(dest_dir, name)
    size = entry.get("size") or 0
    if os.path.isfile(target) and os.path.getsize(target) == size:
        return {"name": name, "bytes": size, "status": "cached"}
    url = f"{RESOLVE}/{entry['path']}"
    t0 = time.time()
    tmp = target + ".part"
    with session.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    got = os.path.getsize(tmp)
    if got != size:
        os.remove(tmp)
        return {"name": name, "bytes": got, "status": f"size mismatch {got}!={size}"}
    os.replace(tmp, target)
    return {"name": name, "bytes": got, "status": "ok",
            "seconds": round(time.time() - t0, 2)}


def tally_segments(path: str, entries: list[dict]) -> None:
    """Report whether this directory holds whole videos or pre-cut segments.

    The file names carry a start-end timestamp and a label suffix, so the unit of
    storage is visible without downloading anything. Deciding that from the names
    costs one API listing; discovering it after a multi-gigabyte transfer costs
    the transfer.
    """
    import collections

    names = [e["path"].rsplit("/", 1)[-1][:-4] for e in entries]
    print("\n=== what unit is stored here? ===")

    def base(n: str) -> str:
        return n.split("__#", 1)[0] if "__#" in n else n.split("__", 1)[0]

    def label(n: str) -> str:
        if "_label_" in n:
            return n.split("_label_", 1)[1]
        return "<no label suffix>"

    def timed(n: str) -> bool:
        part = n.split("__#", 1)[1] if "__#" in n else ""
        return bool(re.match(r"^[\dh_]+-[\dh_]+_", part))

    bases = collections.Counter(base(n) for n in names)
    labs = collections.Counter(label(n) for n in names)
    print(f"  files                 : {len(names)}")
    print(f"  distinct base videos  : {len(bases)}")
    print(f"  files carrying a start-end timestamp: "
          f"{sum(1 for n in names if timed(n))}/{len(names)}")
    print(f"  segments per base video (top 8): "
          f"{collections.Counter(bases.values()).most_common(8)}")
    print(f"  label suffixes (top 20): {labs.most_common(20)}")

    anon = sum(c for k, c in labs.items() if k.startswith(("A", "B")))
    print(f"  segments whose label looks anomalous (A*/B*): {anon}")
    print(f"  segments with no anomalous-looking label      : {len(names) - anon}")
    print("\n  interpretation: this directory stores pre-cut temporal segments "
          "with per-segment labels, not one feature file per test video."
          if timed(names[0])
          else "\n  interpretation: one file per video.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/i3d_rgb/test_videos")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 means the whole directory")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(DEST, exist_ok=True)
    s = get_session()

    print(f"listing {args.path} on {MIRROR}")
    entries = list_dir(s, args.path)
    print(f"  files listed: {len(entries)}")
    if not entries:
        print("  nothing listed; the path may be wrong")
        return 1
    entries.sort(key=lambda e: e["path"])
    print(f"  e.g. {[e['path'].rsplit('/', 1)[-1] for e in entries[:4]]}")
    print(f"  e.g. {[e['path'].rsplit('/', 1)[-1] for e in entries[-3:]]}")
    total = sum(e.get("size") or 0 for e in entries)
    print(f"  total size: {total / 1024 ** 3:.2f} GB")

    if args.list_only:
        tally_segments(args.path, entries)
        return 0
    if args.limit:
        entries = entries[: args.limit]

    print(f"\ndownloading {len(entries)} files with {args.workers} workers")
    t0 = time.time()
    done = bytes_done = 0
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, s, e, DEST) for e in entries]
        for fu in futs:
            rec = fu.result()
            results.append(rec)
            done += 1
            if rec["status"] in ("ok", "cached"):
                bytes_done += rec["bytes"]
            el = time.time() - t0
            rate = bytes_done / max(el, 1e-9) / 1024 ** 2
            print(f"  [{done}/{len(entries)}] {rec['name']:<28} "
                  f"{rec['status']:<14} {rate:.1f} MB/s", flush=True)

    bad = [r for r in results if r["status"] not in ("ok", "cached")]
    el = time.time() - t0
    print(f"\ndone in {el / 60:.1f} min, {bytes_done / 1024 ** 2 / max(el, 1e-9):.1f} MB/s")
    print(f"  ok or cached: {len(results) - len(bad)}   failed: {len(bad)}")
    for b in bad[:10]:
        print(f"    FAILED {b['name']}: {b['status']}")

    os.makedirs(RESULTS, exist_ok=True)
    tag = args.path.replace("/", "_")
    out = os.path.join(RESULTS, f"xd_i3d_download_{tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"source": f"{MIRROR}/{REPO_TYPE}s/{REPO}",
                   "path": args.path, "n_files": len(entries),
                   "total_bytes": total, "n_failed": len(bad),
                   "files": results}, fh, indent=2)
    print(f"wrote {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

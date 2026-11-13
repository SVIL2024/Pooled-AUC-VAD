"""Two cheap checks that decide whether a whole-video I3D dump is reachable.

First check
-----------
The jherng mirror's test directory turned out to hold 800 pre-cut anomalous
segments drawn from 447 videos, not 800 whole test videos, so it cannot support a
pooled-area measurement: there are no normal frames to pair against anomalous
ones. Its training directories are chunked by index range, which would be
consistent with whole videos. One API listing settles whether that is true, and
costs seconds.

Second check
------------
The secondary source is a single 13 GB ZIP whose central directory was already
parsed as 8,000 test members and 39,540 train members, in the per-crop
(T, 1024) form that matches the UCF I3D convention. Downloading 13 GB to learn
that is what produced the earlier BadZipFile. A ZIP is random-access by
construction, though: if the host honours Range requests, the central directory
gives each member's byte offset and only the members actually needed have to be
transferred -- about 2.95 GB of the 13 GB. Whether Range works is a one-request
test.

Nothing is downloaded beyond a directory listing and a few hundred bytes.
"""

from __future__ import annotations

import json
import re
import struct

import requests

MIRROR = "https://hf-mirror.com"
API_LIST = f"{MIRROR}/api/datasets/{{repo}}/tree/main"
RESOLVE = f"{MIRROR}/datasets/{{repo}}/resolve/main"
TIMEOUT = 60


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "curl/8"
    return s


def next_link(header: str) -> str | None:
    if not header:
        return None
    for part in header.split(","):
        m = re.match(r'\s*<([^>]+)>;\s*rel="next"', part)
        if m:
            return re.sub(r"^https?://[^/]+", MIRROR, m.group(1))
    return None


def list_dir(s: requests.Session, repo: str, path: str,
             limit: int = 0) -> list[dict]:
    out: list[dict] = []
    url = API_LIST.format(repo=repo) + f"/{path}"
    params: dict | None = {"recursive": "false", "expand": "false"}
    while url:
        r = s.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        page = r.json()
        if isinstance(page, dict):
            page = [page]
        out.extend(x for x in page if x.get("type") == "file")
        if limit and len(out) >= limit:
            return out[:limit]
        url = next_link(r.headers.get("Link", ""))
        params = None
    return out


def check_jherng_train(s: requests.Session) -> None:
    print("=== jherng/xd-violence training directories ===")
    for sub in ("data/i3d_rgb", "data/i3d_rgb/1-1004"):
        try:
            if sub.endswith("i3d_rgb"):
                entries = list_dir(s, "jherng/xd-violence", sub)
                dirs = [e["path"] for e in entries if e.get("type") == "directory"]
                print(f"  {sub}: {len(dirs)} subdirectories -> "
                      f"{[d.rsplit('/', 1)[-1] for d in dirs[:8]]}")
            else:
                entries = list_dir(s, "jherng/xd-violence", sub, limit=12)
                names = [e["path"].rsplit("/", 1)[-1] for e in entries]
                print(f"  {sub}: first names {names[:6]}")
                print(f"  {sub}: sizes {[e.get('size') for e in entries[:6]]}")
        except Exception as exc:
            print(f"  {sub}: {type(exc).__name__}: {exc}")
    # Does any jherng path look like whole videos?
    print("  note: a name carrying __#<start>_<end>_label_<X> is a pre-cut "
          "segment; a bare movie title is a whole video.")


def check_range(s: requests.Session) -> None:
    print("\n=== qiouzao/XD-Violence: is the 13 GB zip range-addressable? ===")
    path = "XD-Violence.zip"
    url = RESOLVE.format(repo="qiouzao/XD-Violence") + f"/{path}"
    try:
        r = s.head(url, timeout=TIMEOUT, allow_redirects=True)
        size = int(r.headers.get("Content-Length", "0"))
        print(f"  HEAD {r.status_code}  size={size / 1024 ** 3:.2f} GB  "
              f"accept-ranges={r.headers.get('Accept-Ranges')}")
    except Exception as exc:
        print(f"  HEAD failed: {type(exc).__name__}: {exc}")
        return
    if size <= 0:
        print("  no Content-Length; cannot plan range reads")
        return
    tail = min(1 << 20, size)
    try:
        rr = s.get(url, headers={"Range": f"bytes={size - tail}-{size - 1}"},
                   timeout=TIMEOUT)
        print(f"  RANGE status={rr.status_code}  "
              f"received={len(rr.content)} bytes  "
              f"content-range={rr.headers.get('Content-Range', '')[:60]}")
        if rr.status_code != 206:
            print("  -> partial content NOT supported; the whole archive "
                  "would have to be transferred")
            return
        blob = rr.content
        eocd = blob.rfind(b"PK\x05\x06")
        print(f"  EOCD at offset {eocd} within the tail")
        if eocd < 0:
            print("  no classic EOCD in the tail; ZIP64 locator expected")
            loc = blob.rfind(b"PK\x06\x07")
            print(f"  ZIP64 EOCD locator present: {loc >= 0}")
            if loc >= 0:
                (_sig, _disk, z64off, _disks) = struct.unpack_from(
                    "<4sIQI", blob, loc)
                print(f"  ZIP64 EOCD absolute offset: {z64off}")
                r2 = s.get(url, headers={
                    "Range": f"bytes={z64off}-{z64off + 55}"}, timeout=TIMEOUT)
                if r2.status_code == 206 and len(r2.content) >= 56:
                    (_s, _sz, _vm, _vn, _dn, _cd, _cdd, _nn, _tot, _cds,
                     _cdoff, _l) = struct.unpack_from("<4sQHHIIQQQQ", r2.content)
                    print(f"  central directory size={_cd}  "
                          f"entries={_nn}  offset={_cdoff}")
                    print(f"  -> an archive of {_nn} members is random-access "
                          f"addressable; targeted extraction is possible")
                else:
                    print(f"  ZIP64 EOCD fetch failed: "
                          f"{r2.status_code}, {len(r2.content)} bytes")
        else:
            n = struct.unpack_from("<H", blob, eocd + 10)[0]
            print(f"  classic EOCD: {n} entries")
    except Exception as exc:
        print(f"  range fetch failed: {type(exc).__name__}: {exc}")


def main() -> int:
    s = session()
    check_jherng_train(s)
    check_range(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

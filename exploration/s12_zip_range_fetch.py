"""Fetch selected members out of a remote ZIP using HTTP range requests.

Why not download the archive
----------------------------
The official XD-Violence feature dump is a single 12.21 GB ZIP. Downloading it
whole is precisely what produced the earlier 21.9 GB BadZipFile, and the whole
archive is not needed: the test split is 8,000 members out of 47,543, about
2.95 GB. A ZIP is random-access by construction, so the central directory gives
each member's offset and only the wanted members are transferred. The host
answers Range with 206 and advertises accept-ranges: bytes, which was verified
before this script was written.

ZIP64 is mandatory, not optional
--------------------------------
The archive exceeds 4 GB, so the classic end-of-central-directory record stores
0xFFFFFFFF sentinels for the directory offset and size. The 64-bit record and
its locator must be read before the directory can be located at all.

Integrity
---------
Every member carries a CRC-32 in the central directory and the transfer is
verified against it. A member that fails is reported rather than silently
skipped, because a silently short feature set would surface much later as an
implausible measurement.

Writes a manifest recording each member's byte range, so a later run can fetch
exactly the same bytes again.
"""

from __future__ import annotations

import argparse
import binascii
import json
import os
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import requests

RESOLVE = "https://hf-mirror.com/datasets/{repo}/resolve/main"
TIMEOUT = 90
EOCD_SIG = b"PK\x05\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CEN_SIG = b"PK\x01\x02"
LOC_SIG = b"PK\x03\x04"
LOCAL_FIXED = 30          # bytes of a local file header before name and extra
CEN_FIXED = 46            # bytes of a central directory entry before extras


class Member:
    __slots__ = ("name", "comp_size", "uncomp_size", "local_off", "method",
                 "crc32", "data_off")

    def __init__(self, name, comp_size, uncomp_size, local_off, method, crc32):
        self.name = name
        self.comp_size = comp_size
        self.uncomp_size = uncomp_size
        self.local_off = local_off
        self.method = method
        self.crc32 = crc32
        self.data_off = None

    def __repr__(self) -> str:
        return (f"<{self.name} comp={self.comp_size} uncomp={self.uncomp_size} "
                f"method={self.method} off={self.local_off}>")


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "curl/8"
    return s


def zip64_extra(extra: bytes, want: tuple[str, ...]) -> dict[str, int]:
    """Pull the requested fields out of the ZIP64 extended information field."""
    out: dict[str, int] = {}
    i = 0
    order = ("uncomp_size", "comp_size", "local_off", "disk_start")
    while i + 4 <= len(extra):
        hid, hsz = struct.unpack_from("<HH", extra, i)
        body = extra[i + 4: i + 4 + hsz]
        i += 4 + hsz
        if hid != 0x0001:
            continue
        p = 0
        for field in order:
            if field not in want or field in out:
                continue
            width = 4 if field == "disk_start" else 8
            if p + width > len(body):
                break
            out[field] = struct.unpack_from("<I" if width == 4 else "<Q",
                                            body, p)[0]
            p += width
    return out


def read_directory(s: requests.Session, url: str, size: int,
                   verbose: bool = True) -> list[Member]:
    """Locate and parse the central directory, following ZIP64 when required."""
    tail_len = min(1 << 17, size)
    r = s.get(url, headers={"Range": f"bytes={size - tail_len}-{size - 1}"},
              timeout=TIMEOUT)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError(f"host ignored Range (status {r.status_code}); "
                           f"the whole archive would have to be transferred")
    tail = r.content

    eocd = tail.rfind(EOCD_SIG)
    if eocd < 0:
        raise RuntimeError("no end-of-central-directory record in the tail")
    n_entries_16 = struct.unpack_from("<H", tail, eocd + 10)[0]
    cd_size_16, cd_off_16 = struct.unpack_from("<II", tail, eocd + 12)
    if verbose:
        print(f"  classic EOCD at tail offset {eocd}: "
              f"entries={n_entries_16} cd_size={cd_size_16} cd_off={cd_off_16}")

    cd_size, cd_off, n_entries = cd_size_16, cd_off_16, n_entries_16
    sentinel = (cd_off_16 == 0xFFFFFFFF or cd_size_16 == 0xFFFFFFFF
                or n_entries_16 == 0xFFFF)
    if sentinel:
        loc = tail.rfind(EOCD64_LOC_SIG)
        if loc < 0:
            raise RuntimeError("ZIP64 sentinel set but no locator in the tail")
        z64_off = struct.unpack_from("<Q", tail, loc + 8)[0]
        r2 = s.get(url, headers={"Range": f"bytes={z64_off}-{z64_off + 55}"},
                   timeout=TIMEOUT)
        r2.raise_for_status()
        if r2.content[:4] != EOCD64_SIG:
            raise RuntimeError(f"expected ZIP64 EOCD record, got "
                               f"{r2.content[:4]!r}")
        (_, z64_size, _vm, _vn, _dn, _cdn, _cdd,
         n_entries, z64_cd_size, z64_cd_off) = struct.unpack_from(
            "<4sQ2H2I4Q", r2.content, 0)
        cd_size, cd_off = z64_cd_size, z64_cd_off
        if verbose:
            print(f"  ZIP64 EOCD at {z64_off}: entries={n_entries} "
                  f"cd_size={cd_size} cd_off={cd_off}")

    r3 = s.get(url, headers={"Range": f"bytes={cd_off}-{cd_off + cd_size - 1}"},
               timeout=TIMEOUT)
    r3.raise_for_status()
    cd = r3.content
    if verbose:
        print(f"  central directory fetched: {len(cd)} bytes "
              f"(expected {cd_size})")

    members: list[Member] = []
    p = 0
    while p + CEN_FIXED <= len(cd) and cd[p:p + 4] == CEN_SIG:
        (_sig, _vm, _vn, _flags, method, _mt, _md, crc32, comp, uncomp,
         nlen, elen, clen, _ds, _ia, _ea,
         loff) = struct.unpack_from("<4s6H3I5H2I", cd, p)
        name = cd[p + CEN_FIXED: p + CEN_FIXED + nlen].decode("utf-8", "replace")
        extra = cd[p + CEN_FIXED + nlen: p + CEN_FIXED + nlen + elen]
        p += CEN_FIXED + nlen + elen + clen
        need = []
        if uncomp == 0xFFFFFFFF:
            need.append("uncomp_size")
        if comp == 0xFFFFFFFF:
            need.append("comp_size")
        if loff == 0xFFFFFFFF:
            need.append("local_off")
        big = zip64_extra(extra, tuple(need)) if need else {}
        members.append(Member(
            name,
            big.get("comp_size", comp),
            big.get("uncomp_size", uncomp),
            big.get("local_off", loff),
            method, crc32))
    if verbose:
        print(f"  parsed {len(members)} members "
              f"(directory claimed {n_entries})")
    return members


def data_offset(s: requests.Session, url: str, m: Member) -> int:
    """Local headers repeat the name but may carry a different extra field."""
    r = s.get(url, headers={"Range": f"bytes={m.local_off}-{m.local_off + LOCAL_FIXED - 1}"},
              timeout=TIMEOUT)
    r.raise_for_status()
    hdr = r.content
    if hdr[:4] != LOC_SIG:
        raise RuntimeError(f"{m.name}: local header signature {hdr[:4]!r}")
    nlen, elen = struct.unpack_from("<HH", hdr, 26)
    return m.local_off + LOCAL_FIXED + nlen + elen


def archive_size(s: requests.Session, url: str, retries: int = 8) -> int:
    """HEAD with backoff, because the mirror rate-limits even the metadata call.

    A rate-limited HEAD comes back without Content-Length, which previously read
    as a zero-byte archive and sent the run off to parse an empty directory.
    """
    last = ""
    for attempt in range(retries):
        r = s.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 5 * (attempt + 1)))
            print(f"  rate limited on HEAD; waiting {wait:.0f}s")
            time.sleep(min(wait, 90))
            last = "429"
            continue
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        if cl and int(cl) > 0:
            return int(cl)
        last = f"no Content-Length (status {r.status_code})"
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"HEAD gave no usable size: {last}")


def fetch_block(s: requests.Session, url: str, start: int, end: int,
                retries: int = 5) -> bytes:
    """One range request for a contiguous span, with backoff on rate limiting.

    Fetching each member separately costs two requests per member -- one for the
    local header, one for the payload -- which is 16,000 requests for the test
    split and reliably trips the mirror's rate limiter into 429s. Members are laid
    out consecutively in the archive, so a contiguous span is fetched once and
    sliced locally, cutting the request count by orders of magnitude.
    """
    last = ""
    for attempt in range(retries):
        try:
            r = s.get(url, headers={"Range": f"bytes={start}-{end}"},
                      timeout=TIMEOUT * 4)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 60))
                last = "429"
                continue
            r.raise_for_status()
            return r.content
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"block {start}-{end} failed: {last[:140]}")


def extract_from_block(buf: bytes, block_start: int,
                       group: list[Member]) -> list[tuple[Member, bytes]]:
    """Slice members out of an already-fetched span and verify each one.

    The local header repeats the file name but may carry a different extra field
    than the central directory does, so the name and extra lengths are read from
    the buffer at each member's known offset rather than assumed. Every member is
    checked against the CRC-32 in the central directory.
    """
    out: list[tuple[Member, bytes]] = []
    for m in group:
        pos = m.local_off - block_start
        if pos < 0 or pos + LOCAL_FIXED > len(buf):
            out.append((m, b""))
            continue
        if buf[pos:pos + 4] != LOC_SIG:
            out.append((m, b""))
            continue
        nlen, elen = struct.unpack_from("<HH", buf, pos + 26)
        ds = pos + LOCAL_FIXED + nlen + elen
        blob = buf[ds:ds + m.comp_size]
        if len(blob) != m.comp_size:
            out.append((m, b""))
            continue
        try:
            raw = blob if m.method == 0 else zlib.decompress(
                blob, -zlib.MAX_WBITS)
        except zlib.error:
            out.append((m, b""))
            continue
        if len(raw) != m.uncomp_size:
            out.append((m, b""))
            continue
        if (binascii.crc32(raw) & 0xFFFFFFFF) != m.crc32:
            out.append((m, b""))
            continue
        out.append((m, raw))
    return out


def fetch_blocks(s: requests.Session, url: str, members: list[Member],
                 dest_dir: str, block_bytes: int,
                 workers: int) -> dict:
    """Group members into contiguous spans, fetch each span once, write results."""
    pending = [m for m in members
               if not (os.path.isfile(os.path.join(dest_dir, m.name.rsplit("/", 1)[-1]))
                       and os.path.getsize(os.path.join(
                           dest_dir, m.name.rsplit("/", 1)[-1])) == m.uncomp_size)]
    print(f"  already present: {len(members) - len(pending)}   "
          f"to fetch: {len(pending)}")

    pending.sort(key=lambda m: m.local_off)
    groups: list[list[Member]] = []
    cur: list[Member] = []
    for m in pending:
        if cur and m.local_off - cur[-1].local_off > block_bytes:
            groups.append(cur)
            cur = []
        cur.append(m)
    if cur:
        groups.append(cur)
    print(f"  contiguous spans: {len(groups)} "
          f"(mean {len(pending) / max(len(groups), 1):.0f} members each)")

    stats = {"ok": 0, "cached": len(members) - len(pending),
             "failed": 0, "failures": []}
    t0 = time.time()
    done_bytes = 0

    def handle(gidx: int, grp: list[Member]) -> None:
        nonlocal done_bytes
        start = grp[0].local_off
        end = start
        for m in grp:
            end = max(end, m.local_off + LOCAL_FIXED + len(m.name.encode()) + 512
                      + m.comp_size)
        try:
            buf = fetch_block(s, url, start, end - 1)
        except Exception as exc:
            for m in grp:
                stats["failed"] += 1
                stats["failures"].append({"name": m.name, "why": str(exc)[:120]})
            return
        for m, raw in extract_from_block(buf, start, grp):
            leaf = m.name.rsplit("/", 1)[-1]
            if not raw:
                stats["failed"] += 1
                stats["failures"].append({"name": leaf,
                                          "why": "slice/crc/extract failed"})
                continue
            tmp = os.path.join(dest_dir, leaf + ".part")
            with open(tmp, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, os.path.join(dest_dir, leaf))
            stats["ok"] += 1
            done_bytes += len(raw)
        el = time.time() - t0
        print(f"  span {gidx + 1}/{len(groups)}  "
              f"{done_bytes / 1024 ** 2 / max(el, 1e-9):.1f} MB/s  "
              f"ok={stats['ok']} failed={stats['failed']}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda i: handle(i, groups[i]), range(len(groups))))

    el = time.time() - t0
    print(f"\n  wrote {stats['ok']} members, {stats['failed']} failed "
          f"in {el / 60:.1f} min ({done_bytes / 1024 ** 2 / max(el, 1e-9):.1f} MB/s)")
    return stats


def fetch_member(s: requests.Session, url: str, m: Member, dest_dir: str,
                 retries: int = 3) -> dict:
    name = m.name.rsplit("/", 1)[-1]
    target = os.path.join(dest_dir, name)
    if os.path.isfile(target) and os.path.getsize(target) == m.uncomp_size:
        return {"name": name, "status": "cached", "bytes": m.uncomp_size}
    last = ""
    for attempt in range(retries):
        try:
            off = data_offset(s, url, m)
            r = s.get(url, headers={
                "Range": f"bytes={off}-{off + m.comp_size - 1}"},
                timeout=TIMEOUT)
            r.raise_for_status()
            blob = r.content
            if m.method == 0:
                raw = blob
            elif m.method == 8:
                raw = zlib.decompress(blob, -zlib.MAX_WBITS)
            else:
                return {"name": name, "status": f"unsupported method {m.method}"}
            if len(raw) != m.uncomp_size:
                last = f"size {len(raw)} != {m.uncomp_size}"
                continue
            got = binascii.crc32(raw) & 0xFFFFFFFF
            if got != m.crc32:
                last = f"crc {got:08x} != {m.crc32:08x}"
                continue
            tmp = target + ".part"
            with open(tmp, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, target)
            return {"name": name, "status": "ok", "bytes": m.uncomp_size}
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 + attempt)
    return {"name": name, "status": f"failed: {last[:120]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="qiouzao/XD-Violence")
    ap.add_argument("--zip", default="XD-Violence.zip")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--prefix", default="", help="member prefix to select")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block-mb", type=int, default=24,
                    help="contiguous span per range request")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--index-only", action="store_true",
                    help="save the parsed directory and stop")
    args = ap.parse_args()

    url = RESOLVE.format(repo=args.repo) + f"/{args.zip}"
    s = session()
    size = archive_size(s, url)
    print(f"archive {args.zip}  {size / 1024 ** 3:.2f} GB  "
          f"accept-ranges={s.head(url, timeout=TIMEOUT).headers.get('Accept-Ranges')}")

    print("\n=== central directory ===")
    members = read_directory(s, url, size)
    methods = {}
    for m in members:
        methods[m.method] = methods.get(m.method, 0) + 1
    print(f"  compression methods: {methods} "
          f"(0=stored, 8=deflate)")

    if args.prefix:
        sel = [m for m in members if m.name.startswith(args.prefix)]
    else:
        sel = list(members)
    print(f"\n=== selection: prefix={args.prefix!r} ===")
    print(f"  members selected: {len(sel)} of {len(members)}")
    sel_bytes = sum(m.uncomp_size for m in sel)
    print(f"  uncompressed total: {sel_bytes / 1024 ** 3:.2f} GB")
    print(f"  e.g. {[m.name for m in sel[:3]]}")
    tops = sorted({m.name.split("/")[0] for m in members})
    print(f"  archive top-level dirs: {tops[:10]}")

    os.makedirs(args.dest, exist_ok=True)
    idx_path = os.path.join(args.dest, "_zip_index.json")
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump({"archive": url, "size": size,
                   "members": [{"name": m.name, "comp": m.comp_size,
                                "uncomp": m.uncomp_size,
                                "off": m.local_off, "method": m.method,
                                "crc32": m.crc32} for m in sel]}, fh)
    print(f"  wrote {idx_path}")

    if args.list_only or args.index_only:
        return 0
    if args.limit:
        sel = sel[: args.limit]
        print(f"  limited to {len(sel)} members")

    print(f"\n=== fetching in contiguous spans "
          f"({args.block_mb} MB each, {args.workers} workers) ===")
    real = [m for m in sel if not m.name.endswith("/")]
    stats = fetch_blocks(s, url, real, args.dest, args.block_mb << 20,
                         args.workers)

    out = os.path.join(os.path.dirname(args.dest.rstrip("\\/")) or ".",
                       "xd_i3d_fetch_manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"archive": url, "prefix": args.prefix,
                   "n_selected": len(sel), "n_files": len(real),
                   "block_mb": args.block_mb, **stats}, fh, indent=2)
    print(f"wrote {out}")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

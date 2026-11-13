"""Video-level dataset for cross-domain WSVAD (UCF-Crime / XD-Violence).

Path convention
---------------
Source CSVs were generated on a machine whose dataset root was ``E:/dataset``.
The features actually live under ``D:/dataset``.  A naive string replace
therefore silently skips every XD *normal* row, because those rows carry a
different prefix entirely (``/home/xbgydx/Desktop/...``).

We therefore resolve every row by its **last two path components**
(``<parent dir>/<file>.npy``) against the local feature root.  Any row that
cannot be resolved is reported loudly instead of being dropped.

GT convention
-------------
``gt_<tag>.npy`` is a flat float32 array over the concatenated test videos.
Video ``i`` contributes ``N_i * 16`` entries, where ``N_i`` is its feature
length.  Per-video frame-level labels are therefore::

    gt[offset : offset + N*16 : 16]        (one label per feature)
    offset += N * 16

which is what :func:`load_domain` returns as ``y_seg``.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

CLIP_LEN = 16

FEATURE_ROOTS = {
    "ucf": r"D:\dataset\UCFClipFeatures",
    "xd_train": r"D:\dataset\XDTrainClipFeatures",
    "xd_test": r"D:\dataset\XDTestClipFeatures",
}
LIST_DIR = r"D:\program\VadCLIP-main\list"

# Which feature root each split CSV points at, and which GT file it uses.
SPLITS = {
    "ucf_test": dict(
        csv="ucf_CLIP_rgbtest.csv", root="ucf", gt="gt_ucf.npy",
        normal_tokens=("Normal",), tag="UCF"),
    "ucf_train": dict(
        csv="ucf_CLIP_rgb.csv", root="ucf", gt=None,
        normal_tokens=("Normal",), tag="UCF"),
    "xd_test": dict(
        csv="xd_CLIP_rgbtest.csv", root="xd_test", gt="gt.npy",
        normal_tokens=("A",), tag="XD"),
    "xd_train": dict(
        csv="xd_CLIP_rgb.csv", root="xd_train", gt=None,
        normal_tokens=("A",), tag="XD"),
}


@dataclass
class Domain:
    """One split of one dataset, loaded as a list of per-video records."""

    tag: str
    name: str
    paths: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)   # raw string labels
    lengths: List[int] = field(default_factory=list)    # N per video
    y_seg: List[np.ndarray] = field(default_factory=list)  # (N,) float32 {0,1}
    video_y: np.ndarray = None                          # (V,) float32 {0,1}
    root: str = ""

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def n_normal(self) -> int:
        return int((self.video_y < 0.5).sum())

    @property
    def n_anomaly(self) -> int:
        return int((self.video_y >= 0.5).sum())

    @property
    def anomaly_rate(self) -> float:
        """Frame-level positive rate (the duty cycle the MIL top-k must cover)."""
        tot = sum(int(y.sum()) for y in self.y_seg)
        n = sum(len(y) for y in self.y_seg)
        return tot / max(n, 1)

    def load_video(self, i: int, max_seg: Optional[int] = None) -> np.ndarray:
        """Return (N, 512) float32 features for video ``i``.

        ``max_seg`` truncates to the first ``max_seg`` segments (cheap probes).
        """
        x = np.load(self.paths[i]).astype(np.float32)
        if max_seg is not None and x.shape[0] > max_seg:
            x = x[:max_seg]
        return x


def _resolve(path: str, root: str) -> Optional[str]:
    """Map a CSV path onto the local feature root.

    The CSVs were written on a machine whose root was ``E:/dataset`` (and, for
    XD normal rows, under a completely different prefix), while the features
    live under ``D:/dataset``.  Layouts differ too: UCF nests one directory per
    class, XD is flat.  So we try the plausible reconstructions in order and
    return None if none exists -- the caller reports misses rather than
    silently dropping rows.
    """
    p = path.replace("\\", "/")
    parts = [x for x in p.split("/") if x]
    if not parts:
        return None
    fname = parts[-1]
    cands = [os.path.join(root, fname)]
    if len(parts) >= 2:
        cands.insert(0, os.path.join(root, parts[-2], fname))
    for c in cands:
        if os.path.isfile(c):
            return c
    # Last resort: one level of subdirectories.
    try:
        subs = os.listdir(root)
    except OSError:
        return None
    for sub in subs:
        c2 = os.path.join(root, sub, fname)
        if os.path.isfile(c2):
            return c2
    return None


def _is_normal(label: str, tokens) -> bool:
    parts = [p for p in str(label).split("-") if p]
    return any(p in tokens for p in parts)


def _read_split_csv(path: str) -> List[Tuple[str, str]]:
    """Read a ``path,label`` split CSV with the stdlib (no pandas dependency)."""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return rows
        # tolerate column reordering
        pi = li = 0
        for i, h in enumerate(h for h in header):
            hl = h.strip().lower()
            if hl == "path":
                pi = i
            elif hl == "label":
                li = i
        for r in reader:
            if len(r) <= max(pi, li):
                continue
            rows.append((r[pi].strip(), r[li].strip()))
    return rows


def load_domain(name: str, list_dir: str = LIST_DIR, verbose: bool = True) -> Domain:
    """Load one split.  ``name`` is a key of :data:`SPLITS`."""
    spec = SPLITS[name]
    root = FEATURE_ROOTS[spec["root"]]
    rows = _read_split_csv(os.path.join(list_dir, spec["csv"]))
    norm_tok = spec["normal_tokens"]

    # Keep exactly one crop per video (the __0 crop for XD, __5 for UCF test).
    crop = "__5.npy" if name == "ucf_test" else "__0.npy"
    rows = [(p, l) for p, l in rows if p.replace("\\", "/").endswith(crop)]

    dom = Domain(tag=spec["tag"], name=name, root=root)

    # Resolve every path first, so we can report misses instead of dropping them.
    resolved = []
    n_miss = 0
    for raw, _ in rows:
        rp = _resolve(raw, root)
        if rp is None:
            n_miss += 1
        resolved.append(rp)
    if n_miss and verbose:
        print(f"[load_domain:{name}] WARNING {n_miss}/{len(resolved)} paths "
              f"unresolved under {root}")

    gt = None
    off = 0
    if spec["gt"] is not None:
        gt = np.load(os.path.join(list_dir, spec["gt"])).astype(np.float32)

    for (raw, label), rp in zip(rows, resolved):
        if rp is None:
            continue
        n = np.load(rp, mmap_mode="r").shape[0]
        dom.paths.append(rp)
        dom.labels.append(label)
        dom.lengths.append(int(n))
        if gt is not None:
            dom.y_seg.append((gt[off:off + n * CLIP_LEN:CLIP_LEN] > 0.5).astype(np.float32))
            off += n * CLIP_LEN
        else:
            dom.y_seg.append(np.zeros(n, dtype=np.float32))

    dom.video_y = np.array(
        [0.0 if _is_normal(l, norm_tok) else 1.0 for l in dom.labels],
        dtype=np.float32)

    if verbose:
        print(f"[load_domain:{name}] videos={len(dom)} "
              f"normal={dom.n_normal} anomaly={dom.n_anomaly} "
              f"frame_rate={dom.anomaly_rate:.4f}")
    return dom


def cross_domain_pair(source: str, target: str, **kw):
    """Convenience: load a (source, target) pair for transfer experiments."""
    return load_domain(source, **kw), load_domain(target, **kw)
"""Rescore ShanghaiTech against the dataset's own frame masks.

Why this exists
---------------
The manuscript's third-benchmark results were computed against a label vector
reconstructed to clip resolution. Comparing that reconstruction against the
dataset's original per-frame masks showed the two disagree on the overall
positive rate by 0.035, which is far more than the difference in resolution
between a frame mask and a 16-frame clip can explain.

The ShanghaiTech result is the paper's only evidence that the effect reverses
sign on a split with no normal videos, and it is the increment over
Song and Lee's published decomposition. It therefore cannot rest on labels whose
provenance is in question. This script rebuilds the clip-level labels directly
from the frame masks and recomputes every reported ShanghaiTech quantity.

What is verified before anything is recomputed
----------------------------------------------
1. Video order. The score archive stores a `names` array; the mask files are
   keyed by those names, so the mapping is explicit rather than assumed.
2. Clip segmentation. Each video's clip count must equal ceil(frames / 16),
   since one clip spans 16 frames. If this fails, the score archive's clip
   boundaries do not match the masks and no label rebuild is meaningful.
3. Padding. Reconstruction stores 16 frames per clip, so its length exceeds the
   mask frame count by the padding. That is expected and is excluded here.

Nothing on the server is written. Scores are read, labels are rebuilt, metrics
are recomputed locally.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ast
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
MASK_DIR = os.environ.get("SHT_FRAME_MASKS",
                        os.path.join(REPO_ROOT, "sht_frame_masks"))
SCORE_DIR = os.path.join(REPO_ROOT, "data")
FRAMES_PER_CLIP = 16
OUT = os.path.join(RES, "sht_rescored.json")


def _basename(entry) -> str:
    """Recover the mask filename from whatever the archive stored.

    The `names` array holds strings that are themselves the repr of a 1-tuple,
    e.g. "('01_0014.npy',)" -- the tuple was stringified when the archive was
    written. So the entry is parsed first and only then unwrapped; a plain
    regex fallback covers entries stored as bare names.
    """
    if isinstance(entry, (tuple, list, np.ndarray)):
        entry = entry[0]
    s = str(entry).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (tuple, list)):
            s = str(parsed[0])
        else:
            s = str(parsed)
    except (ValueError, SyntaxError):
        m = re.search(r"[\w.-]+\.npy", s)
        if m:
            s = m.group(0)
    return s


def load_masks(names):
    """Per-video frame label vectors, in the order given by `names`."""
    fracs, lens = [], []
    for raw in names:
        n = _basename(raw)
        p = os.path.join(MASK_DIR, n if n.endswith(".npy") else n + ".npy")
        v = np.atleast_1d(np.load(p, allow_pickle=False)).astype(np.int64)
        fracs.append(v)
        lens.append(int(v.size))
    return fracs, lens


def clip_labels(fracs, lens, rule):
    """One label per 16-frame clip, under the given aggregation rule.

    'any'  clip positive if it contains at least one anomalous frame
    'maj'  clip positive if most of its real frames are anomalous
    """
    out = []
    for v, L in zip(fracs, lens):
        n_clip = int(np.ceil(L / FRAMES_PER_CLIP))
        pos = 0
        for c in range(n_clip):
            lo, hi = c * FRAMES_PER_CLIP, min((c + 1) * FRAMES_PER_CLIP, L)
            seg = v[lo:hi]
            if rule == "any":
                pos += int(seg.max() > 0)
            else:
                pos += int(seg.mean() >= 0.5)
        out.append((n_clip, pos))
    return out


def auc(scores, labels):
    """Rank-based AUC, no ties correction needed for a first pass but ties are
    averaged so the value does not depend on the sample order."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks within tie groups
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ap(scores, labels):
    """Average precision, threshold-swept, ties grouped."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    total = int(labels.sum())
    if total == 0:
        return float("nan")
    return float((tp / k)[y].sum() / total)


def within_cross(scores, labels, clip_of, video_of, seed=0):
    """Within-video AUC and cross-video AUC by explicit pair sampling."""
    rng = np.random.RandomState(seed)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)

    pos = np.flatnonzero(labels)
    neg = np.flatnonzero(~labels)

    # within: pairs from the same video
    ws = []
    for v in np.unique(video_of):
        p = pos[video_of[pos] == v]
        q = neg[video_of[neg] == v]
        if p.size == 0 or q.size == 0:
            continue
        idx_p = rng.choice(p, size=min(400000, p.size * q.size), replace=True)
        idx_q = rng.choice(q, size=idx_p.size, replace=True)
        ws.append((scores[idx_p], scores[idx_q]))
    if ws:
        a = np.concatenate([x[0] for x in ws])
        b = np.concatenate([x[1] for x in ws])
        within = float((a[:, None] > b[None, :]).mean()
                       + 0.5 * (a[:, None] == b[None, :]).mean())
    else:
        within = float("nan")

    m = min(400000, pos.size * neg.size)
    if m:
        ip = rng.choice(pos, size=m, replace=True)
        iq = rng.choice(neg, size=m, replace=True)
        ok = video_of[ip] != video_of[iq]
        ip, iq = ip[ok], iq[ok]
        cross = float((scores[ip] > scores[iq]).mean()
                      + 0.5 * (scores[ip] == scores[iq]).mean())
    else:
        cross = float("nan")

    return within, cross, len(ws)


def main() -> int:
    report = {"frames_per_clip": FRAMES_PER_CLIP, "configs": {}}

    for tag in ("ucf2sht", "xd2sht"):
        f = os.path.join(SCORE_DIR, f"cdv_scores_{tag}.npz")
        if not os.path.exists(f):
            print(f"missing score archive {f}")
            continue
        d = np.load(f, allow_pickle=True)
        # Keep the raw objects: the archive stores 1-tuples, and stringifying
        # here would bake the tuple repr into the filename before load_masks
        # can unwrap it.
        names = list(d["names"])
        lens_clip = np.asarray(d["lens"], dtype=np.int64)
        p1 = np.asarray(d["prob1"], dtype=np.float64)
        p2 = np.asarray(d["prob2"], dtype=np.float64)

        print(f"\n=== {tag} ===")
        print(f"  videos={len(names)}  lens_entries={len(lens_clip)}  "
              f"sum(lens)={int(lens_clip.sum())}  "
              f"prob1={len(p1)}  prob2={len(p2)}")

        fracs, lens_frame = load_masks(names)

        # --- check 1: clip segmentation must match ceil(frames/16) --------
        implied = np.array([int(np.ceil(L / FRAMES_PER_CLIP))
                            for L in lens_frame])
        seg_ok = bool((implied == lens_clip).all())
        print(f"  clip segmentation matches ceil(frames/16): {seg_ok}")
        if not seg_ok:
            bad = np.flatnonzero(implied != lens_clip)[:5]
            for i in bad:
                print(f"    {names[i]}: frames={lens_frame[i]} "
                      f"archive_clips={lens_clip[i]} implied={implied[i]}")
        print(f"  mask frames total={sum(lens_frame)}  "
              f"reconstruction frames={int(lens_clip.sum()) * FRAMES_PER_CLIP}  "
              f"padding={int(lens_clip.sum()) * FRAMES_PER_CLIP - sum(lens_frame)}")

        if not seg_ok:
            print("  STOP: clip boundaries disagree with masks; "
                  "rebuilding labels would be meaningless")
            report["configs"][tag] = {"seg_ok": False}
            continue

        video_of = np.repeat(np.arange(len(names)), lens_clip)
        clip_of = np.concatenate(
            [np.arange(lens_clip[i]) + int(lens_clip[:i].sum())
             for i in range(len(names))])

        entry = {"seg_ok": True, "n_videos": len(names),
                 "n_clips": int(lens_clip.sum()),
                 "n_frames": int(sum(lens_frame))}

        for rule in ("any", "maj"):
            per_video = clip_labels(fracs, lens_frame, rule)
            labels = np.concatenate(
                [np.r_[np.ones(p, bool), np.zeros(n - p, bool)]
                 for n, p in per_video])
            pos_rate = float(labels.mean())
            n_full = int(sum(1 for n, p in per_video if p == n))
            n_none = int(sum(1 for n, p in per_video if p == 0))
            entry[rule] = {
                "pos_rate": round(pos_rate, 6),
                "n_clips": int(labels.size),
                "n_clips_positive": int(labels.sum()),
                "n_fully_anomalous_videos": n_full,
                "n_fully_normal_videos": n_none,
                "n_mixed_videos": int(len(per_video) - n_full - n_none),
            }
            print(f"  [{rule}] clips={labels.size}  pos_rate={pos_rate:.4f}  "
                  f"fully-anom videos={n_full}  fully-normal={n_none}")

            for pname, sc in (("prob1", p1), ("prob2", p2)):
                within, cross, _ = within_cross(sc, labels, clip_of, video_of)
                entry[rule][pname] = {
                    "pooled_auc": round(auc(sc, labels), 4),
                    "pooled_ap": round(ap(sc, labels), 4),
                    "within_auc": round(within, 4),
                    "cross_auc": round(cross, 4),
                    "gap": round(cross - within, 4),
                }
                print(f"    {pname}: pooled_auc={entry[rule][pname]['pooled_auc']:.4f} "
                      f"within={within:.4f}  cross={cross:.4f}  "
                      f"gap={cross - within:+.4f}")

        report["configs"][tag] = entry

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
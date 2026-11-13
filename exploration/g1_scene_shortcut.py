"""Stage 0 / G1 — Is cross-domain pooled AUC measuring anomaly or scene?

Question
--------
Cross-domain VAD reports pooled frame-level AUC on an *unseen* domain.  Two
2026 audits (arXiv:2608.21854, arXiv:2608.11985) show pooled AUC is dominated
by cross-video comparison, and that on normal footage alone models separate
videos by recording properties.  If both hold, then a cross-domain pooled AUC
can be inflated by a scene/domain shortcut that has nothing to do with anomaly
evidence -- and any method that wins by removing the shortcut can *lose*
pooled AUC while genuinely improving detection.

This script tests the cheapest decisive version of that claim, on features we
already have on disk, with no training and no GPU:

  A. Granularity decomposition of a fixed, non-trained scoring rule on UCF/XD.
  B. Capacity control  — how much pooled AUC survives video-constant scores.
  C. Normal-only domain separability — can we tell UCF from XD using *normal*
     videos and a video-mean score alone?  (pure shortcut capacity)

Gates
-----
  G1 PASS requires BOTH:
    (B1) normal-only domain AUC > 0.70   -> shortcut capacity is large
    (B2) within-video AUC clearly < video-level AUC -> localisation is the weak part
  Anything else -> report and stop; do not build a paper on it.

Run:  python probe/g1_scene_shortcut.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cdvad import data as D
from cdvad import metrics as M

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MAX_SEG = 96          # truncate long videos; all probes are segment-level
MIN_SEG = 8           # skip degenerate videos


def clip_norm_score(x: np.ndarray) -> np.ndarray:
    """Training-free per-frame anomaly score: distance from the video centroid.

    Deliberately weak and parameter-free.  We are measuring the *metric*, not
    proposing a detector.
    """
    c = x.mean(axis=0, keepdims=True)
    return np.linalg.norm(x - c, axis=1)


def collect(dom: D.Domain):
    ys, ss = [], []
    for i in range(len(dom)):
        n = dom.lengths[i]
        if n < MIN_SEG:
            continue
        x = dom.load_video(i, max_seg=min(n, MAX_SEG))
        y = dom.y_seg[i][:x.shape[0]]
        ys.append(y)
        ss.append(clip_norm_score(x))
    return ys, ss


def probe_domain_separation(src: D.Domain, tgt: D.Domain, tag_src: str, tag_tgt: str):
    """C. Use *normal videos only*; can a video-mean score tell the domains apart?

    Each domain is described by its (video-mean score, label, video id) triples.
    A domain classifier is fit on a random half and evaluated on the held-out
    half, so the number is not an in-sample artefact.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    feats, ys, vids = [], [], []
    for tag, dom in ((tag_src, src), (tag_tgt, tgt)):
        for i in range(len(dom)):
            if dom.video_y[i] >= 0.5:      # NORMAL videos only
                continue
            n = dom.lengths[i]
            if n < MIN_SEG:
                continue
            x = dom.load_video(i, max_seg=min(n, MAX_SEG))
            s = clip_norm_score(x)
            feats.append(s)
            ys.append(0 if tag == tag_src else 1)
            vids.append(len(vids))

    y = np.array(ys)
    F = np.array([float(f.mean()) for f in feats])[:, None]
    Ft = np.array([[float(f.std()),
                    float(np.mean(np.abs(np.diff(f, prepend=f[:1]))))]
                   for f in feats])

    out = {}
    for name, X in (("video_mean", F), ("std_plus_roughness", Ft),
                    ("both", np.hstack([F, Ft]))):
        if len(np.unique(y)) < 2:
            out[name] = float("nan")
            continue
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.5, random_state=0, stratify=y)
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        out[name] = float(clf.score(Xte, yte))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"config": dict(max_seg=MAX_SEG, min_seg=MIN_SEG,
                              score="||f_t - mean(f)|| (training-free)")}

    ucf = D.load_domain("ucf_test")
    xd = D.load_domain("xd_test")

    # ---- A + B: granularity decomposition and capacity control ----
    for dom in (ucf, xd):
        ys, ss = collect(dom)
        rep = M.full_report(ys, ss)
        results[dom.name] = {k: (round(float(v), 6) if np.isfinite(v) else None)
                             for k, v in rep.items()}
        w = rep["within_w"]
        micro = rep["micro_recheck"]
        print(f"\n=== {dom.name} ({dom.tag}) ===")
        print(f"  pooled AUC (micro)      : {micro:.4f}")
        print(f"  pooled AP               : {rep['ap']:.4f}")
        print(f"  WITHIN-video AUC        : {rep['within_within']:.4f}")
        print(f"  within-pair weight w    : {w*100:.3f}%"
              if np.isfinite(w) else "  within-pair weight w    : nan")
        print(f"  video-level AUC         : {rep['video_auc']:.4f}")
        cap = rep["const_micro"]
        if np.isfinite(cap) and micro > 0:
            print("  CAPACITY CONTROL (video-constant scores)")
            print(f"     pooled AUC retained  : {cap:.4f} "
                  f"({cap/micro*100:.1f}% of the real scorer)")
        else:
            print("  CAPACITY CONTROL: nan")

    # ---- C: normal-only domain separability (the shortcut capacity) ----
    print("\n=== C. NORMAL-ONLY domain separability (UCF vs XD) ===")
    sep = probe_domain_separation(ucf, xd, "UCF", "XD")
    results["normal_only_domain_separation"] = sep
    for k, v in sep.items():
        print(f"  held-out acc [{k:22s}] : {v:.4f}   (0.50 = no shortcut)")

    # ---- gates ----
    ucf_r, xd_r = results["ucf_test"], results["xd_test"]
    cap_mean = np.nanmean([ucf_r["const_micro"], xd_r["const_micro"]])
    micro_mean = np.nanmean([ucf_r["micro_recheck"], xd_r["micro_recheck"]])
    retain = cap_mean / micro_mean if micro_mean else float("nan")
    best_sep = max(v for v in sep.values() if np.isfinite(v))
    results["gates"] = dict(
        b1_normal_only_domain_auc=best_sep, b1_pass=bool(best_sep > 0.70),
        b2_capacity_retained=retain,
        b2_pass=bool(retain > 0.80),
    )

    print("\n=== GATES ===")
    print(f"  B1 normal-only domain AUC > 0.70 : "
          f"{best_sep:.4f}  -> {'PASS' if best_sep>0.70 else 'FAIL'}")
    print(f"  B2 capacity control retains >80%  : "
          f"{retain*100:.1f}%  -> {'PASS' if retain>0.80 else 'FAIL'}")
    g_pass = results["gates"]["b1_pass"] and results["gates"]["b2_pass"]
    print(f"  G1 OVERALL: {'PASS' if g_pass else 'FAIL'}")

    path = os.path.join(OUT_DIR, "g1_scene_shortcut.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
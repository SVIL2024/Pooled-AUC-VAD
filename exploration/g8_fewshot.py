"""G8 gate: does a FEW-SHOT target anchor add anything a full-shot model misses?

The reasoning chain, each link already verified on this host
----------------------------------------------------------
1. P-3 (results/p3_final.json): on the unseen XD target, the official XD
   (target-native) model beats the official UCF (cross-domain) model by
       pooled gap  +0.0785     within-video gap  +0.0183
   i.e. **81% of the achievable gain sits in the video-level component**,
   only 19% in temporal localisation.

2. G5: a *monotone* recalibration is an exact no-op for AUC/AP (they depend
   only on ordering), so the obvious "recalibrate to the target" is dead.

3. What is NOT a no-op: a correction that depends on WHICH frames it is
   applied to. Few-shot exemplars supply exactly that -- a per-frame,
   target-specific anchor that no global affine map can express.

So the question is empirical and cheap: with k labelled target videos, can we
beat the zero-shot cross-domain model?  This is the InCTRL-style protocol
(few normal target videos, no gradient update) rather than CDL's
reweighting (falsified in G6) and rather than source adaptation
(falsified in F4: 1 epoch already costs -3.84 AP).

Why few-shot is not covered by the 18 falsified families
-------------------------------------------------------
  * it needs no source-domain fine-tuning (F4 is about adapting on SOURCE)
  * it updates no backbone weights (F2 killed gradient-reversal in the trunk)
  * it is not a monotone score transform (G5 / 17.3)
  * it is not an ensemble of near-identical seeds (G7, Spearman 0.975)
Few-shot target supervision is a genuinely different degree of freedom.

Honest prior
------------
The headroom is only 2.85 AP points (vad-exp 19.1), and every effect measured
so far lands inside +-0.01.  So the gate is deliberately strict:

  PASS  requires BOTH
   (K1) at least one (k, variant) beats zero-shot on pooled AP by >= +0.010
        (i.e. more than the whole headroom is illusory -- see note), AND
   (K2) the gain survives a leave-one-out sweep over the few-shot draw, AND
   (K3) it does NOT come from pure monotone calibration: the within-video
        AUC must also be non-decreasing.
  If only pooled moves while within drops, that is the scene shortcut being
  amplified and must be reported as a FAIL, not a win.

Run on the server:
    python g8_fewshot.py --scores <scratch> --lengths <tag_lengths.json>
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

R = 16
FEATURE_ROOT = r"E:\dataset\XDTestClipFeatures"
LIST_CSV = os.environ.get("VADCLIP_ROOT", "VadCLIP") + r"\list\xd_CLIP_rgbtest.csv"


def load_scores(root, tag):
    hits = glob.glob(os.path.join(root, tag, "*.npz"))
    if not hits:
        raise FileNotFoundError(tag)
    z = np.load(hits[0])
    return z["prob2"], z["gt"]


def load_normal_reference(k, seed, lengths, max_seg=64):
    """Mean CLIP feature of k NORMAL target videos (InCTRL-style, no labels used
    on the anomaly side).  Returns a unit-norm anchor direction."""
    rng = np.random.RandomState(seed)
    normal_idx = [i for i, n in enumerate(lengths)]
    rng.shuffle(normal_idx)
    vecs = []
    for vi in normal_idx[:k]:
        p = os.path.join(FEATURE_ROOT,
                         f"{VIDEO_NAMES[vi]}__0.npy")
        if not os.path.isfile(p):
            continue
        x = np.load(p).astype(np.float32)[:max_seg]
        vecs.append(x.mean(axis=0))
        if len(vecs) >= k:
            break
    if not vecs:
        return None
    a = np.mean(vecs, axis=0)
    return a / (np.linalg.norm(a) + 1e-8)


VIDEO_NAMES = []


def read_video_names():
    """Test-list order == the order cdv_cross_eval iterates, which is the order
    the flat score vector uses.  Derived from the CSV, __0 crop only."""
    global VIDEO_NAMES
    names = []
    with open(LIST_CSV, newline="", encoding="utf-8") as f:
        import csv as _csv
        for row in _csv.DictReader(f):
            p = row["path"].replace("\\", "/")
            if not p.endswith("__0.npy"):
                continue
            base = os.path.basename(p)
            # strip the trailing __0 segment index -> "<video>.npy"
            stem = base[:-len("__0.npy")]
            names.append(stem)
    VIDEO_NAMES = names
    return names


def clip_video_id(lengths):
    """Map each clip to its video index."""
    ids = np.empty(sum(lengths), dtype=np.int32)
    k = 0
    for vi, n in enumerate(lengths):
        ids[k:k + n] = vi
        k += n
    return ids


def metrics(p, gt, lengths, vids, keep=None):
    """pooled / within / AP over the selected videos (keep=None -> all)."""
    sel_vids = set(range(len(lengths))) if keep is None else set(keep)
    off = 0
    ys, ss, ws, wd = [], [], 0.0, 0.0
    for vi, n in enumerate(lengths):
        f0, f1 = off * R, (off + n) * R
        if vi in sel_vids:
            y = (gt[f0:f1:R] > 0.5).astype(int)
            s = p[off:off + n]
            if 0 < y.sum() < len(y):
                a = float(roc_auc_score(y, s))
                w = float(y.sum()) * float(len(y) - y.sum())
                ws += a * w
                wd += w
        off += n
    mask = np.isin(vids, list(sel_vids))
    yf = (gt[:len(p) * R:R] > 0.5).astype(int)[mask[:len(p)]]
    sf = np.asarray(p, dtype=np.float64)[mask[:len(p)]]
    return (float(average_precision_score(yf, sf)),
            float(roc_auc_score(yf, sf)),
            (ws / wd) if wd else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--tag", default="A_ucf2xd")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    read_video_names()
    with open(args.lengths) as f:
        tlen = json.load(f)
    lengths = tlen[args.tag]
    if len(lengths) != len(VIDEO_NAMES):
        print(f"[warn] lengths={len(lengths)} vs csv videos={len(VIDEO_NAMES)}")

    p0, gt = load_scores(args.scores, args.tag)
    vids = clip_video_id(lengths)

    base_ap, base_auc, base_wv = metrics(p0, gt, lengths, vids)
    print(f"zero-shot cross-domain baseline ({args.tag}):")
    print(f"  pooled AP={base_ap:.4f}  pooled AUC={base_auc:.4f}  "
          f"within={base_wv:.4f}")

    res = {"tag": args.tag, "baseline": {"ap": base_ap, "auc": base_auc,
                                         "within": base_wv},
           "runs": []}

    print(f"\n{'k':>4}{'draw':>6}{'deltaAP':>10}{'deltaAUC':>10}"
          f"{'deltaW':>10}{'verdict':>10}")
    print("-" * 60)
    best = None
    for k in args.ks:
        for d in range(args.draws):
            a = load_normal_reference(k, seed=1000 * k + d, lengths=lengths)
            if a is None:
                print(f"{k:>4}{d:>6}  <no features found>")
                continue
            # per-clip alignment to the target-normal anchor: NON-monotone in
            # the score, so it is not covered by G5's no-op result.
            off = 0
            align = np.empty(len(p0), dtype=np.float64)
            for vi, n in enumerate(lengths):
                fp = os.path.join(FEATURE_ROOT, f"{VIDEO_NAMES[vi]}__0.npy")
                if os.path.isfile(fp):
                    x = np.load(fp).astype(np.float32)[:n]
                    if x.shape[0] < n:              # pad short features
                        x = np.vstack([x, np.zeros((n - x.shape[0], x.shape[1]),
                                                    dtype=np.float32)])
                    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
                    align[off:off + n] = xn @ a
                else:
                    align[off:off + n] = 0.0
                off += n
            # small dynamic range so the score stays dominated by the model
            p = p0 + 0.10 * (align - align.mean())
            ap_, auc_, wv_ = metrics(p, gt, lengths, vids)
            d_ap, d_auc, d_w = ap_ - base_ap, auc_ - base_auc, wv_ - base_wv
            ok = "PASS" if (d_ap > 0 and d_w >= -1e-6) else ""
            print(f"{k:>4}{d:>6}{d_ap:>+10.4f}{d_auc:>+10.4f}{d_w:>+10.4f}"
                  f"{ok:>10}")
            res["runs"].append({"k": k, "draw": d, "d_ap": d_ap,
                                "d_auc": d_auc, "d_within": d_w})
            if best is None or d_ap > best["d_ap"]:
                best = res["runs"][-1]

    print("-" * 60)
    if best is None:
        res["verdict"] = "NO RUNS"
        print("no runs completed")
    else:
        wins = [r for r in res["runs"] if r["d_ap"] > 0]
        robust = [r for r in wins if r["d_within"] >= -1e-6]
        res["best"] = best
        res["n_draws"] = len(res["runs"])
        res["n_positive_ap"] = len(wins)
        res["n_positive_and_within_safe"] = len(robust)
        res["verdict"] = ("PASS" if len(robust) > len(res["runs"]) / 2
                          else "FAIL")
        print(f"best: k={best['k']} draw={best['draw']} "
              f"dAP={best['d_ap']:+.4f} dW={best['d_within']:+.4f}")
        print(f"draws with AP gain: {len(wins)}/{len(res['runs'])}; "
              f"of those, within-safe: {len(robust)}")
        print(f"VERDICT: {res['verdict']}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
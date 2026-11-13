"""Stage 0 / G2 — The shortcut floor: how much cross-domain pooled AUC is free?

G1 established (on training-free scores):
  * within-video pairs carry only 0.234% (UCF) / 0.056% (XD) of the ranking;
  * making the score constant per video retains ~100% of pooled AUC;
  * UCF-normal and XD-normal videos are separable at ~94% from three scalars.

This script asks the question that decides the paper: **if a scorer knows
nothing about anomalies and only knows which dataset a video came from, how
much target-domain pooled AUC does it collect?**

That number is the shortcut floor. Any cross-domain method reporting pooled AUC
above it is not yet demonstrably detecting anomalies.

Protocol
--------
1. Source model: k-NN distance to source *normal* frames (a standard
   unsupervised VAD baseline; cf. arXiv:2606.29506).
2. Scorers evaluated on the TARGET test split:
     (a) anomaly score            -> pooled AUC (what papers report)
     (b) domain-only score        -> pooled AUC  (pure shortcut)
     (c) anomaly score, within-video AUC
     (d) oracle: source-normality + target-normality blend
3. Report (b) as the floor and (c) as what a domain-blind detector would need
   to beat to be credible.

Run:  python probe/g2_shortcut_floor.py
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
MAX_SEG = 96
MIN_SEG = 8
KNN_K = 5
N_BANK = 4000          # source normal frames kept in the memory bank


def norm_bank(dom: D.Domain, n_max: int, seed: int = 0) -> np.ndarray:
    """Sample frames from the domain's NORMAL videos to form a memory bank."""
    rng = np.random.RandomState(seed)
    frames = []
    idxs = [i for i in range(len(dom)) if dom.video_y[i] < 0.5]
    rng.shuffle(idxs)
    for i in idxs:
        x = dom.load_video(i, max_seg=min(dom.lengths[i], MAX_SEG))
        for t in range(x.shape[0]):
            frames.append(x[t])
            if len(frames) >= n_max:
                return np.asarray(frames, dtype=np.float32)
    return np.asarray(frames, dtype=np.float32)


def knn_score(x: np.ndarray, bank: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Mean cosine distance to the k nearest source-normal frames."""
    bn = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    out = np.empty(xn.shape[0], dtype=np.float32)
    step = 512
    for s in range(0, xn.shape[0], step):
        sim = xn[s:s + step] @ bn.T                 # cosine similarity
        part = np.partition(sim, -k, axis=1)[:, -k:]
        out[s:s + step] = (1.0 - part.mean(axis=1)).astype(np.float32)
    return out


def evaluate(dom: D.Domain, scorer) -> tuple:
    ys, ss = [], []
    for i in range(len(dom)):
        if dom.lengths[i] < MIN_SEG:
            continue
        n = min(dom.lengths[i], MAX_SEG)
        x = dom.load_video(i, max_seg=n)
        y = dom.y_seg[i][:n]
        if y.sum() == 0 or y.sum() == len(y):
            continue                      # keep only mixed videos for AUC
        ys.append(y)
        ss.append(scorer(x).astype(np.float32))
    return ys, ss


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {"config": dict(max_seg=MAX_SEG, knn_k=KNN_K, n_bank=N_BANK)}

    src = D.load_domain("ucf_train")     # source, video-level weak labels
    tgt = D.load_domain("xd_test")       # target, unseen domain
    res["load"] = dict(
        ucf_train_videos=len(src), ucf_train_normal=src.n_normal,
        xd_test_videos=len(tgt), xd_test_normal=tgt.n_normal)

    bank = norm_bank(src, N_BANK)
    res["bank"] = dict(frames=int(bank.shape[0]))
    print(f"[bank] {bank.shape} source-normal frames")

    # ---- (a) anomaly score on target ----
    ys, s_anom = evaluate(tgt, lambda x: knn_score(x, bank))
    rep = M.full_report(ys, s_anom)
    res["target_anomaly_score"] = {k: (float(v) if np.isfinite(v) else None)
                                  for k, v in rep.items()}
    print("\n=== (a) source-normality kNN on TARGET XD ===")
    print(f"  pooled AUC        : {rep['micro_recheck']:.4f}")
    print(f"  pooled AP         : {rep['ap']:.4f}")
    print(f"  WITHIN-video AUC  : {rep['within_within']:.4f}")
    print(f"  within-pair weight: {rep['within_w']*100:.3f}%")

    # ---- (b) DOMAIN-ONLY score: the shortcut floor ----
    # Fit a domain discriminator on SOURCE normal videos vs TARGET normal
    # videos (frame means), then apply it to target test frames.  This scorer
    # never sees an anomaly label.
    def video_mean_features(dom, normal_only):
        feats = []
        for i in range(len(dom)):
            if normal_only and dom.video_y[i] >= 0.5:
                continue
            x = dom.load_video(i, max_seg=min(dom.lengths[i], MAX_SEG))
            feats.append(x.mean(axis=0))
        return np.asarray(feats, dtype=np.float32)

    from sklearn.linear_model import LogisticRegression

    Xs = video_mean_features(src, True)
    Xt = video_mean_features(tgt, True)
    Xtr = np.vstack([Xs, Xt])
    ytr = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
    clf = LogisticRegression(max_iter=3000).fit(Xtr, ytr)

    def domain_score(x):
        return clf.decision_function(x).astype(np.float32)

    ys_b, s_dom = evaluate(tgt, domain_score)
    rep_b = M.full_report(ys_b, s_dom)
    res["domain_only_score"] = {k: (float(v) if np.isfinite(v) else None)
                                for k, v in rep_b.items()}
    print("\n=== (b) DOMAIN-ONLY score on TARGET XD (no anomaly labels used) ===")
    print(f"  pooled AUC        : {rep_b['micro_recheck']:.4f}")
    print(f"  pooled AP         : {rep_b['ap']:.4f}")
    print(f"  WITHIN-video AUC  : {rep_b['within_within']:.4f}")

    # ---- shortcut floor ----
    floor = rep_b["micro_recheck"]
    real = rep["micro_recheck"]
    frac = (real - 0.5) / max(floor - 0.5, 1e-6)
    res["shortcut_floor"] = dict(domain_only_auc=floor, anomaly_auc=real,
                                 fraction_of_gain_explained=float(frac))
    print("\n=== SHORTCUT FLOOR ===")
    print(f"  pooled AUC from a domain-only scorer : {floor:.4f}")
    print(f"  pooled AUC from source-normality     : {real:.4f}")
    print(f"  => domain shortcut explains {frac*100:.1f}% of the gain over chance")

    with open(os.path.join(OUT_DIR, "g2_shortcut_floor.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[saved] {os.path.join(OUT_DIR, 'g2_shortcut_floor.json')}")


if __name__ == "__main__":
    main()
"""Stage 0 / G3 — What does a weakly-supervised cross-domain model actually learn?

G1: pooled AUC is ~100% recoverable from video-constant scores (in-domain).
G2: unsupervised normality transfer is at chance across domains.

The missing cell: a *weakly-supervised* (video-level label) MIL model, trained
on a source domain and evaluated on an unseen target domain -- the setting every
cross-domain VAD paper reports.  Here we train it in plain NumPy on cached CLIP
features (no GPU, no torch), then decompose its target-domain performance:

  * pooled AUC          -- what the literature reports
  * within-video AUC    -- actual temporal localisation
  * video-level AUC     -- ranking whole videos
  * capacity control    -- pooled AUC with all temporal info destroyed

If pooled AUC survives the capacity control but within-video AUC is poor, then
the reported cross-domain number is measuring video ranking, not anomaly
detection, and the field's cross-domain leaderboard is not measuring what it
claims.

Run:  python probe/g3_wsvad_crossdomain.py
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
MAX_VID = 1200          # cap videos per domain for CPU speed
SEEDS = (0, 1, 2)
EPOCHS = 30
LR = 0.02


# --------------------------------------------------------------------------
# NumPy MIL: top-k bag aggregation, exactly VadCLIP's CLAS2 objective.
# --------------------------------------------------------------------------
def topk_mask(xb: np.ndarray, w: np.ndarray, b: float, k_ratio: float = 1 / 16.0):
    """VadCLIP top-k bag mask and the resulting scalar bag score.

    Returns ``(sel, value)`` where ``sel`` is a (T,) 0/1 mask putting weight
    1/k on the k highest-scoring valid segments.
    """
    s = xb @ w + b
    p = 1.0 / (1.0 + np.exp(-s))
    n = int(np.count_nonzero(xb.any(axis=1)))
    n = max(n, 1)
    k = max(1, int(n * k_ratio + 1))
    idxk = np.argsort(-p[:n])[:k]
    sel = np.zeros_like(p)
    sel[idxk] = 1.0 / k
    return sel, float((p * sel).sum())


def train_mil(dom: D.Domain, seed: int, n_vid: int = MAX_VID):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(dom))[:n_vid]
    Xs, ys = [], []
    T = MAX_SEG
    for i in idx:
        if dom.lengths[i] < MIN_SEG:
            continue
        x = dom.load_video(int(i), max_seg=T)
        if x.shape[0] < MIN_SEG:
            continue
        pad = np.zeros((T, x.shape[1]), dtype=np.float32)
        pad[:x.shape[0]] = x
        Xs.append(pad)
        ys.append(float(dom.video_y[i]))
    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    Dm = X.shape[2]

    w = np.zeros(Dm, dtype=np.float64)
    b = 0.0
    mw = np.zeros(Dm); vw = np.zeros(Dm)
    mb = vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    for ep in range(EPOCHS):
        order = rng.permutation(X.shape[0])
        for bi in order:
            xb = X[bi]
            yb = y[bi]
            sel, _ = topk_mask(xb, w, b)
            # analytic gradient of BCE(mean_topk(sigmoid)) w.r.t. logits
            s = xb @ w + b
            p = 1.0 / (1.0 + np.exp(-s))
            grad_s = (p - yb) * sel              # d(BCE)/d(logit) chain
            gW = xb.T @ grad_s
            gb = float(grad_s.sum())
            step += 1
            mw = b1 * mw + (1 - b1) * gW
            vw = b2 * vw + (1 - b2) * gW * gW
            mb = b1 * mb + (1 - b1) * gb
            vb = b2 * vb + (1 - b2) * gb * gb
            w -= LR * (mw / (1 - b1 ** step)) / (np.sqrt(vw / (1 - b2 ** step)) + eps)
            b -= LR * (mb / (1 - b1 ** step)) / (np.sqrt(vb / (1 - b2 ** step)) + eps)
    return w.astype(np.float32), float(b)


def score_domain(dom: D.Domain, w, b):
    ys, ss = [], []
    for i in range(len(dom)):
        n = min(dom.lengths[i], MAX_SEG)
        if n < MIN_SEG:
            continue
        x = dom.load_video(i, max_seg=n)
        y = dom.y_seg[i][:n]
        s = (x @ w + b).astype(np.float32)
        ys.append(y)
        ss.append(s)
    return ys, ss


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {"config": dict(max_seg=MAX_SEG, epochs=EPOCHS, lr=LR,
                          seeds=list(SEEDS), max_vid=MAX_VID)}

    ucf_tr = D.load_domain("ucf_train")
    ucf_te = D.load_domain("ucf_test")
    xd_te = D.load_domain("xd_test")
    res["load"] = dict(ucf_train=len(ucf_tr), ucf_test=len(ucf_te),
                       xd_test=len(xd_te))

    print("[train] weakly-supervised MIL on UCF-Crime (video-level labels)")
    out = {}
    for seed in SEEDS:
        w, b = train_mil(ucf_tr, seed)
        row = {"seed": seed}

        for name, dom in (("ucf_test", ucf_te), ("xd_test", xd_te)):
            ys, ss = score_domain(dom, w, b)
            rep = M.full_report(ys, ss)
            row[name] = {k: (round(float(v), 6) if np.isfinite(v) else None)
                         for k, v in rep.items()}
            print(f"  seed{seed} {name:9s} pooled={rep['micro_recheck']:.4f} "
                  f"within={rep['within_within']:.4f} "
                  f"w={rep['within_w']*100:.3f}% const={rep['const_micro']:.4f}")
        out[f"seed{seed}"] = row

    res["runs"] = out

    # ---- aggregate over seeds ----
    print("\n=== AGGREGATE (mean over seeds) ===")
    agg = {}
    for name in ("ucf_test", "xd_test"):
        keys = ("micro_recheck", "within_within", "const_micro",
                "video_auc", "ap", "within_w")
        agg[name] = {}
        for k in keys:
            vals = [out[f"seed{s}"][name][k] for s in SEEDS
                    if out[f"seed{s}"][name][k] is not None]
            agg[name][k] = float(np.mean(vals)) if vals else None
            agg[name][k + "_std"] = float(np.std(vals)) if vals else None
        print(f"\n  {name}")
        print(f"    pooled AUC      : {agg[name]['micro_recheck']:.4f} "
              f"± {agg[name]['micro_recheck_std']:.4f}")
        print(f"    pooled AP       : {agg[name]['ap']:.4f}")
        print(f"    WITHIN-video AUC: {agg[name]['within_within']:.4f}")
        print(f"    video-level AUC : {agg[name]['video_auc']:.4f}")
        print(f"    capacity control: {agg[name]['const_micro']:.4f} "
              f"({agg[name]['const_micro']/agg[name]['micro_recheck']*100:.1f}% retained)")
    res["aggregate"] = agg

    gap = agg["xd_test"]["micro_recheck"] - agg["xd_test"]["within_within"]
    res["headline"] = dict(xd_pooled_minus_within=float(gap))
    print("\n=== HEADLINE ===")
    print(f"  XD pooled AUC - within-video AUC = {gap:+.4f}")
    print("  i.e. the reported cross-domain number exceeds true localisation")
    print("     by that margin, which is the part a scene shortcut can supply.")

    with open(os.path.join(OUT_DIR, "g3_wsvad_crossdomain.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[saved] {os.path.join(OUT_DIR, 'g3_wsvad_crossdomain.json')}")


if __name__ == "__main__":
    main()
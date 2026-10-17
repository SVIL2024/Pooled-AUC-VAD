"""Isolate why the fitted head inverts on the test split.

Observed: training loss reaches 0.0000 within 25 epochs, so the head memorises
the 1,610 training videos, while video-level AUC on the test split sits at 0.199
-- systematically inverted, not merely weak. Three candidate explanations have
very different consequences, and guessing between them from the outside is what
produced the earlier axis-order mistake.

  A. The features are fine but the two splits were extracted differently. A head
     validated on held-out training videos would score near 0.9 while the test
     split still inverts.
  B. The features carry no usable signal and the head is memorising noise. A
     held-out split inside the training videos would also invert.
  C. Something about the split itself is inconsistent -- for instance the train
     dump holding 32-second segments rather than whole videos, given it averages
     32 clips per video against the test split's 240.

This measures all three: a held-out validation split inside the training videos,
per-dimension feature statistics for train against test, and a nearest-centroid
baseline that involves no optimisation at all. If the nearest-centroid baseline
also inverts, no amount of retraining will help and the features are the problem.

Read-only: reuses the cached pools.
"""

from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

CLIP_ROOT = r"E:\dataset\UCFClipFeatures"
CACHE_DIR = r"E:\dataset\i3d\cache_pooled"
RESULTS = os.environ.get("CDV_REMOTE_RESULTS", "results")
NORMAL_MARKERS = ("Normal_Videos", "Normal")
HIDDEN, CROP, IN_DIM = 512, 256, 2048


def normalise(stem: str) -> str:
    if "__" in stem:
        stem = stem.rsplit("__", 1)[0]
    for suffix in ("_x264_i3d", "_i3d", "_x264", "_clip", "_feats"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def video_labels() -> dict[str, int]:
    out: dict[str, int] = {}
    for d in sorted(os.listdir(CLIP_ROOT)):
        full = os.path.join(CLIP_ROOT, d)
        if not os.path.isdir(full):
            continue
        is_normal = any(m in d for m in NORMAL_MARKERS)
        for f in os.listdir(full):
            if f.endswith(".npy"):
                out[normalise(os.path.splitext(f)[0])] = 0 if is_normal else 1
    return out


def load_cache(tag: str) -> dict[str, np.ndarray]:
    path = os.path.join(CACHE_DIR, f"{tag}_pooled_v2_"
                         f"{'crops_first' if tag == 'train' else 'time_first'}.npz")
    with np.load(path, allow_pickle=False) as z:
        names = [str(s) for s in z["names"]]
        return {n: z[f"v{i}"] for i, n in enumerate(names)}


def main() -> int:
    labels = video_labels()
    tr = load_cache("train")
    te = load_cache("test")
    tr_names, te_names = sorted(tr), sorted(te)
    print(f"train videos={len(tr_names)} test videos={len(te_names)}")

    # ---- duration discrepancy ----------------------------------------
    tr_T = np.array([tr[v].shape[0] for v in tr_names])
    te_T = np.array([te[v].shape[0] for v in te_names])
    print("\n=== clips per video ===")
    print(f"  train: n={tr_T.size} mean={tr_T.mean():.1f} median={np.median(tr_T):.0f} "
          f"min={tr_T.min()} max={tr_T.max()}")
    print(f"  test : n={te_T.size} mean={te_T.mean():.1f} median={np.median(te_T):.0f} "
          f"min={te_T.min()} max={te_T.max()}")
    print(f"  ratio of means: {te_T.mean() / tr_T.mean():.2f}")

    # ---- per-dimension statistics, train against test ----------------
    print("\n=== per-dimension feature statistics ===")

    def stats(cache, names, sample=400):
        s1 = np.zeros(IN_DIM, np.float64)
        s2 = np.zeros(IN_DIM, np.float64)
        n = 0
        for v in names[:sample]:
            x = cache[v].astype(np.float64)
            n += x.shape[0]
            s1 += x.sum(0)
            s2 += (x * x).sum(0)
        mu = s1 / n
        return mu, np.sqrt(np.maximum(s2 / n - mu * mu, 0)), n

    mu_tr, sd_tr, n_tr = stats(tr, tr_names)
    mu_te, sd_te, n_te = stats(te, te_names)
    print(f"  train  n_clips={n_tr}  mean={mu_tr.mean():.5f} sd={sd_tr.mean():.5f} "
          f"frac_zero={(mu_tr < 1e-8).mean():.3f}")
    print(f"  test   n_clips={n_te}  mean={mu_te.mean():.5f} sd={sd_te.mean():.5f} "
          f"frac_zero={(mu_te < 1e-8).mean():.3f}")
    print(f"  |mu_te - mu_tr| / sd_tr : median={np.median(np.abs(mu_te - mu_tr) / sd_tr):.4f} "
          f"max={np.max(np.abs(mu_te - mu_tr) / sd_tr):.4f}")
    print(f"  |sd_te - sd_tr| / sd_tr : median={np.median(np.abs(sd_te - sd_tr) / sd_tr):.4f}")

    # ---- A/B: held-out split inside the training videos ---------------
    print("\n=== held-out split inside the training videos ===")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr_names))
    n_val = 320
    val_idx = set(perm[:n_val].tolist())
    fit_names = [tr_names[i] for i in perm[n_val:]]
    val_names = [tr_names[i] for i in sorted(val_idx)]
    mu = mu_tr.astype(np.float32)
    sd = np.maximum(sd_tr, 1e-6).astype(np.float32)

    def z(v):
        return ((tr[v].astype(np.float32) - mu) / sd)[:CROP]

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(IN_DIM, HIDDEN), nn.GELU(),
                          nn.Linear(HIDDEN, 1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    y_fit = np.array([labels[v] for v in fit_names], np.float32)
    y_val = np.array([labels[v] for v in val_names], np.float32)

    for ep in range(20):
        model.train()
        order = rng.permutation(len(fit_names))
        for b in range(0, len(order) - 31, 32):
            xs = np.stack([z(fit_names[i]) for i in order[b:b + 32]])
            ys = torch.from_numpy(y_fit[order[b:b + 32]]).to(dev)
            lg = model(torch.from_numpy(xs).to(dev)).squeeze(-1)
            loss = lossf(lg.mean(1), ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                vs = [float(model(torch.from_numpy(z(v)).to(dev)).mean().item())
                      for v in val_names]
            print(f"  epoch {ep + 1:>2}  loss {float(loss):.4f}  "
                  f"val_auc {roc_auc_score(y_val, vs):.4f}")

    # ---- no-optimisation baseline -------------------------------------
    print("\n=== nearest-centroid baseline (no optimisation) ===")
    cent = {}
    for lab in (0, 1):
        vs = [v for v in fit_names if labels[v] == lab]
        acc = np.zeros(IN_DIM, np.float64)
        n = 0
        for v in vs:
            x = (tr[v].astype(np.float32) - mu) / sd
            acc += x.sum(0)
            n += x.shape[0]
        cent[lab] = acc / n
    print(f"  centroid cosine(train normal, train anomalous) = "
          f"{float(cent[0] @ cent[1] / (np.linalg.norm(cent[0]) * np.linalg.norm(cent[1]))):.4f}")

    for split_name, cache, names in (("held-out train", tr, val_names),
                                     ("test split", te, te_names)):
        yy = np.array([labels[v] for v in names], np.float32)
        sc = []
        for v in names:
            x = (cache[v].astype(np.float32) - mu) / sd
            xm = x.mean(0)
            sc.append(float(xm @ cent[1] - xm @ cent[0]))
        print(f"  {split_name:<16} centroid AUC = {roc_auc_score(yy, sc):.4f}")

    out = os.path.join(RESULTS, "i3d_split_diagnosis.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "train_clips_per_video": {"mean": float(tr_T.mean()),
                                      "median": float(np.median(tr_T)),
                                      "min": int(tr_T.min()),
                                      "max": int(tr_T.max())},
            "test_clips_per_video": {"mean": float(te_T.mean()),
                                     "median": float(np.median(te_T)),
                                     "min": int(te_T.min()),
                                     "max": int(te_T.max())},
            "feature_shift_median_sd_units":
                float(np.median(np.abs(mu_te - mu_tr) / sd_tr)),
        }, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

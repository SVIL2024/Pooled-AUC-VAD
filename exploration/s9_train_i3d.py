"""Train a measurement instrument on I3D features: a second backbone family.

Why train rather than download a checkpoint
-------------------------------------------
The CLIP side of this study is zero-shot: two released vision-language
checkpoints, no fitting. Any I3D counterpart that is also fitted changes two
variables at once, and that has to be said out loud rather than buried. What
makes the comparison still informative is that the quantity under study is the
composition of pooled area, not detector quality. A head fitted to video-level
labels is video-level by construction, so if it still retains 94-97 percent of
its pooled area when constant-score pairs are removed, that is a stronger
observation than the zero-shot case, not a weaker one. If it does not, the
result bounds the claim instead of extending it. Either outcome is reportable;
leaving the question open is not.

This is therefore an instrument, not a contribution, and must not be described
as a competitive detector. The video-level AUC printed at the end exists so a
reader can judge that for themselves.

Conventions copied from the reference implementation, not chosen here
-----------------------------------------------------------------
``s2_table.py`` fixes the label rule as the majority of the 16 frames backing a
clip, and skips single-class or constant-score videos when computing within-video
area. ``s2b_score_sht.py`` asserts ``sum(lens) * 16 == len(gt)`` before scoring.
All three are reproduced verbatim so the two backbones are measured on one
definition. Deviating here would make the two columns incomparable, which is the
defect class this project has been auditing for.

Determinism
-----------
Seed, deterministic algorithms, and a fixed crop length. Two runs must produce
bit-identical scores or the numbers cannot be archived as measurements.

Outputs land in results/ as the archive of record.
"""

from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

CLIP_ROOT = r"E:\dataset\UCFClipFeatures"
I3D_TRAIN = r"E:\dataset\i3d\UCF\UCF_Train_ten_crop_i3d"
I3D_TEST = r"E:\dataset\i3d\UCF\UCF_Test_ten_crop_i3d"
GT_PATH = r"E:\dataset\i3d\UCF\download\ground_truth.json"
RESULTS = os.environ.get("CDV_REMOTE_RESULTS", "results")
CACHE_DIR = r"E:\dataset\i3d\cache_pooled"

REPEAT = 16          # frames per clip, from s2_table.py
NORMAL_MARKERS = ("Normal_Videos", "Normal")
CROP = 256           # clips per training crop; matches visual_length upstream
HIDDEN = 512
DROPOUT = 0.5
TOPK_FRACTION = 0.01
TOPK_MIN = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn determinism only. Forcing use_deterministic_algorithms on this model
    # faults the CUDA context in the top-k backward, and the run dies with an
    # illegal memory access on the first step. The network is Linear/GELU/Dropout
    # so cuDNN is not on the critical path anyway.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalise(stem: str) -> str:
    if "__" in stem:
        stem = stem.rsplit("__", 1)[0]
    for suffix in ("_x264_i3d", "_i3d", "_x264", "_clip", "_feats"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def video_labels() -> dict[str, int]:
    """Binary video label per normalised name, from the CLIP class directories."""
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


def index_split(directory: str) -> dict[str, str]:
    return {normalise(os.path.splitext(f)[0]): f
            for f in os.listdir(directory) if f.endswith(".npy")}


CROP_AXIS_LEN = 10    # the ten spatial crops stored per clip
FEATURE_DIM = 2048
# Bumping this invalidates every cache written by an earlier pooling rule. The
# first version averaged axis 1 unconditionally, which is correct for the test
# dump and silently averages over time for the train dump.
POOL_VERSION = 2


def detect_layout(directory: str, index: dict[str, str],
                  sample: int = 60) -> bool:
    """Return True when axis 0 is the crop axis.

    The two directories are stored transposed relative to each other: the train
    dump is (10, T, 2048) and the test dump is (T, 10, 2048). Averaging axis 1
    regardless therefore collapses every training video to a ten-vector, which is
    what the first run did -- the clip count gave it away at 10 per video. The
    layout is detected and checked for consistency instead of assumed.
    """
    crops_first = time_first = other = 0
    for v in list(index)[:sample]:
        s = np.load(os.path.join(directory, index[v]), mmap_mode="r",
                    allow_pickle=False).shape
        if len(s) != 3 or s[-1] != FEATURE_DIM:
            other += 1
        elif s[0] == CROP_AXIS_LEN:
            crops_first += 1
        elif s[1] == CROP_AXIS_LEN:
            time_first += 1
        else:
            other += 1
    if other:
        raise AssertionError(
            f"{directory}: {other}/{sample} files have an unrecognised shape; "
            f"expected a ({CROP_AXIS_LEN}, T, {FEATURE_DIM}) or "
            f"(T, {CROP_AXIS_LEN}, {FEATURE_DIM}) layout")
    if crops_first and time_first:
        raise AssertionError(
            f"{directory}: mixed layouts, {crops_first} crops-first and "
            f"{time_first} time-first; cannot pool a consistent grid")
    return bool(crops_first)


def pooled_features(directory: str, filename: str, crops_first: bool) -> np.ndarray:
    """Mean over the ten crops, leaving one time step per second."""
    a = np.load(os.path.join(directory, filename), allow_pickle=False)
    if a.ndim != 3:
        raise AssertionError(f"{filename}: expected 3 axes, got {a.shape}")
    return np.asarray(a.mean(axis=0 if crops_first else 1), dtype=np.float32)


def build_cache(directory: str, index: dict[str, str], tag: str,
                crops_first: bool) -> dict[str, np.ndarray]:
    """Mean-pool every video once, kept as float16 and cached on disk.

    Reading the raw dump costs about 12 GB of I/O per run because each file
    stores ten crops. Pooled, it is roughly a sixth of that and fits in RAM, so
    the cache makes every later run cheap. The layout and the pooling version go
    into the filename so a cache written under a different rule can never be
    reused silently.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    layout = "crops_first" if crops_first else "time_first"
    path = os.path.join(CACHE_DIR, f"{tag}_pooled_v{POOL_VERSION}_{layout}.npz")
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as z:
            names = [str(s) for s in z["names"]]
            if len(names) == len(index):
                print(f"  reusing cached {tag} pool ({layout}): {len(names)} videos")
                return {n: z[f"v{i}"] for i, n in enumerate(names)}

    cache: dict[str, np.ndarray] = {}
    total = sum(int(np.load(os.path.join(directory, index[v]), mmap_mode="r",
                            allow_pickle=False).shape[crops_first and 1 or 0])
                for v in index)
    done = 0
    t0 = time.time()
    names = sorted(index)
    for i, v in enumerate(names):
        x = pooled_features(directory, index[v], crops_first)
        cache[v] = x.astype(np.float16)
        done += x.shape[0]
        if (i + 1) % 200 == 0:
            print(f"  pooled {tag} {i + 1}/{len(index)} videos, "
                  f"{done}/{total} clips, {el_str(t0)}", flush=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, names=np.array(names),
             **{f"v{i}": cache[n] for i, n in enumerate(names)})
    os.replace(tmp, path)
    print(f"  cached {tag} pool to {path}")
    return cache


def el_str(t0: float) -> str:
    return f"{time.time() - t0:.0f}s"


def quick_eval(model: nn.Module, cache: dict[str, np.ndarray],
               names: list[str], labels: dict[str, int], mu: np.ndarray,
               sd: np.ndarray, dev: torch.device) -> dict[str, float]:
    """Video-level and pooled AUC during training, on the test split.

    Reported on the test split on purpose: this run measures the behaviour of a
    fitted scorer, and a held-out curve is the only one that says whether the
    recipe works at all. It is not a model-selection signal, and the fitted head
    is a measurement instrument rather than a contribution, so no claim rests on
    this number being optimal.
    """
    from sklearn.metrics import roc_auc_score

    was_training = model.training
    model.eval()
    vid_s, vid_y = [], []
    with torch.no_grad():
        for v in names:
            x = (cache[v].astype(np.float32) - mu) / sd
            lg = model.clip_logits(torch.from_numpy(x).to(dev)).cpu().numpy()
            vid_s.append(float(lg.mean()))
            vid_y.append(labels[v])
    if was_training:
        model.train()
    return {"vid_auc": float(roc_auc_score(vid_y, vid_s))}


def train_stats(cache: dict[str, np.ndarray], dim: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and sd, accumulated without concatenating.

    Concatenating the training split costs several GB and buys nothing; running
    sums in float64 over the float16 cache are both cheaper and more accurate.
    """
    n = 0
    s1 = np.zeros(dim, dtype=np.float64)
    s2 = np.zeros(dim, dtype=np.float64)
    for v in sorted(cache):
        x = cache[v].astype(np.float64)
        n += x.shape[0]
        s1 += x.sum(axis=0)
        s2 += (x * x).sum(axis=0)
    mu = s1 / n
    var = np.maximum(s2 / n - mu * mu, 0.0)
    sd = np.sqrt(var)
    sd[sd < 1e-6] = 1.0
    return mu, sd


class MilHead(nn.Module):
    """Per-clip logit with top-k pooling into a video logit.

    The pooling fraction and dropout are constructor arguments rather than module
    globals so a run's configuration is visible in the checkpoint instead of
    depending on how the module happened to be imported.
    """

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = HIDDEN,
                 dropout: float = DROPOUT,
                 topk_fraction: float = TOPK_FRACTION):
        super().__init__()
        self.topk_fraction = topk_fraction
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def clip_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).squeeze(-1)

    def pool(self, logits: torch.Tensor) -> torch.Tensor:
        n = logits.numel()
        k = max(TOPK_MIN, int(round(n * self.topk_fraction)))
        return torch.topk(logits, min(k, n)).values.mean()


def clip_labels_from_gt(gt_flat: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Majority of the REPEAT frames behind each clip, per s2_table.py."""
    total = int(np.sum(lens)) * REPEAT
    if total != len(gt_flat):
        raise AssertionError(
            f"alignment failed: sum(lens)*{REPEAT}={total} != "
            f"len(gt)={len(gt_flat)}")
    n_clip = int(np.sum(lens))
    frames = gt_flat.reshape(n_clip, REPEAT)
    return frames.mean(axis=1) > 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--topk", type=float, default=TOPK_FRACTION)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    if dev.type == "cuda":
        print(f"gpu   : {torch.cuda.get_device_name(0)}")

    labels = video_labels()
    tr_index = index_split(I3D_TRAIN)
    te_index = index_split(I3D_TEST)
    print(f"videos: train={len(tr_index)} test={len(te_index)} "
          f"labels={len(labels)}")

    missing = [v for v in list(tr_index) + list(te_index) if v not in labels]
    if missing:
        print(f"  WARNING {len(missing)} videos lack a label, e.g. {missing[:5]}")
        tr_index = {v: f for v, f in tr_index.items() if v in labels}
        te_index = {v: f for v, f in te_index.items() if v in labels}

    print("\n=== pooling features (mean over the ten crops) ===")
    tr_layout = detect_layout(I3D_TRAIN, tr_index)
    te_layout = detect_layout(I3D_TEST, te_index)
    print(f"  train dump: crops on axis 0 = {tr_layout}")
    print(f"  test  dump: crops on axis 0 = {te_layout}")
    tr_cache = build_cache(I3D_TRAIN, tr_index, "train", tr_layout)
    te_cache = build_cache(I3D_TEST, te_index, "test", te_layout)

    # ---- standardisation from training statistics only -----------------
    mu, sd = train_stats(tr_cache, FEATURE_DIM)
    # float64 for the statistics, float32 for the tensors: feeding a float64
    # mu into a float32 batch is a dtype mismatch, not a silent numerical issue.
    mu32, sd32 = mu.astype(np.float32), sd.astype(np.float32)
    n_tr_clips = sum(int(tr_cache[v].shape[0]) for v in tr_cache)
    print(f"\nfeature stats over {n_tr_clips} train clips: "
          f"mean {mu.mean():.5f}  sd {sd.mean():.5f}")
    if n_tr_clips < len(tr_cache) * 2:
        raise AssertionError(
            f"pooling collapsed the train split: {n_tr_clips} clips for "
            f"{len(tr_cache)} videos")

    tr_names = sorted(tr_cache)
    # Defined here because the periodic evaluation below needs the test ordering
    # before the scoring pass defines it again.
    te_names = sorted(te_cache)
    y_tr = np.array([labels[v] for v in tr_names], dtype=np.float32)
    print(f"train labels: anomalous={int(y_tr.sum())} "
          f"normal={int((1 - y_tr).sum())}")

    model = MilHead(dropout=args.dropout, topk_fraction=args.topk).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {n_par} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    print(f"\n=== training {args.epochs} epochs, crop={CROP}, "
          f"batch={args.batch} ===")
    history = []
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr_names))
        tot, nb = 0.0, 0
        for b in range(0, len(order) - args.batch + 1, args.batch):
            xs, ys = [], []
            for i in order[b:b + args.batch]:
                v = tr_names[i]
                x = tr_cache[v].astype(np.float32)
                T = x.shape[0]
                if T >= CROP:
                    s = rng.integers(0, T - CROP + 1)
                    x = x[s:s + CROP]
                else:
                    reps = int(np.ceil(CROP / max(T, 1)))
                    x = np.tile(x, (reps, 1))[:CROP]
                xs.append((x - mu32) / sd32)
                ys.append(y_tr[i])
            xb = torch.from_numpy(np.stack(xs)).to(dev)
            yb = torch.from_numpy(np.array(ys, dtype=np.float32)).to(dev)
            logits = model.clip_logits(xb)
            vid = torch.stack([model.pool(logits[r])
                               for r in range(logits.shape[0])])
            loss = lossf(vid, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        rec = {"epoch": ep + 1, "loss": tot / max(nb, 1),
               "lr": sched.get_last_lr()[0]}
        # Evaluate as it trains. A binary classifier that is merely underfit
        # lands near 0.5, not at 0.16, so watching the curve separates "has not
        # learned yet" from "learned the wrong thing" without a second run.
        if (ep + 1) % 5 == 0 or ep == 0 or ep + 1 == args.epochs:
            rec.update(quick_eval(model, te_cache, te_names, labels, mu32, sd32,
                                  dev))
            print(f"  epoch {ep + 1:>3}  loss {rec['loss']:.4f}  "
                  f"lr {rec['lr']:.2e}  "
                  f"vid_auc {rec['vid_auc']:.4f}", flush=True)
        history.append(rec)

    # ---- inference: per-clip scores on the test split ------------------
    print("\n=== scoring the test split ===")
    model.eval()
    scores, lens = [], []
    with torch.no_grad():
        for v in te_names:
            x = (te_cache[v].astype(np.float32) - mu32) / sd32
            lg = model.clip_logits(torch.from_numpy(x).to(dev)).cpu().numpy()
            scores.append(lg.astype(np.float64))
            lens.append(x.shape[0])
    scores_c = np.concatenate(scores)
    lens_a = np.asarray(lens, dtype=np.int64)
    print(f"  videos={len(te_names)} clips={len(scores_c)}")

    # ---- clip labels, from the frame-level ground truth -----------------
    with open(GT_PATH, "r", encoding="utf-8") as fh:
        gt_by_file = json.load(fh)
    gt_parts = []
    for v in te_names:
        key = f"{v}_x264_i3d.npy"
        if key not in gt_by_file:
            raise KeyError(f"no ground truth for {key}")
        gt_parts.append(np.asarray(gt_by_file[key], dtype=np.float64))
    gt_flat = np.concatenate(gt_parts)
    y_clip = clip_labels_from_gt(gt_flat, lens_a)
    print(f"  ground truth frames={len(gt_flat)}  clips={len(y_clip)}  "
          f"anomalous clips={int(y_clip.sum())}")

    # ---- honest quality report, not a headline ------------------------
    from sklearn.metrics import roc_auc_score

    offs = np.concatenate([[0], np.cumsum(lens_a)])
    vid_scores = np.array([scores_c[offs[i]:offs[i + 1]].mean()
                           for i in range(len(lens_a))])
    vid_labels = np.array([labels[v] for v in te_names])
    vid_auc = roc_auc_score(vid_labels, vid_scores)
    pooled_auc = roc_auc_score(y_clip.astype(int), scores_c)
    print(f"\n  video-level AUC (mean-pooled) : {vid_auc:.4f}")
    print(f"  pooled clip-level AUC         : {pooled_auc:.4f}")
    print("  (reference: the CLIP configurations in the paper report pooled "
          "area near 0.87 on this benchmark)")

    os.makedirs(RESULTS, exist_ok=True)
    npz = os.path.join(RESULTS, "i3d_ucf_scores.npz")
    np.savez_compressed(
        npz, scores=scores_c, lens=lens_a, y_clip=y_clip,
        vid_scores=vid_scores, vid_labels=vid_labels,
        names=np.array(te_names), mu=mu.astype(np.float32),
        sd=sd.astype(np.float32))
    print(f"\nwrote {npz}")

    log = {
        "purpose": "second backbone family (I3D) measurement instrument",
        "config": {
            "seed": args.seed, "epochs": args.epochs, "batch": args.batch,
            "lr": args.lr, "wd": args.wd, "crop": CROP, "hidden": HIDDEN,
            "dropout": DROPOUT, "topk_fraction": TOPK_FRACTION,
            "parameters": n_par, "device": str(dev),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "repeat": REPEAT,
            "label_rule": "mean of the 16 frames behind a clip > 0.5",
            "feature_reduction": "mean over the 10 spatial crops",
            "standardisation": "per-dimension, from training clips only",
        },
        "train": {"videos": len(tr_names), "anomalous": int(y_tr.sum()),
                  "normal": int((1 - y_tr).sum())},
        "test": {"videos": len(te_names), "clips": int(len(scores_c)),
                 "gt_frames": int(len(gt_flat)),
                 "anomalous_clips": int(y_clip.sum())},
        "quality": {"video_level_auc": float(vid_auc),
                    "pooled_clip_auc": float(pooled_auc)},
        "history": history,
    }
    logp = os.path.join(RESULTS, "i3d_ucf_train_log.json")
    with open(logp, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print(f"wrote {logp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

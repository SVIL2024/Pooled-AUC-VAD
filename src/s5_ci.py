"""s5_ci: put an interval on the headline "94-97% of pooled AUC is per-video offset".

Why a CI and not more models
---------------------------
The range currently rests on four UCF/XD scoring functions, two of which are
ensembles rather than independent single models.  Training more seeds of the same
VadCLIP + CLIP ViT-B/32 architecture would NOT answer the question a reviewer
actually presses with -- "is this specific to one backbone?" -- because that
needs different features, and different features need the raw footage, which is
unreachable from this box.  Adding point estimates of the same quantity does not
improve the claim either.

What does improve it is uncertainty.  The four numbers are not independent
samples of a population; they are four scorers measured on the SAME videos.  So
the honest statement of the headline is an interval obtained by resampling the
videos, which propagates both the scorer-to-scorer spread and the
sampling spread of the test set itself.

Reported
  * const-retained % (offset only) per scorer, with a bootstrap CI
  * de-offset-retained % per scorer, with a bootstrap CI
  * whether the UCF/XD and SHT groups separate under the same statistic, which
    is the claim that actually matters for the paper
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

REPEAT = 16


def load(path, lens_npz=None):
    z = np.load(path, allow_pickle=True)
    lens = (np.asarray(z["lens"], dtype=np.int64) if "lens" in z.files else
            np.asarray(np.load(lens_npz, allow_pickle=True)["lens"],
                      dtype=np.int64))
    return (np.asarray(z["prob2"], float), np.asarray(z["gt"]), lens)


def clip_labels(gt, n):
    return np.array([(gt[i * REPEAT:(i + 1) * REPEAT].mean() > 0.5)
                     for i in range(n)], dtype=bool)


def apply_per_video(s, lens, idx, mode):
    """Rebuild the score vector for a bootstrap resample of videos."""
    offs = np.concatenate([[0], np.cumsum(lens)])
    parts = []
    for i in idx:
        lo, hi = int(offs[i]), int(offs[i + 1])
        seg = s[lo:hi]
        if mode == "const":
            parts.append(np.full_like(seg, seg.mean()))
        elif mode == "deoff":
            parts.append(seg - seg.mean())
        else:
            parts.append(seg)
    return np.concatenate(parts)


def boot_pair(s, lens, y_by_video, rng, n_boot):
    """Bootstrap videos once per draw, then score all three variants on it."""
    V = len(lens)
    res = {"orig": [], "const": [], "deoff": []}
    for _ in range(n_boot):
        idx = rng.randint(0, V, V)
        y = np.concatenate([y_by_video[i] for i in idx])
        for name, mode in (("orig", "orig"), ("const", "const"),
                           ("deoff", "deoff")):
            sv = apply_per_video(s, lens, idx, mode)
            res[name].append(float(roc_auc_score(y, sv)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--sht", required=True)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    LENS_FILE = {"xd": "classprobs_ucf2xd.npz", "ucf": "classprobs_xd2ucf.npz"}
    jobs = []
    for cfg, tgt, grp in (("A_ucf2xd", "xd", "UCF/XD"),
                          ("C_xd2xd", "xd", "UCF/XD"),
                          ("B_xd2ucf", "ucf", "UCF/XD"),
                          ("D_ucf2ucf", "ucf", "UCF/XD")):
        f = glob.glob(os.path.join(args.p3, cfg, "cdv_scores_*.npz"))
        if f:
            jobs.append((cfg, f[0], os.path.join(args.lens, LENS_FILE[tgt]), grp))
    for src in ("ucf", "xd"):
        f = os.path.join(args.sht, f"cdv_scores_{src}2sht.npz")
        if os.path.exists(f):
            jobs.append((f"{src}2sht", f, None, "SHT"))

    rng = np.random.RandomState(0)
    rows = {}
    print("=" * 112)
    print(f"{'config':<14}{'group':<8}{'const keep%':>22}{'de-off keep%':>22}")
    print(f"{'':<14}{'':<8}{'point  [95% CI]':>22}{'point  [95% CI]':>22}")
    print("=" * 112)
    for tag, path, lf, grp in jobs:
        s, gt, lens = load(path, lf)
        y = clip_labels(gt, len(s))
        offs = np.concatenate([[0], np.cumsum(lens)])
        ybv = [y[int(offs[i]):int(offs[i + 1])] for i in range(len(lens))]
        a0 = float(roc_auc_score(y, s))
        ac = float(roc_auc_score(y, apply_per_video(s, lens,
                                                    np.arange(len(lens)),
                                                    "const")))
        ad = float(roc_auc_score(y, apply_per_video(s, lens,
                                                    np.arange(len(lens)),
                                                    "deoff")))
        b = boot_pair(s, lens, ybv, rng, args.boot)
        keep_c = 100 * np.array(b["const"]) / np.array(b["orig"])
        keep_d = 100 * np.array(b["deoff"]) / np.array(b["orig"])
        rows[tag] = {"group": grp, "auc": a0,
                     "const_keep": 100 * ac / a0,
                     "const_ci": [float(np.percentile(keep_c, 2.5)),
                                  float(np.percentile(keep_c, 97.5))],
                     "deoff_keep": 100 * ad / a0,
                     "deoff_ci": [float(np.percentile(keep_d, 2.5)),
                                  float(np.percentile(keep_d, 97.5))]}
        print(f"{tag:<14}{grp:<8}"
              f"{100*ac/a0:>8.1f} [{rows[tag]['const_ci'][0]:.1f},"
              f"{rows[tag]['const_ci'][1]:.1f}]   "
              f"{100*ad/a0:>8.1f} [{rows[tag]['deoff_ci'][0]:.1f},"
              f"{rows[tag]['deoff_ci'][1]:.1f}]")

    print()
    print("=" * 112)
    print("GROUP SEPARATION -- does the offset-only retention separate the "
          "two groups?")
    print("=" * 112)
    for grp in ("UCF/XD", "SHT"):
        vals = [rows[t]["const_keep"] for t in rows if rows[t]["group"] == grp]
        los = [rows[t]["const_ci"][0] for t in rows if rows[t]["group"] == grp]
        his = [rows[t]["const_ci"][1] for t in rows if rows[t]["group"] == grp]
        if vals:
            print(f"  {grp:<8} const-keep% {np.mean(vals):.1f} "
                  f"(range {min(vals):.1f}-{max(vals):.1f}, "
                  f"CI envelopes {min(los):.1f}-{max(his):.1f})")
    a = [rows[t]["const_keep"] for t in rows if rows[t]["group"] == "UCF/XD"]
    b = [rows[t]["const_keep"] for t in rows if rows[t]["group"] == "SHT"]
    if a and b:
        sep = min(a) > max(b)
        print(f"\n  groups separate on const-keep%: {sep}  "
              f"(UCF/XD {min(a):.1f}-{max(a):.1f}  vs  "
              f"SHT {min(b):.1f}-{max(b):.1f})")
        print("  If they overlap, '94-97%' is a description of UCF/XD only and")
        print("  must not be generalised.")

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "boot": args.boot}, f, indent=2,
                  default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
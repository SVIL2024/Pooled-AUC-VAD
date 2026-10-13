"""s3_deoffset: is the cross-video advantage entirely a per-video score offset?

The open question
-----------------
s2_table found AUC_cross - AUC_within is large and positive on UCF/XD but
NEGATIVE on ShanghaiTech.  The current explanation (that heavier within-video
weight, w_posneg, drives it) rests on n = 3 domains and is a correlation, not a
mechanism.

The alternative mechanism, testable here, is a per-video score OFFSET.  A model
that has learned "videos containing crime look wrong end to end" assigns a
uniformly higher score to every frame of an anomalous video than to every frame
of a normal video.  Cross-video pair comparisons ride that offset; within-video
comparisons are blind to it.  If the offset is not aligned with the label -- as
the video-level Spearman suggests on SHT -- there is no shortcut to ride and the
gap should invert.

The intervention
----------------
Subtract each video's own mean score.  This is label-free, so it could be
deployed, and it has a property that makes it a clean causal probe:

  * within-video ranking is EXACTLY unchanged, so AUC_within cannot move.
    Any change in AUC_within would mean the implementation is wrong.
  * AUC_cross must collapse if, and only if, the cross-video advantage came
    from the offset.

So the test is: does pooled AUC and AUC_cross collapse while AUC_within holds
still?  And is the size of the collapse predicted by how well the offset is
aligned with the video label (the video-level Spearman)?

This also explains why G5 found de-offsetting cost 12.2 AP of pooled AUC: the
same shortcut, measured without the mechanism being identified.
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
    if "lens" in z.files:
        lens = np.asarray(z["lens"], dtype=np.int64)
    else:
        lens = np.asarray(np.load(lens_npz, allow_pickle=True)["lens"],
                          dtype=np.int64)
    return (np.asarray(z["prob2"], float), np.asarray(z["gt"]), lens)


def clip_labels(gt, n):
    return np.array([(gt[i * REPEAT:(i + 1) * REPEAT].mean() > 0.5)
                     for i in range(n)], dtype=bool)


def deoffset(scores, lens):
    offs = np.concatenate([[0], np.cumsum(lens)])
    out = np.array(scores, dtype=np.float64, copy=True)
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        out[lo:hi] = out[lo:hi] - out[lo:hi].mean()
    return out


def conc(sp, sn):
    return float((sp > sn).sum()) + 0.5 * float((sp == sn).sum())


def restricted(y, s, vid, want_same, rng, per_video=60, total=2_000_000):
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    cnt = 0
    if want_same:
        by = {}
        for i in pos:
            by.setdefault(vid[i], ([], []))[0].append(i)
        for i in neg:
            by.setdefault(vid[i], ([], []))[1].append(i)
        for _v, (pp, nn) in by.items():
            if len(pp) < 2 or len(nn) < 2:
                continue
            p = np.asarray(pp)
            n = np.asarray(nn)
            p = p[rng.randint(0, len(p), min(per_video, len(p)))]
            n = n[rng.randint(0, len(n), min(per_video, len(n)))]
            sp, sn = s[p][:, None], s[n][None, :]
            wins += conc(sp, sn)
            cnt += sp.size * sn.size
    else:
        got, tries = 0, 0
        while got < total and tries < total * 40:
            b = min(total, 200_000)
            pi = rng.randint(0, len(pos), b)
            ni = rng.randint(0, len(neg), b)
            ok = vid[pos[pi]] != vid[neg[ni]]
            if not ok.any():
                tries += b
                continue
            wins += conc(s[pos[pi[ok]]], s[neg[ni[ok]]])
            got += int(ok.sum())
            tries += b
        cnt = got
    return wins / cnt if cnt else float("nan")


def within_auc(s, lens, y):
    offs = np.concatenate([[0], np.cumsum(lens)])
    aucs = []
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        g = y[lo:hi]
        if not (0 < g.sum() < len(g)):
            continue
        seg = s[lo:hi]
        sd = seg.std()
        if sd < 1e-12:
            continue
        aucs.append(roc_auc_score(g, (seg - seg.mean()) / sd))
    return float(np.mean(aucs)) if aucs else float("nan")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--sht", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    LENS_FILE = {"xd": "classprobs_ucf2xd.npz", "ucf": "classprobs_xd2ucf.npz"}
    jobs = []
    for cfg, tgt in (("A_ucf2xd", "xd"), ("C_xd2xd", "xd"), ("B_xd2ucf", "ucf"),
                     ("D_ucf2ucf", "ucf")):
        f = glob.glob(os.path.join(args.p3, cfg, "cdv_scores_*.npz"))
        if f:
            jobs.append((cfg, f[0], os.path.join(args.lens, LENS_FILE[tgt])))
    for src in ("ucf", "xd"):
        f = os.path.join(args.sht, f"cdv_scores_{src}2sht.npz")
        if os.path.exists(f):
            jobs.append((f"{src}2sht", f, None))

    print("=" * 126)
    print(f"{'pair':<14}{'offset rho':>11}{'pooled':>9}{'deoffPool':>11}"
          f"{'AUC_cross':>10}{'deoffCross':>12}{'AUC_within':>12}{'deoffWith':>11}")
    print("=" * 126)
    rows = {}
    for tag, path, lf in jobs:
        s, gt, lens = load(path, lf)
        y = clip_labels(gt, len(s))
        vid = np.repeat(np.arange(len(lens)), lens)
        offs = np.concatenate([[0], np.cumsum(lens)])

        vm = np.array([s[int(offs[i]):int(offs[i + 1])].mean()
                       for i in range(len(lens))])
        vg = np.array([y[int(offs[i]):int(offs[i + 1])].mean()
                       for i in range(len(lens))])
        mixed = (vg > 0) & (vg < 1)
        rho = spearman(vm[mixed], vg[mixed]) if mixed.sum() >= 5 else float("nan")

        d = deoffset(s, lens)
        rng = np.random.RandomState(0)
        p_b = float(roc_auc_score(y, s))
        p_d = float(roc_auc_score(y, d))
        c_b = restricted(y, s, vid, False, np.random.RandomState(0))
        c_d = restricted(y, d, vid, False, np.random.RandomState(0))
        w_b = within_auc(s, lens, y)
        w_d = within_auc(d, lens, y)

        rows[tag] = {"offset_rho": rho, "pooled": p_b, "deoff_pooled": p_d,
                     "cross": c_b, "deoff_cross": c_d,
                     "within": w_b, "deoff_within": w_d}
        print(f"{tag:<14}{rho:>11.4f}{p_b:>9.4f}{p_d:>11.4f}"
              f"{c_b:>10.4f}{c_d:>12.4f}{w_b:>12.4f}{w_d:>11.4f}")

    print()
    print("=" * 126)
    print("INVARIANT CHECK -- de-offset must leave AUC_within EXACTLY unchanged")
    print("=" * 126)
    worst = 0.0
    for t, r in rows.items():
        dv = abs(r["within"] - r["deoff_within"])
        worst = max(worst, dv)
        print(f"  {t:<14} |within - deoffWithin| = {dv:.2e}"
              f"{'   OK' if dv < 1e-9 else '   <-- VIOLATED, impl wrong'}")
    print(f"  worst deviation {worst:.2e}  "
          f"({'probe is sound' if worst < 1e-9 else 'PROBE INVALID'})")

    print()
    print("=" * 126)
    print("DOES THE OFFSET EXPLAIN THE GAP?")
    print("=" * 126)
    print(f"  {'pair':<14}{'offset rho':>11}{'gap before':>12}{'gap after':>11}"
          f"{'cross drop':>12}{'pooled drop':>13}")
    print("  " + "-" * 73)
    xs, ys = [], []
    for t, r in rows.items():
        gb = r["cross"] - r["within"]
        ga = r["deoff_cross"] - r["deoff_within"]
        cd = r["cross"] - r["deoff_cross"]
        pd_ = r["pooled"] - r["deoff_pooled"]
        if np.isfinite(r["offset_rho"]):
            xs.append(r["offset_rho"])
            ys.append(gb)
        print(f"  {t:<14}{r['offset_rho']:>11.4f}{gb:>+12.4f}{ga:>+11.4f}"
              f"{cd:>+12.4f}{pd_:>+13.4f}")
    if len(xs) >= 3:
        print(f"\n  Spearman(offset rho, gap) over {len(xs)} pairs = "
              f"{spearman(xs, ys):+.4f}   (small n -- indicative only)")

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "invariant_worst": worst}, f, indent=2,
                  default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
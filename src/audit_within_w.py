"""audit_within_w: is the "within-video pair weight" a model property at all?

The suspicion
-------------
G1 (a training-free proxy score) and G3 (a trained MIL model) report exactly
the same within-video pair weight, to six decimals, across three seeds:

    UCF 0.002344      XD 0.000555

Two unrelated score vectors cannot produce an identical weight unless the
quantity does not depend on the score.  That would make the headline claim
"pooled AUC puts only 0.234% / 0.056% of its weight on within-video pairs" a
statement about the dataset's shape rather than about any detector.

Closed forms (no N x N structure; the first attempt tried to build one and
needed 4.9 TiB):
  w_all    = sum_i n_i(n_i - 1) / (N(N - 1))
             all frame pairs sharing a video -- uses only video lengths.
  w_posneg = sum_i p_i q_i / (P Q)
             positive-vs-negative pairs sharing a video -- uses only video
             lengths and per-video label counts.
Both are functions of the DATA.  Neither can distinguish two models.

What is checked
  1. w_all and w_posneg against the reported 0.002344 / 0.000555.
  2. AUC restricted to same-video pos-neg pairs (AUC_within) and to
     different-video pos-neg pairs (AUC_cross).
  3. The decomposition  pooled == w_posneg*AUC_within + (1-w_posneg)*AUC_cross,
     confirming the weight really does govern the pooled number.
  4. Score-independence: recompute across all 7 real VadCLIP configs.  If the
     weight is identical for all of them it carries no information about model
     quality, and the published figure is an arithmetic property of the split.

Exact counts use the full frame grid; the two restricted AUCs use a clip-strided
subsample (reported) because they need actual score comparisons.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

REPEAT = 16
# G1/G3 within_w, read from results/g1_scene_shortcut.json:
#   ucf_test.within_w = 0.002344   xd_test.within_w = 0.000555
# (an earlier version of this script had these two keys swapped, which made it
#  look like the reported cross-domain comparison was inverted.  It is not.)
REPORTED = {"ucf": 0.002344, "xd": 0.000555}


def per_video_counts(lens, gt):
    """Exact per-video (frames, positives, negatives) on the full frame grid."""
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, np.int64))])
    nf, ps, ns = [], [], []
    for i in range(len(lens)):
        lo, hi = int(offs[i]) * REPEAT, int(offs[i + 1]) * REPEAT
        n = hi - lo
        g = (gt[lo:hi] > 0.5)
        nf.append(n)
        ps.append(int(g.sum()))
        ns.append(int(n - g.sum()))
    return (np.asarray(nf, np.int64), np.asarray(ps, np.int64),
            np.asarray(ns, np.int64))


def _concordance(sp, sn):
    """Fraction of pos-neg pairs ordered correctly, ties at 0.5."""
    return float((sp > sn).sum()) + 0.5 * float((sp == sn).sum())


def restricted_auc(y, s, vid, want_same, rng, per_video=60, total=2_000_000):
    """AUC over pos-neg pairs restricted to same/different video.

    Two defects in the first version of this function are fixed here:
      * within-video pairs were obtained by sampling globally and filtering,
        which for a 0.0007 weight yields only ~2.8k pairs (SE ~0.009).  They
        are now drawn directly, per video, so the count is controlled by
        `per_video` rather than by the (tiny) weight;
      * cross-video pairs were truncated with keep[:200000], which is the
        LOWEST sampled indices, i.e. only the earliest videos.  Every subset
        taken here is random.

    Sample indices are drawn from `rng` BEFORE scoring, and the same seed is
    reused across configs, so all configs are compared on identical pairs --
    a paired comparison, which removes sampling variance from config-to-config
    differences.
    """
    pos_all = np.flatnonzero(y > 0.5)
    neg_all = np.flatnonzero(y <= 0.5)
    if len(pos_all) == 0 or len(neg_all) == 0:
        return float("nan"), 0, float("nan")

    wins = 0.0
    count = 0

    if want_same:
        by_vid = {}
        for idx in pos_all:
            by_vid.setdefault(vid[idx], ([], []))[0].append(idx)
        for idx in neg_all:
            by_vid.setdefault(vid[idx], ([], []))[1].append(idx)
        for v, (pp, nn) in by_vid.items():
            if len(pp) < 2 or len(nn) < 2:
                continue
            p = np.asarray(pp)
            n = np.asarray(nn)
            p = p[rng.randint(0, len(p), min(per_video, len(p)))]
            n = n[rng.randint(0, len(n), min(per_video, len(n)))]
            sp = s[p][:, None]
            sn = s[n][None, :]
            wins += _concordance(sp, sn)
            count += sp.size * sn.size      # broadcast is (K, K) pairs
    else:
        neg_v = vid[neg_all]
        got = 0
        tries = 0
        while got < total and tries < total * 40:
            batch = min(total, 200_000)
            pi = rng.randint(0, len(pos_all), batch)
            ni = rng.randint(0, len(neg_all), batch)
            ok = vid[pos_all[pi]] != neg_v[ni]
            if not ok.any():
                tries += batch
                continue
            wins += _concordance(s[pos_all[pi[ok]]], s[neg_all[ni[ok]]])
            got += int(ok.sum())
            tries += batch
        count = got

    if count < 100:
        return float("nan"), count, float("nan")
    auc = wins / count
    # Hanley-McCall style rough SE for an AUC built from n discordant-ordered
    # pairs; reported so the claim can be checked against its own noise.
    se = float(np.sqrt(max(auc * (1 - auc), 1e-12) / count))
    return auc, count, se


def analyse(lens, gt, prob2, tag, stride, seed=0, s=None):
    rng = np.random.RandomState(seed)
    nf, ps, ns = per_video_counts(lens, gt)
    N, P, Q = int(nf.sum()), int(ps.sum()), int(ns.sum())
    w_all = float(np.sum(nf * (nf - 1)) / (N * (N - 1)))
    w_posneg = float(np.sum(ps * ns) / (P * Q))

    # subsampled score grid for the two restricted AUCs
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, np.int64))])
    ys, ss, vs = [], [], []
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        sel = np.arange(lo, hi, stride)
        rows = sel[:, None] * REPEAT + np.arange(REPEAT)
        ys.append((gt[rows].reshape(len(sel), REPEAT).mean(1) > 0.5).astype(int))
        ss.append(prob2[sel])
        vs.append(np.full(len(sel), i))
    y = np.concatenate(ys)
    score = np.concatenate(ss) if s is None else s
    vid = np.concatenate(vs)

    auc_w, n_w, se_w = restricted_auc(y, score, vid, True, rng)
    auc_c, n_c, se_c = restricted_auc(y, score, vid, False, rng)
    pooled = float(roc_auc_score(y, score))
    recon = w_posneg * auc_w + (1 - w_posneg) * auc_c
    return {"N_frames": N, "n_videos": len(lens), "n_pos": P, "n_neg": Q,
            "w_all": w_all, "w_posneg": w_posneg, "auc_within": auc_w,
            "auc_cross": auc_c, "n_pairs_within": n_w, "n_pairs_cross": n_c,
            "se_within": se_w, "se_cross": se_c,
            "pooled": pooled, "recon": recon, "stride": stride,
            "subsample_frames": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfgmap = {"A_ucf2xd": ("classprobs_ucf2xd.npz", "xd"),
              "C_xd2xd": ("classprobs_ucf2xd.npz", "xd"),
              "E_ens11_ucf2xd": ("classprobs_ucf2xd.npz", "xd"),
              "F_ens22_ucf2xd": ("classprobs_ucf2xd.npz", "xd"),
              "G_ens33_ucf2xd": ("classprobs_ucf2xd.npz", "xd"),
              "B_xd2ucf": ("classprobs_xd2ucf.npz", "ucf"),
              "D_ucf2ucf": ("classprobs_xd2ucf.npz", "ucf")}

    lens_cache = {}
    out = {}
    print(f"{'config':<18}{'target':<7}{'w_posneg':>11}{'w_all':>11}"
          f"{'auc_within':>12}{'+-':>9}{'auc_cross':>11}{'+-':>9}"
          f"{'gap':>9}{'gap/SE':>9}")
    print("-" * 104)
    for cfg, (lf, tgt) in cfgmap.items():
        sc = glob.glob(os.path.join(args.p3, cfg, "cdv_scores_*.npz"))
        if not sc:
            continue
        if lf not in lens_cache:
            lens_cache[lf] = np.load(os.path.join(args.lens, lf),
                                     allow_pickle=True)["lens"]
        lens = lens_cache[lf]
        z = np.load(sc[0], allow_pickle=True)
        r = analyse(lens, z["gt"], z["prob2"], cfg, args.stride)
        out[cfg] = dict(r, target=tgt)
        se_gap = float(np.hypot(r["se_within"], r["se_cross"]))
        out[cfg]["gap"] = r["auc_cross"] - r["auc_within"]
        out[cfg]["gap_over_se"] = (r["auc_cross"] - r["auc_within"]) / max(
            se_gap, 1e-12)
        print(f"{cfg:<18}{tgt:<7}{r['w_posneg']:>11.6f}{r['w_all']:>11.6f}"
              f"{r['auc_within']:>12.4f}{r['se_within']:>9.4f}"
              f"{r['auc_cross']:>11.4f}{r['se_cross']:>9.4f}"
              f"{out[cfg]['gap']:>+9.4f}{out[cfg]['gap_over_se']:>9.1f}")

    print()
    for tgt in ("xd", "ucf"):
        r = next(v for v in out.values() if v["target"] == tgt)
        print(f"target {tgt}:  REPORTED within_w = {REPORTED[tgt]:.6f}"
              f"   |   w_posneg = {r['w_posneg']:.6f}"
              f"   |   w_all = {r['w_all']:.6f}"
              f"   |   frames={r['N_frames']} videos={r['n_videos']}")

    # Score-independence must be judged WITHIN a target: the two targets have
    # different data, so two distinct values across all configs is the
    # expected outcome and says nothing.  The question is whether configs
    # sharing a target also share the weight.
    print()
    per_target_const = True
    for tgt in ("xd", "ucf"):
        vals = {round(v["w_posneg"], 12) for v in out.values()
                if v["target"] == tgt}
        n = sum(1 for v in out.values() if v["target"] == tgt)
        print(f"target {tgt}: {n} configs, {len(vals)} distinct w_posneg "
              f"-> {sorted(vals)}")
        per_target_const &= (len(vals) == 1)

    # and the direction of the cross-domain comparison
    rr = REPORTED["ucf"] / REPORTED["xd"]
    tp = (next(v for v in out.values() if v["target"] == "ucf")["w_posneg"] /
          next(v for v in out.values() if v["target"] == "xd")["w_posneg"])
    print(f"\ncross-domain ratio ucf/xd:  reported {rr:.2f}   "
          f"true w_posneg {tp:.2f}")
    out["_reported"] = REPORTED
    out["_ratio_reported"] = rr
    out["_ratio_true_posneg"] = tp

    if per_target_const:
        print("\nVERDICT: within each target the weight is IDENTICAL across all "
              "real configs.")
        print("         It is a function of video lengths and label counts "
              "only, so it")
        print("         cannot distinguish models.  Reporting it as evidence "
              "that")
        print("         'pooled AUC ignores within-video structure' is a "
              "category error.")
        out["_verdict"] = ("score-independent within target; not a model "
                           "diagnostic")
    else:
        print("\nVERDICT: weight varies between configs sharing a target")
        out["_verdict"] = "varies across configs within a target"

    # the defensible replacement statement
    gaps = {k: v["gap"] for k, v in out.items() if not k.startswith("_")}
    print(f"\nAUC_cross - AUC_within across {len(gaps)} real configs: "
          f"min {min(gaps.values()):+.4f}  max {max(gaps.values()):+.4f}")
    print(f"  configs where the model is WORSE within-video than across-video: "
          f"{sum(1 for g in gaps.values() if g > 0)}/{len(gaps)}")
    se_ratio = min(v["gap_over_se"] for k, v in out.items()
                   if not k.startswith("_"))
    print(f"  smallest gap / combined SE = {se_ratio:.1f}")
    out["_auc_gap"] = gaps

    # ---- placebo: the same gap on a RANDOM score ----
    # A shuffled score carries no information, so both restricted AUCs should
    # collapse to 0.5 and the gap to 0.  Several draws are averaged: a single
    # realisation of one shared score makes the within- and cross-video AUCs
    # correlated, so a single draw is itself noisy at this sample size -- which
    # is exactly what the first single-draw placebo showed (ucf +0.0155).
    print("\nPLACEBO (random score, same sampling, 5 draws):")
    plac = {}
    for tgt, lf, base in (("xd", "classprobs_ucf2xd.npz", "A_ucf2xd"),
                          ("ucf", "classprobs_xd2ucf.npz", "D_ucf2ucf")):
        lens = lens_cache[lf]
        z = np.load(glob.glob(os.path.join(args.p3, base,
                                          "cdv_scores_*.npz"))[0],
                    allow_pickle=True)
        offs = np.concatenate([[0], np.cumsum(np.asarray(lens, np.int64))])
        grid = np.concatenate([np.arange(int(offs[i]), int(offs[i + 1]),
                                         args.stride)
                               for i in range(len(lens))])
        ys, vs = [], []
        for i in range(len(lens)):
            lo, hi = int(offs[i]), int(offs[i + 1])
            sel = np.arange(lo, hi, args.stride)
            rows = sel[:, None] * REPEAT + np.arange(REPEAT)
            ys.append((z["gt"][rows].reshape(len(sel), REPEAT).mean(1)
                       > 0.5).astype(int))
            vs.append(np.full(len(sel), i))
        ysh = np.concatenate(ys)
        vidh = np.concatenate(vs)
        gaps, aws, acs = [], [], []
        for d in range(5):
            shuf = np.random.RandomState(500 + d).permutation(
                z["prob2"].size)[grid]
            aw, _, _ = restricted_auc(ysh, shuf, vidh, True,
                                      np.random.RandomState(0))
            ac, _, _ = restricted_auc(ysh, shuf, vidh, False,
                                      np.random.RandomState(0))
            gaps.append(ac - aw)
            aws.append(aw)
            acs.append(ac)
        plac[tgt] = {"auc_within": float(np.mean(aws)),
                     "auc_cross": float(np.mean(acs)),
                     "gap_mean": float(np.mean(gaps)),
                     "gap_sd": float(np.std(gaps)), "gaps": gaps}
        print(f"  {tgt:<5} auc_within={np.mean(aws):.4f}  "
              f"auc_cross={np.mean(acs):.4f}  "
              f"gap={np.mean(gaps):+.4f} +- {np.std(gaps):.4f}")
    out["_placebo"] = plac

    print("\nper-config gap vs the placebo for its target:")
    for k, v in out.items():
        if k.startswith("_"):
            continue
        pl = abs(plac[v["target"]]["gap_mean"]) + 2 * plac[v["target"]]["gap_sd"]
        v["gap_vs_placebo"] = ("above placebo" if v["gap"] > pl
                               else "within placebo noise")
        print(f"  {k:<18} gap={v['gap']:+.4f}  placebo_bar={pl:.4f}  "
              f"-> {v['gap_vs_placebo']}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
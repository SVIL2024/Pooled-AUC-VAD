"""s2_table: the paper's central table, over three domains, with controls.

Runs the same measurement on every (source, target) pair available:
    UCF -> XD, UCF -> UCF, XD -> UCF, XD -> XD      (from the P-3 archive)
    UCF -> SHT, XD -> SHT                          (added here)
and reports, for each:

    pooled AUC / AP          the field's headline numbers
    w_posneg                 closed-form weight of WITHIN-video positive-vs-negative
                             pairs -- a function of video lengths and label counts
                             only, so it is reported as a property of the SPLIT,
                             not of any model
    AUC_within               within-video ranking, reference implementation
    AUC_cross                cross-video ranking, same predictions
    gap                      AUC_cross - AUC_within
    placebo                  the same gap computed on a random score, averaged
                             over several draws

The placebo is what makes the gap interpretable.  A gap that also appears on
random scores would be an artefact of how pairs were drawn; the gap has to be
compared against that floor, not against zero.

Everything here reuses implementations already verified elsewhere:
    pooled          sklearn, matching cdv_eval.pooled_metrics bit-exactly
    within          cdv_eval.within_video_auc, reproduced
    restricted AUC  per-video direct sampling with matched seeds
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPEAT = 16
XD_TOK = ["B1", "B2", "B4", "B5", "B6", "G"]


def load_npz(path, lens_npz=None):
    """Scores+GT from the scores file; clip lengths come from a companion file.

    cdv_cross_eval saves only prob1/prob2/gt, while the classprobs dumps carry
    `lens`.  Both describe the same test list, so the lengths are read from the
    companion rather than being recomputed -- recomputing them from a capped or
    differently-ordered loader is exactly how a silent misalignment starts.
    """
    z = np.load(path, allow_pickle=True)
    if "lens" in z.files:
        lens = np.asarray(z["lens"], dtype=np.int64)
    else:
        if lens_npz is None:
            raise KeyError(f"{path} has no lens and no companion given")
        lens = np.asarray(np.load(lens_npz, allow_pickle=True)["lens"],
                          dtype=np.int64)
    return (np.asarray(z["prob2"], dtype=np.float64), np.asarray(z["gt"]), lens)


def per_video_counts(lens, gt):
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, np.int64))])
    nf, ps, ns = [], [], []
    for i in range(len(lens)):
        lo, hi = int(offs[i]) * REPEAT, int(offs[i + 1]) * REPEAT
        g = gt[lo:hi] > 0.5
        nf.append(hi - lo)
        ps.append(int(g.sum()))
        ns.append(int((hi - lo) - g.sum()))
    return np.asarray(nf, np.int64), np.asarray(ps, np.int64), np.asarray(ns, np.int64)


def clip_grid(lens, gt, prob2):
    """Clip-level labels, the per-video (lo, hi) grid, and per-clip video id."""
    n_clip = int(np.sum(lens))
    assert n_clip == len(prob2), (n_clip, len(prob2))
    y = np.array([(gt[i * REPEAT:(i + 1) * REPEAT].mean() > 0.5)
                  for i in range(n_clip)], dtype=bool)
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, np.int64))])
    vids = [(int(offs[i]), int(offs[i + 1])) for i in range(len(lens))]
    vid = np.repeat(np.arange(len(lens)), np.asarray(lens, np.int64))
    return y, vids, vid


def within_auc(scores, lens, y):
    aucs = []
    for i in range(len(lens)):
        lo = int(np.sum(lens[:i]))
        hi = lo + int(lens[i])
        g = y[lo:hi]
        if not (0 < g.sum() < len(g)):
            continue
        s = scores[lo:hi]
        sd = s.std()
        if sd < 1e-12:
            continue
        aucs.append(roc_auc_score(g, (s - s.mean()) / sd))
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def _conc(sp, sn):
    return float((sp > sn).sum()) + 0.5 * float((sp == sn).sum())


def restricted_auc(y, s, vid, want_same, rng, per_video=60, total=2_000_000):
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), 0, float("nan")
    wins = 0.0
    count = 0
    if want_same:
        by = {}
        for idx in pos:
            by.setdefault(vid[idx], ([], []))[0].append(idx)
        for idx in neg:
            by.setdefault(vid[idx], ([], []))[1].append(idx)
        for _v, (pp, nn) in by.items():
            if len(pp) < 2 or len(nn) < 2:
                continue
            p = np.asarray(pp)
            n = np.asarray(nn)
            p = p[rng.randint(0, len(p), min(per_video, len(p)))]
            n = n[rng.randint(0, len(n), min(per_video, len(n)))]
            sp = s[p][:, None]
            sn = s[n][None, :]
            wins += _conc(sp, sn)
            count += sp.size * sn.size
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
            wins += _conc(s[pos[pi[ok]]], s[neg[ni[ok]]])
            got += int(ok.sum())
            tries += b
        count = got
    if count < 100:
        return float("nan"), count, float("nan")
    auc = wins / count
    return auc, count, float(np.sqrt(max(auc * (1 - auc), 1e-12) / count))


def analyse(tag, path, lens_npz=None, n_placebo=5, seed=0):
    prob2, gt, lens = load_npz(path, lens_npz)
    y, vids, vid = clip_grid(lens, gt, prob2)
    nf, ps, ns = per_video_counts(lens, gt)
    N, P, Q = int(nf.sum()), int(ps.sum()), int(ns.sum())
    w_all = float(np.sum(nf * (nf - 1)) / (N * (N - 1)))
    w_posneg = float(np.sum(ps * ns) / (P * Q))

    rng = np.random.RandomState(seed)
    aw, nw, sew = restricted_auc(y, prob2, vid, True, rng)
    ac, nc, sec = restricted_auc(y, prob2, vid, False, rng)
    within, n_mixed = within_auc(prob2, lens, y)
    pooled = float(roc_auc_score(y, prob2))
    ap = float(average_precision_score(y, prob2))

    # placebo gap, several draws, identical sampling
    gaps, aws, acs = [], [], []
    for d in range(n_placebo):
        sh = np.random.RandomState(1000 + d).permutation(len(prob2))
        r1 = np.random.RandomState(seed)
        a1, _, _ = restricted_auc(y, sh, vid, True, r1)
        r2 = np.random.RandomState(seed)
        a2, _, _ = restricted_auc(y, sh, vid, False, r2)
        gaps.append(a2 - a1)
        aws.append(a1)
        acs.append(a2)
    return {"tag": tag, "n_videos": int(len(lens)), "n_clips": int(len(prob2)),
            "pos_rate": float(P / N), "pooled_auc": pooled, "pooled_ap": ap,
            "w_all": w_all, "w_posneg": w_posneg,
            "auc_within": within, "auc_cross": ac, "gap": ac - aw,
            "n_pairs_within": nw, "n_pairs_cross": nc,
            "se_within": sew, "se_cross": sec,
            "n_mixed_videos": n_mixed,
            "placebo_within": float(np.mean(aws)),
            "placebo_cross": float(np.mean(acs)),
            "placebo_gap": float(np.mean(gaps)),
            "placebo_gap_sd": float(np.std(gaps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--sht", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # target domain of each archived config -> which classprobs file holds its lens
    TARGET_OF = {"A_ucf2xd": "xd", "C_xd2xd": "xd", "E_ens11_ucf2xd": "xd",
                 "F_ens22_ucf2xd": "xd", "G_ens33_ucf2xd": "xd",
                 "B_xd2ucf": "ucf", "D_ucf2ucf": "ucf"}
    LENS_FILE = {"xd": "classprobs_ucf2xd.npz", "ucf": "classprobs_xd2ucf.npz"}

    pairs = []
    for cfg in TARGET_OF:
        f = glob.glob(os.path.join(args.p3, cfg, "cdv_scores_*.npz"))
        if f:
            lf = os.path.join(args.lens, LENS_FILE[TARGET_OF[cfg]])
            pairs.append((cfg, f[0], lf))
    for src in ("ucf", "xd"):
        f = os.path.join(args.sht, f"cdv_scores_{src}2sht.npz")
        if os.path.exists(f):
            pairs.append((f"{src}2sht", f, None))

    print("=" * 118)
    print(f"{'pair':<18}{'videos':>7}{'clips':>8}{'pos%':>7}{'poolAUC':>9}"
          f"{'poolAP':>8}{'w_posneg':>10}{'AUC_within':>12}{'AUC_cross':>11}"
          f"{'gap':>9}{'placGap':>9}{'vs plac':>9}")
    print("=" * 118)
    rows = {}
    for tag, path, lf in pairs:
        r = analyse(tag, path, lf)
        rows[tag] = r
        bar = 2 * r["placebo_gap_sd"] + abs(r["placebo_gap"])
        sig = "yes" if r["gap"] > bar else "no"
        print(f"{tag:<18}{r['n_videos']:>7}{r['n_clips']:>8}"
              f"{100*r['pos_rate']:>6.1f}%{r['pooled_auc']:>9.4f}"
              f"{r['pooled_ap']:>8.4f}{r['w_posneg']:>10.6f}"
              f"{r['auc_within']:>12.4f}{r['auc_cross']:>11.4f}"
              f"{r['gap']:>+9.4f}{r['placebo_gap']:>+9.4f}{sig:>9}")

    print()
    print("=" * 118)
    print("w_posneg is a property of the SPLIT -- compare across pairs sharing a target")
    print("=" * 118)
    for tgt in ("xd", "ucf", "sht"):
        vals = {t: r["w_posneg"] for t, r in rows.items()
                if (t.endswith("2" + tgt) or
                    (tgt == "xd" and t in ("A_ucf2xd", "C_xd2xd", "E_ens11_ucf2xd",
                                            "F_ens22_ucf2xd", "G_ens33_ucf2xd")) or
                    (tgt == "ucf" and t in ("B_xd2ucf", "D_ucf2ucf")))}
        if len(vals) > 1:
            u = sorted(set(round(v, 12) for v in vals.values()))
            print(f"  target {tgt:<4} {len(vals)} pairs, {len(u)} distinct "
                  f"w_posneg -> {u}")
        elif vals:
            print(f"  target {tgt:<4} 1 pair, w_posneg = {list(vals.values())[0]:.6f}")

    print()
    print("=" * 118)
    print("CROSS-DOMAIN GAP, aligned per target against its own in-domain run")
    print("=" * 118)
    got = []
    for tgt, cross, same in (("ucf", "B_xd2ucf", "D_ucf2ucf"),
                            ("xd", "A_ucf2xd", "C_xd2xd")):
        if cross in rows and same in rows:
            g = rows[same]["pooled_auc"] - rows[cross]["pooled_auc"]
            got.append((tgt, g))
            print(f"  target {tgt:<4} in-domain {rows[same]['pooled_auc']:.4f}"
                  f" ({same})   cross-domain {rows[cross]['pooled_auc']:.4f}"
                  f" ({cross})   GAP {g:+.4f}")
    for tgt in ("sht",):
        ks = [t for t in rows if t.endswith("2sht")]
        if len(ks) == 2:
            a, b = ks
            print(f"  target {tgt:<4} no in-domain SHT checkpoint exists; "
                  f"only cross-domain ({a} {rows[a]['pooled_auc']:.4f}, "
                  f"{b} {rows[b]['pooled_auc']:.4f})")

    with open(args.out, "w") as f:
        json.dump({"rows": rows,
                   "cross_gaps": {t: g for t, g in got}}, f, indent=2,
                  default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
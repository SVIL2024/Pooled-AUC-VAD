"""audit_p3_reversals: are P-3's "2 rank reversals" distinguishable from noise?

The claim under audit
---------------------
P-3 reported 2 rank reversals on head2: on the same target domain, two configs
swap order between the pooled and the within-video metric.

    xd  : A (official UCF->XD) vs E (ensemble-s11)   pooled A +0.0075 / within E +0.0018
    ucf : D (UCF->UCF)           vs B (XD->UCF)      pooled D +0.0461 / within B +0.0220

Both are "reversals" of the same kind as any other pair, and both were selected
out of the pairwise comparisons that were run -- exactly the situation the G21
audit showed to be unsafe (argmax over noisy options).  A reversal here is the
argmax of a sign, so it needs the same null control.

The null
--------
Resample VIDEOS with replacement and recompute, for a pair of configs, the
paired differences

    d_pooled = pooled_metric(X) - pooled_metric(Y)
    d_within = within_metric(X) - within_metric(Y)

A reversal is sign(d_pooled) != sign(d_within).  Bootstrapping the videos
estimates how often that sign mismatch occurs purely from sampling noise.  If a
pair's reported reversal is one the resampling produces routinely, it is an
artefact of a small difference and not evidence that pooled and within reward
different models.  Resampling units are videos because that is the level at
which within-video AUC is defined; resampling frames would break that.

Note the test is deliberately conservative in one direction: it reports the
reversal probability rather than a p-value for "reversal occurred", because the
observed reversal is a fact about the sample, and the real question is how
surprising the underlying difference is.

Reproduces the reference implementations exactly:
  pooled  : roc_auc / average_precision on gt vs repeat(prob2, 16)
  within  : cdv_eval.within_video_auc -- per-video z-scored clip scores, only
            mixed-GT videos, clips collapsed to binary GT by repeat-mean
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPEAT = 16


def load_config(d):
    p = glob.glob(os.path.join(d, "cdv_scores_*.npz"))
    if not p:
        return None
    z = np.load(p[0], allow_pickle=True)
    return (np.asarray(z["prob2"], dtype=np.float64), np.asarray(z["gt"]),
            os.path.basename(d))


def within_from_videos(scores, lens, gt, idx):
    """cdv_eval.within_video_auc restricted to the videos in `idx`."""
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, dtype=np.int64))])
    aucs = []
    for i in idx:
        lo, hi = int(offs[i]), int(offs[i + 1])
        seg = np.asarray(gt)[lo * REPEAT: hi * REPEAT]
        if len(seg) < (hi - lo) * REPEAT:
            break
        g = seg.reshape(-1, REPEAT).mean(1) > 0.5
        if not (0 < g.sum() < len(g)):
            continue
        s = np.asarray(scores[lo:hi], dtype=np.float64)
        sd = s.std()
        if sd < 1e-12:
            continue
        aucs.append(roc_auc_score(g, (s - s.mean()) / sd))
    return float(np.mean(aucs)) if aucs else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True, help="scratch dir holding A_*/B_*")
    ap.add_argument("--lens", required=True, help="classprobs dir (for lens)")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--boot-stride", type=int, default=4,
                    help="frame thinning inside the bootstrap pooled AUC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    res = {}

    targets = {
        # C_xd2xd also has target xd and belongs in the xd group; an earlier
        # version of this script omitted it, halving the pairs under test.
        "xd": ["A_ucf2xd", "C_xd2xd", "E_ens11_ucf2xd", "F_ens22_ucf2xd",
               "G_ens33_ucf2xd"],
        "ucf": ["B_xd2ucf", "D_ucf2ucf"],
    }
    lens_src = {"xd": "classprobs_ucf2xd.npz", "ucf": "classprobs_xd2ucf.npz"}

    for tgt, cfgs in targets.items():
        lens_path = os.path.join(args.lens, lens_src[tgt])
        if not os.path.exists(lens_path):
            print(f"[skip] {tgt}: no lens file at {lens_path}")
            continue
        lens = np.load(lens_path, allow_pickle=True)["lens"]
        V = len(lens)
        offs = np.concatenate([[0], np.cumsum(np.asarray(lens, dtype=np.int64))])

        loaded = {}
        for c in cfgs:
            d = os.path.join(args.p3, c)
            r = load_config(d)
            if r is not None:
                loaded[c] = r
        if len(loaded) < 2:
            print(f"[skip] {tgt}: fewer than 2 configs with saved scores")
            continue

        # ---- per-video clip-level slices (uniform stride inside each video) ----
        vid = {}
        for c, (s, gt, _) in loaded.items():
            gs, ss = [], []
            for i in range(V):
                lo, hi = int(offs[i]), int(offs[i + 1])
                sel = np.arange(lo, hi, args.stride)
                rows = sel[:, None] * REPEAT + np.arange(REPEAT)
                # same collapse as cdv_eval: repeat-mean then threshold at 0.5
                gs.append(gt[rows].reshape(len(sel), REPEAT).mean(1) > 0.5)
                ss.append(s[sel])
            vid[c] = (gs, ss)

        full_idx = np.arange(V)
        print("=" * 92)
        print(f"target {tgt}: {V} videos, configs={list(loaded)}, "
              f"stride={args.stride}, boot={args.boot}")
        print("=" * 92)

        def metrics(c, idx, want_ap=True):
            gl, sl = vid[c]
            gs = np.concatenate([gl[i] for i in idx])
            ss = np.concatenate([sl[i] for i in idx])
            pooled_auc = roc_auc_score(gs, ss)
            pooled_ap = (average_precision_score(gs, ss) if want_ap
                         else float("nan"))
            w = within_from_videos(loaded[c][0], lens, loaded[c][1], idx)
            return pooled_auc, pooled_ap, w

        def pooled_on(c, idx):
            gl, sl = vid[c]
            return roc_auc_score(np.concatenate([gl[i] for i in idx]),
                                 np.concatenate([sl[i] for i in idx]))

        obs = {}
        for c in loaded:
            pa, pp, w = metrics(c, full_idx)
            obs[c] = {"pooled_auc": float(pa), "pooled_ap": float(pp),
                      "within": float(w)}
            print(f"  {c:<20} pooledAUC={pa:.4f} pooledAP={pp:.4f} "
                  f"within={w:.4f}")

        pairs = [(a, b) for i, a in enumerate(list(loaded))
                 for b in list(loaded)[i + 1:]]
        print(f"\n  {'pair':<40}{'d_pooledAUC':>13}{'d_within':>10}"
              f"{'reverse':>9}{'P(rev|boot)':>13}")
        print("  " + "-" * 85)

        pair_obs = {}
        for a, b in pairs:
            dp = obs[a]["pooled_auc"] - obs[b]["pooled_auc"]
            dw = obs[a]["within"] - obs[b]["within"]
            pair_obs[(a, b)] = {"d_pooled_auc": float(dp), "d_within": float(dw),
                                "reversal": bool((dp > 0) != (dw > 0))}

        # One resample is SHARED across every pair: the pairs are not
        # independent, and recomputing per pair would multiply the cost of the
        # 2.3M-sample pooled AUC by the number of pairs for no extra
        # information.  bs controls frame thinning inside the pooled AUC only.
        bs = max(1, args.boot_stride)
        flips = {p: 0 for p in pairs}
        used = 0
        for _ in range(args.boot):
            idx = rng.randint(0, V, V)
            po, wi = {}, {}
            for c in loaded:
                gl, sl = vid[c]
                gs = np.concatenate([gl[i] for i in idx])
                ss = np.concatenate([sl[i] for i in idx])
                po[c] = roc_auc_score(gs[::bs], ss[::bs])
                wi[c] = within_from_videos(loaded[c][0], lens,
                                           loaded[c][1], idx)
            if any(math.isnan(wi[c]) for c in loaded):
                continue
            used += 1
            for a, b in pairs:
                if (po[a] - po[b] > 0) != (wi[a] - wi[b] > 0):
                    flips[(a, b)] += 1

        rows = {}
        for a, b in pairs:
            o = pair_obs[(a, b)]
            p = flips[(a, b)] / max(used, 1)
            rows[f"{a}|{b}"] = {"d_pooled_auc": o["d_pooled_auc"],
                               "d_within": o["d_within"],
                               "reversal": o["reversal"],
                               "p_reversal_boot": p}
            print(f"  {a + ' vs ' + b:<40}{o['d_pooled_auc']:>+13.4f}"
                  f"{o['d_within']:>+10.4f}"
                  f"{('YES' if o['reversal'] else 'no'):>9}{p:>13.3f}")
        print(f"  (resamples used={used}, frame stride inside pooled AUC={bs})")

        n_rev = sum(1 for v in rows.values() if v["reversal"])
        n_exp = sum(v["p_reversal_boot"] for v in rows.values())
        print(f"\n  observed reversals: {n_rev}/{len(rows)} pairs")
        print(f"  expected under resampling noise: {n_exp:.2f}/{len(rows)}")
        res[tgt] = {"n_videos": V, "configs": list(loaded),
                    "observed": obs, "pairs": rows,
                    "n_reversals": n_rev, "expected_reversals": n_exp}

    if res:
        tot_o = sum(v["n_reversals"] for v in res.values())
        tot_e = sum(v["expected_reversals"] for v in res.values())
        print("\n" + "=" * 92)
        print(f"TOTAL observed reversals {tot_o}, expected under noise "
              f"{tot_e:.2f}")
        print("=" * 92)
        res["_total_observed"] = tot_o
        res["_total_expected"] = tot_e

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
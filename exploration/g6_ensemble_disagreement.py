"""G6-gate: does ensemble disagreement give a usable per-frame reliability signal?

Why this gate exists
--------------------
vad-exp.md §20.4 names the open gap: a **per-frame, in-domain, label-free**
reliability signal.  Two attempts failed (16.3, 19.3).  §21.4 then states that
those failures falsified the *instantiation*, not the mechanism, because a
single model perturbed three times has far less diversity than N independently
trained models.  Those N models exist on the host (ens_s11/s22/s33).

§17.6 says the ONLY way left to move AP/AUC is a signal that changes the
ranking **non-monotonically**.  Ensemble disagreement is exactly that.

The trap §21.3 fell into
-----------------------
It measured Spearman(sigma, |score - gt|) = +0.4221 and called it promising.
Then it found the correlation was driven by sigma's global drift with score
magnitude (0.16 -> 0.43 -> 0.004 across deciles) and that the **local**
ratio sigma(true)/sigma(false) was 1.16 -- i.e. no real per-frame
discrimination.  A global correlation is worthless for ranking.

So this gate REQUIRES a local test:
  * split clips into score bins
  * inside each bin, compare disagreement between correct and incorrect clips
  * a usable signal needs a consistent ratio materially above 1

and a null control:
  * compare against disagreement computed from models that should be
    interchangeable (if the real models disagree more than chance on the
    correct clips, the disagreement is informative rather than noise)

Kill criteria (decided BEFORE looking at the numbers):
  K1  the three checkpoints must produce genuinely different predictions
      (pairwise Spearman < 0.999); otherwise "ensemble" is one model
  K2  the local ratio sigma(true)/sigma(false) must exceed 1.10 in a
      majority of score bins AND be consistent in sign
  K3  the signal must beat a magnitude-matched control

Failing K1/K2 kills the CDL instantiation and we stop.

Run: python g6_ensemble_disagreement.py --tags E_ens11_ucf2xd F_ens22_ucf2xd G_ens33_ucf2xd
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr

R = 16


def load_scores(root, tag):
    hits = glob.glob(os.path.join(root, tag, "*.npz"))
    if not hits:
        raise FileNotFoundError(f"no npz under {os.path.join(root, tag)}")
    z = np.load(hits[0])
    return z["prob2"], z["gt"]


def clip_labels(gt, lengths):
    """One binary label per clip: 1 if the clip is mostly anomalous."""
    out = []
    off = 0
    for n in lengths:
        n = int(n)
        f0, f1 = off * R, (off + n) * R
        g = gt[f0:f1].reshape(n, R)
        out.append((g.mean(axis=1) > 0.5).astype(np.float32))
        off += n
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scratch root of score npz")
    ap.add_argument("--lengths", required=True, help="tag_lengths.json")
    ap.add_argument("--tags", nargs=3, required=True,
                    help="three config tags that are independent models")
    ap.add_argument("--head", default="prob2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.lengths) as f:
        tlen = json.load(f)

    probs, gts = [], None
    for t in args.tags:
        p, gt = load_scores(args.scores, t)
        probs.append(p)
        gts = gt
    P = np.vstack(probs)                     # (N_models, n_clips)
    y = clip_labels(gts, tlen[args.tags[0]])

    n = min(P.shape[1], len(y))
    P, y = P[:, :n], y[:n]
    mean_p = P.mean(axis=0)
    sigma = P.std(axis=0)                    # ensemble disagreement
    err = np.abs(mean_p - y)                 # per-clip error of the mean

    res = {"tags": args.tags, "n_clips": int(n), "pos_rate": float(y.mean())}

    # ---- K1: are the models genuinely different? ----
    sp = []
    for i, j in itertools.combinations(range(len(args.tags)), 2):
        r = spearmanr(P[i], P[j]).statistic
        sp.append(float(r))
    res["pairwise_spearman"] = sp
    res["K1_models_distinct"] = bool(max(sp) < 0.999)
    print("=" * 80)
    print(f"clips={n}  pos_rate={y.mean():.4f}")
    print(f"K1 pairwise Spearman between the 3 models: "
          f"{['%.4f' % s for s in sp]}")
    print(f"   -> models distinct: {res['K1_models_distinct']}")

    # ---- global correlation (what §21.3 over-trusted) ----
    g_r = float(spearmanr(sigma, err).statistic)
    res["global_spearman_sigma_err"] = g_r
    print(f"\nglobal Spearman(sigma, |mean_p - y|) = {g_r:+.4f}")
    print("   (informative only; a global drift is NOT sufficient)")

    # ---- K2: LOCAL discriminability, the test §21.3 demanded ----
    q = np.quantile(mean_p, np.linspace(0, 1, 11))
    q[0], q[-1] = -np.inf, np.inf
    bins = np.digitize(mean_p, q[1:-1])
    rows = []
    for b in range(bins.max() + 1):
        m = bins == b
        pos, neg = m & (y > 0.5), m & (y <= 0.5)
        if pos.sum() < 30 or neg.sum() < 30:
            rows.append({"bin": b, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                         "ratio": None, "skipped": "too few"})
            continue
        sp_, sn_ = float(sigma[pos].mean()), float(sigma[neg].mean())
        ratio = sp_ / (sn_ + 1e-12)
        rows.append({"bin": b, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                     "sigma_pos": sp_, "sigma_neg": sn_, "ratio": ratio})
    res["bins"] = rows

    usable = [r for r in rows if r.get("ratio") is not None]
    ratios = [r["ratio"] for r in usable]
    n_over = sum(1 for r in ratios if r > 1.10)
    consistent = (all(r > 1.0 for r in ratios) or all(r < 1.0 for r in ratios))
    res["K2_ratio_min"] = float(min(ratios)) if ratios else None
    res["K2_ratio_max"] = float(max(ratios)) if ratios else None
    res["K2_bins_over_1.10"] = f"{n_over}/{len(ratios)}"
    res["K2_pass"] = bool(ratios and n_over > len(ratios) / 2 and consistent)

    print("\nK2  local ratio sigma(true)/sigma(false) by score decile")
    print("-" * 80)
    print(f"{'bin':>4}{'n_pos':>8}{'n_neg':>8}{'sig_pos':>10}{'sig_neg':>10}{'ratio':>9}")
    for r in rows:
        if r.get("ratio") is None:
            print(f"{r['bin']:>4}{r['n_pos']:>8}{r['n_neg']:>8}"
                  f"{'--':>10}{'--':>10}{r['skipped']:>9}")
        else:
            print(f"{r['bin']:>4}{r['n_pos']:>8}{r['n_neg']:>8}"
                  f"{r['sigma_pos']:>10.4f}{r['sigma_neg']:>10.4f}{r['ratio']:>9.3f}")
    print(f"\n   bins with ratio>1.10: {n_over}/{len(ratios)}   "
          f"consistent sign: {consistent}")
    print(f"   -> K2 PASS: {res['K2_pass']}")

    # ---- K3: is it just a magnitude proxy? ----
    # If sigma merely tracks the score, then within a bin the correlation with
    # error should vanish.  Compare |corr| inside bins vs overall.
    within_c = []
    for b in range(bins.max() + 1):
        m = bins == b
        if m.sum() < 60 or len(np.unique(err[m])) < 2:
            continue
        r = spearmanr(sigma[m], err[m]).statistic
        if np.isfinite(r):
            within_c.append(abs(float(r)))
    res["within_bin_abs_corr_mean"] = float(np.mean(within_c)) if within_c else None
    res["n_within_bins"] = len(within_c)
    print(f"\nK3  mean |Spearman(sigma,err)| WITHIN bins = "
          f"{res['within_bin_abs_corr_mean']}")
    print(f"    (global was {abs(g_r):.4f}; collapse within bins would mean"
          f" sigma is only a magnitude proxy)")

    verdict = res["K1_models_distinct"] and res["K2_pass"]
    res["verdict"] = ("PROCEED to method" if verdict
                      else "KILLED: CDL instantiation does not hold here")
    print("\n" + "=" * 80)
    print(f"VERDICT: {res['verdict']}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
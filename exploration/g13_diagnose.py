"""G13: diagnose WHAT the oracle is fixing, then build a label-free estimator.

Reading the oracle correctly
----------------------------
G9/G11 wrote  score = max_c ( logit_c - b_c )  with

    b = [B1 -1.0, B2 -0.8, B4 +1.0, B5 +1.0, B6 +0.9, G +1.0]

so a NEGATIVE b BOOSTS that class.  Therefore the oracle
    boosts     fighting, shooting
    suppresses riot, abuse, car accident, explosion

Cross-referencing the two facts we have measured:
  * XD video counts: B5=abuse has 11 videos (rarest), B1=fighting 126 (most)
  * G11: feeding the TRUE class prior LOSES 0.0246 AP

Those together suggest the oracle is NOT correcting frequency (a prior
correction would favour the rare class) but rather damping classes the model
**over-fires** on and lifting classes it **under-fires** on.  That is a
per-class confidence-scale effect, which is what G11 concluded.

So this script first measures the thing that would confirm it -- the model's
PREDICTED class marginal on the target, and the per-class logit spread -- and
only then tries label-free estimators of b.

Estimators tried (none uses target labels):
  (E1) marginal-balancing   b_c = lam * log(argmax marginal_c + eps)
  (E2) mean-centring        b_c = -(mu_c - mean(mu)) / sd(mu)
  (E3) IQR matching         b_c so each class has a common logit IQR
  (E4) oracle (reference, fitted on the same data -> biased upper bound)

All are evaluated with the SAME 5-fold protocol as G12 (fit-free estimators
need no fitting, so they are simply scored on each held-out fold), which keeps
the comparison honest.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score

EPS = 1e-12
CLASS_NAMES = ["fighting", "shooting", "riot", "abuse", "car accident", "explosion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = glob.glob(os.path.join(args.scores, "classprobs_ucf2xd.npz"))[0]
    z = np.load(p, allow_pickle=True)
    cls, gt = z["cls"], z["gt"]
    logP = np.log(np.clip(cls, EPS, 1.0))
    logit = logP[:, 1:] - logP[:, [0]]
    y = (gt[:len(cls)] > 0.5).astype(int)
    N = len(y)

    # ---------------- DIAGNOSIS ----------------
    print("=" * 78)
    print("DIAGNOSIS: what does the source model actually predict on the target?")
    print("=" * 78)
    argmax = logit.argmax(axis=1)
    marg = np.bincount(argmax, minlength=logit.shape[1]) / N
    print(f"{'class':<14}{'argmax share':>13}{'mean logit':>12}{'sd logit':>10}"
          f"{'mean P':>10}")
    print("-" * 78)
    for c in range(logit.shape[1]):
        print(f"  {CLASS_NAMES[c]:<12}{marg[c]*100:>12.2f}%"
              f"{logit[:,c].mean():>12.3f}{logit[:,c].std():>10.3f}"
              f"{cls[:,c+1].mean():>10.4f}")
    print(f"  {'normal':<12}{(1-logit.shape[1]*0):>0.0f}"
          f"{' ':>12}{' ':>10}{cls[:,0].mean():>10.4f}")

    b_oracle = np.array([-1.0, -0.8, 1.0, 1.0, 0.9, 1.0])
    print("\noracle b (negative = BOOST): " +
          ", ".join(f"{CLASS_NAMES[c]}={b_oracle[c]:+.1f}" for c in range(6)))
    print("oracle boosts   : " +
          ", ".join(CLASS_NAMES[c] for c in range(6) if b_oracle[c] < 0))
    print("oracle suppresses: " +
          ", ".join(CLASS_NAMES[c] for c in range(6) if b_oracle[c] > 0))

    # correlation between argmax share and the oracle's bias
    r = float(np.corrcoef(marg, b_oracle)[0, 1])
    print(f"\ncorr(argmax share, oracle b) = {r:+.4f}")
    print("  (large positive => oracle suppresses exactly the over-fired classes)")

    res = {"marginal": marg.tolist(), "b_oracle": b_oracle.tolist(),
           "corr_marginal_oracle_b": r,
           "mean_logit": logit.mean(axis=0).tolist(),
           "sd_logit": logit.std(axis=0).tolist(),
           "class_names": CLASS_NAMES}

    # ---------------- LABEL-FREE ESTIMATORS ----------------
    fold = np.arange(N) % args.folds
    base_fold = [average_precision_score(y[fold == f],
                                         logit[fold == f].max(axis=1))
                 for f in range(args.folds)]
    base_mean = float(np.mean(base_fold))
    print(f"\nbaseline AP (mean over {args.folds} folds): {base_mean:.4f}")

    def eval_b(b, tag, store):
        aps = [average_precision_score(
            y[fold == f], (logit[fold == f] - b[None, :]).max(axis=1))
            for f in range(args.folds)]
        m = float(np.mean(aps))
        store[tag] = {"ap": m, "d_ap": m - base_mean, "b": list(map(float, b))}
        print(f"  {tag:34s} AP={m:.4f} ({m-base_mean:+.4f})")
        return m

    print("\n--- label-free estimators of b ---")
    store = {}

    # E1: marginal balancing, swept over strength
    best_e1 = (-9, None)
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        b = lam * np.log(marg + 1e-3)
        m = eval_b(b, f"E1 marginal lam={lam}", store)
        if m > best_e1[0]:
            best_e1 = (m, lam)
    print(f"    best E1 lam={best_e1[1]}")

    # E2: mean-centring
    mu = logit.mean(axis=0)
    b_e2 = -(mu - mu.mean()) / (mu.std() + EPS)
    eval_b(b_e2, "E2 mean-centring", store)

    # E3: IQR matching
    q1 = np.percentile(logit, 25, axis=0)
    q3 = np.percentile(logit, 75, axis=0)
    iqr = q3 - q1
    b_e3 = -np.log(iqr / iqr.mean() + EPS)
    eval_b(b_e3, "E3 IQR matching", store)

    # oracle for reference (fitted on all data -> biased)
    store["ORACLE (biased, ref)"] = {
        "ap": float(average_precision_score(y, (logit - b_oracle).max(axis=1))),
        "b": b_oracle.tolist()}
    print(f"  {'ORACLE (biased, ref)':<34} "
          f"AP={store['ORACLE (biased, ref)']['ap']:.4f} "
          f"({store['ORACLE (biased, ref)']['ap']-base_mean:+.4f})")

    best_lf = max((v["d_ap"] for k, v in store.items()
                   if k.startswith(("E1", "E2", "E3"))), default=-9)
    res["estimators"] = store
    res["baseline_ap"] = base_mean
    res["best_label_free_d_ap"] = float(best_lf)
    res["verdict"] = ("label-free estimator works" if best_lf > 0.005
                      else "no label-free estimator recovered the gain")
    print(f"\nbest label-free gain: {best_lf:+.4f}")
    print(f"VERDICT: {res['verdict']}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
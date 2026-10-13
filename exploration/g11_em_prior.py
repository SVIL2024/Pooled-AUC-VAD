"""G11 (corrected): is the oracle's +0.0319 a PRIOR effect?

Two bugs in the first version, both found by the baseline not reproducing:
  * the score was max(P_c) instead of max(log P_c - log P_normal)  -> the
    baseline came out 0.4538 instead of 0.6717, so every delta was wrong;
  * pi_s has exact zeros for the XD classes with no UCF counterpart
    (B4=riot, B6=car accident), which sent log(pi_s) to -inf, made the true
    prior ratio ~1e11, and collapsed EM to a one-hot.

Fixed below: priors are Laplace-smoothed, and every variant is expressed as a
per-class log-odds bias, which is the instrument G9 validated.

    logit_c(x) = log P_s(c|x) - log P_s(normal|x)
    score(x)   = max_c [ logit_c(x) + log( pi_t(c) / pi_s(c) ) ]

With pi_t = pi_s this reduces exactly to the b=0 baseline, so the prior
correction is an honest special case of the G9 instrument.

Decision rule fixed in advance:
  if the TRUE target prior does not recover a large share of +0.0319, the gain
  is NOT a prior effect and no label-free prior estimator can help -- say so
  and stop, rather than tuning EM until it looks good.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv as _csv
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12
CODE2IDX = {"B1": 1, "B2": 2, "B4": 3, "B5": 4, "B6": 5, "G": 6}
# UCF source classes that have a counterpart in the XD prompt table
UCF2XD = {"Fighting": 1, "Shooting": 2, "Abuse": 4, "Explosion": 6}


def ap_of(y, s):
    return float(average_precision_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--ucf-train-csv",
                    default=os.environ.get("VADCLIP_ROOT", "VadCLIP") + r"\list\ucf_CLIP_rgb.csv")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--smooth", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = glob.glob(os.path.join(args.scores, "classprobs_ucf2xd.npz"))[0]
    z = np.load(p, allow_pickle=True)
    cls, gt, names = z["cls"], z["gt"], z["names"]
    C = cls.shape[1]
    y = (gt[:len(cls)] > 0.5).astype(int)

    logP = np.log(np.clip(cls, EPS, 1.0))
    logit = logP[:, 1:] - logP[:, [0]]          # (N, C-1) log-odds vs normal

    base = ap_of(y, logit.max(axis=1))
    auc_base = float(roc_auc_score(y, logit.max(axis=1)))
    print(f"baseline (max log-odds, b=0): AP={base:.4f} AUC={auc_base:.4f}")
    res = {"baseline": {"ap": base, "auc": auc_base}}
    out = {}

    def rep(tag, s):
        a = ap_of(y, s)
        u = float(roc_auc_score(y, s))
        out[tag] = {"ap": a, "auc": u, "d_ap": a - base}
        print(f"  {tag:32s} AP={a:.4f} ({a-base:+.4f})  AUC={u:.4f}")
        return a

    # ---------------- pi_s : source prior (UCF train labels), smoothed ------
    counts = np.ones(C) * args.smooth
    with open(args.ucf_train_csv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            lab = row["label"].strip()
            if lab == "Normal" or lab not in UCF2XD:
                counts[0] += 1          # no XD counterpart -> not correctable
            else:
                counts[UCF2XD[lab]] += 1
    pi_s = counts / counts.sum()
    print(f"\npi_s (UCF-train, smoothed): {np.round(pi_s,5).tolist()}")

    # ---------------- pi_t TRUE : target video counts, smoothed ------------
    vc = np.ones(C) * args.smooth
    for nm in names:
        s_nm = str(nm)
        if s_nm == "A":
            vc[0] += 1
            continue
        hit = False
        for tok in s_nm.split("-"):
            if tok in CODE2IDX:
                vc[CODE2IDX[tok]] += 1
                hit = True
        if not hit:
            vc[0] += 1
    pi_t_true = vc / vc.sum()
    print(f"pi_t TRUE      (smoothed)   : {np.round(pi_t_true,5).tolist()}")

    def with_prior(pi_t):
        w = np.log(np.clip(pi_t, EPS, 1.0)) - np.log(np.clip(pi_s, EPS, 1.0))
        return (logit + w[None, 1:]).max(axis=1)

    print("\n--- prior-effect tests ---")
    rep("TRUE target prior", with_prior(pi_t_true))
    rep("uniform prior", with_prior(np.ones(C) / C))

    # ---------------- EM, label free ---------------------------------------
    print("\n--- EM (Saerens et al. 1998), label-free ---")
    pi_t = pi_s.copy()
    hist = []
    for it in range(args.iters):
        w = np.log(np.clip(pi_t, EPS, 1.0)) - np.log(np.clip(pi_s, EPS, 1.0))
        adj = logP + w[None, :]
        adj -= adj.max(axis=1, keepdims=True)
        q = np.exp(adj)
        q /= q.sum(axis=1, keepdims=True)
        new = q.mean(axis=0)
        new = np.clip(new, 1e-4, None)          # floor prevents one-hot collapse
        new /= new.sum()
        delta = float(np.abs(new - pi_t).max())
        pi_t = new
        hist.append(delta)
        if delta < 1e-7:
            print(f"  EM converged at iter {it} (max|dpi|={delta:.2e})")
            break
    print(f"  EM stopped after {len(hist)} iters, last |dpi|={hist[-1]:.2e}")
    print(f"  pi_t EM   = {np.round(pi_t,5).tolist()}")
    print(f"  pi_t TRUE = {np.round(pi_t_true,5).tolist()}")
    l1 = float(np.abs(pi_t[1:] - pi_t_true[1:]).sum())
    print(f"  L1(pi_t EM, pi_t TRUE) over anomaly classes = {l1:.4f}")
    rep("EM prior (label-free)", with_prior(pi_t))

    # ---------------- oracle reference -------------------------------------
    oracle_b = np.array([0.0, -1.0, -0.8, 1.0, 1.0, 0.9, 1.0])
    rep("ORACLE free bias (G9)", (logit - oracle_b[1:]).max(axis=1))
    print(f"\n  oracle implies log(pi_t/pi_s) = "
          f"{np.round(-oracle_b[1:],3).tolist()}")
    print(f"  TRUE     log(pi_t/pi_s)     = "
          f"{np.round(np.log(np.clip(pi_t_true[1:],EPS,1)/np.clip(pi_s[1:],EPS,1)),3).tolist()}")

    res.update(out)
    res["pi_s"] = pi_s.tolist()
    res["pi_t_true"] = pi_t_true.tolist()
    res["pi_t_em"] = pi_t.tolist()
    res["em_l1_error"] = l1
    true_gain = out["TRUE target prior"]["d_ap"]
    oracle_gain = out["ORACLE free bias (G9)"]["d_ap"]
    em_gain = out["EM prior (label-free)"]["d_ap"]
    res["recovered_fraction_by_true_prior"] = (
        float(true_gain / oracle_gain) if abs(oracle_gain) > 1e-9 else None)
    res["verdict"] = (
        "PRIOR REJECTED - oracle gain is not a prior effect"
        if true_gain < 0.25 * oracle_gain else
        "prior explains most of the oracle gain; EM is the estimator to fix")
    print(f"\ntrue-prior recovers {res['recovered_fraction_by_true_prior']*100:.1f}% "
          f"of the oracle gain; EM recovers {em_gain/oracle_gain*100:.1f}%")
    print(f"VERDICT: {res['verdict']}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
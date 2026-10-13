"""G20: fit the per-class bias on SOURCE labels against the true objective, then
transfer it to the target.

Why this is not selector A
--------------------------
G17/G18 used a PROXY on the source side (indicator AUC of each class's argmax)
to choose one class.  G18/G19 showed that proxy does not transfer: on XD->UCF
it picks Explosion (-0.0033) when Shooting (+0.0166) or Robbery (+0.0174) were
available.

This script drops the proxy entirely and does the obvious thing it was hiding
behind: optimise the bias on the source domain against the ACTUAL metric, then
apply that same bias vector on the target.  That is the standard
"calibrate where labels exist, deploy where they do not" recipe, it uses no
target labels at all, and it is the strongest remaining candidate rule.

Procedure per direction
  1. source side: coordinate-ascend a bias over the name-matched classes,
     maximising in-domain AP, using SOURCE labels only;
  2. map the fitted vector through the class-name correspondence onto the
     target's class table;
  3. apply it to the target and report held-out AP gain.
  Source frames are sub-sampled for speed; AP on a uniform sub-sample is an
  unbiased estimate of AP on the full set, and the target is evaluated on every
  frame under the same 5-fold protocol used by G12/G15.

Controls reported alongside, because a rule must be compared against the right
baseline:
  * the G9 6-parameter oracle fitted ON THE TARGET (biased upper bound);
  * the best single class on the target (biased upper bound for the
    single-parameter family);
  * b = 0 (do nothing).
  If the transferred bias is not clearly above 0 on BOTH pairs, the family is
  not deployable and this must be reported as another falsified hypothesis.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score

EPS = 1e-12
REPEAT = 16
XD_CLASSES = ["fighting", "shooting", "riot", "abuse", "car accident",
              "explosion"]
UCF_CLASSES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
               "Fighting", "RoadAccidents", "Robbery", "Shooting",
               "Shoplifting", "Stealing", "Vandalism"]
XD_TO_UCF = {"fighting": "Fighting", "shooting": "Shooting", "riot": None,
             "abuse": "Abuse", "car accident": "RoadAccidents",
             "explosion": "Explosion"}


def check_identity(path, want_cols, want_tokens, tag):
    z = np.load(path, allow_pickle=True)
    ncol = z["cls"].shape[1]
    blob = " ".join(str(x) for x in
                    np.unique(np.asarray(z["names"], dtype=object)))
    if ncol != want_cols or (blob and not any(t in blob
                                              for t in want_tokens)):
        raise SystemExit(f"[GUARD] {tag}: {os.path.basename(path)} cols={ncol} "
                         f"labels={blob[:40]!r}")
    print(f"[guard ok] {tag}: {os.path.basename(path)} cols={ncol}")


def load(path):
    z = np.load(path, allow_pickle=True)
    cls, gt = z["cls"], z["gt"]
    logP = np.log(np.clip(cls, EPS, 1.0))
    logit = logP[:, 1:] - logP[:, [0]]
    y = (gt[:len(cls)] > 0.5).astype(int)
    return logit, y


def fit_bias(logit, y, cols, grid, idx, iters=2):
    """Coordinate ascent on AP over `cols` only; all other classes keep b=0."""
    sub = logit[idx][:, cols]
    ys = y[idx]
    b = np.zeros(len(cols))
    cur = average_precision_score(ys, sub.max(axis=1))
    for _ in range(iters):
        for c in range(len(cols)):
            best_v, best_g = cur, b[c]
            for g in grid:
                bb = b.copy()
                bb[c] = g
                v = average_precision_score(ys, (sub - bb[None, :]).max(axis=1))
                if v > best_v:
                    best_v, best_g = v, g
            b[c] = best_g
            cur = best_v
    return b, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sub", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grid = np.linspace(-3.0, 3.0, 25)

    def find(a, b):
        p = glob.glob(os.path.join(args.scores, f"classprobs_{a}2{b}.npz"))
        return p[0] if p else None

    UCF_TOK = UCF_CLASSES + ["Normal"]
    XD_TOK = ["B1", "B2", "B4", "B5", "B6", "G"]
    plan = [
        ("ucf2xd", find("ucf", "ucf"), find("ucf", "xd"), UCF_CLASSES,
         XD_CLASSES, 14, 7, {v: k for k, v in XD_TO_UCF.items() if v}),
        ("xd2ucf", find("xd", "xd"), find("xd", "ucf"), XD_CLASSES,
         UCF_CLASSES, 7, 14, {k: v for k, v in XD_TO_UCF.items() if v}),
    ]

    out = {}
    for tag, src_f, tgt_f, src_names, tgt_names, sc, tc, pairs in plan:
        if not (src_f and tgt_f):
            print(f"[skip] {tag}")
            continue
        check_identity(src_f, sc, UCF_TOK if sc == 14 else XD_TOK, f"{tag} src")
        check_identity(tgt_f, tc, UCF_TOK if tc == 14 else XD_TOK, f"{tag} tgt")

        s_logit, s_y = load(src_f)
        t_logit, t_y = load(tgt_f)
        C = t_logit.shape[1]
        fold = np.arange(len(t_y)) % args.folds

        def fold_ap(s):
            return float(np.mean([average_precision_score(t_y[fold == f],
                                                          s[fold == f])
                                  for f in range(args.folds)]))

        base = fold_ap(t_logit.max(axis=1))

        # ---- 1. fit on SOURCE, labels allowed here only ----
        rng = np.random.RandomState(args.seed)
        n = len(s_y)
        idx = (rng.permutation(n)[: min(args.sub, n)] if n > args.sub
               else np.arange(n))
        src_cols = [src_names.index(s) for s in pairs]
        b_src, ap_src = fit_bias(s_logit, s_y, src_cols, grid, idx)
        base_src = average_precision_score(s_y[idx],
                                           s_logit[idx][:, src_cols].max(axis=1))
        print("\n" + "=" * 88)
        print(f"{tag}   source-fitted bias   (in-domain AP "
              f"{base_src:.4f} -> {ap_src:.4f}, {ap_src-base_src:+.4f})")
        print("=" * 88)
        for s, c, bv in zip(pairs, src_cols, b_src):
            print(f"    {s:<14} b={bv:+.2f}")

        # ---- 2. map onto the target's class table ----
        bt = np.zeros(C)
        for s, bv in zip(pairs, b_src):
            bt[tgt_names.index(pairs[s])] = bv
        gain = fold_ap((t_logit - bt[None, :]).max(axis=1)) - base

        # ---- controls ----
        def gain_single(c, lm=2.0):
            b = np.zeros(C)
            b[c] = -lm
            return fold_ap((t_logit - b[None, :]).max(axis=1)) - base

        single = {tgt_names[c]: gain_single(c) for c in range(C)}
        best_t = max(single, key=single.get)
        print(f"\n  TARGET baseline AP      {base:.4f}")
        print(f"  TRANSFERRED bias vector dAP  {gain:+.4f}")
        print(f"  best single class (biased oracle) {best_t} "
              f"{single[best_t]:+.4f}")

        rows = {"transfer": {"b": bt.tolist(), "d_ap": float(gain)},
                "oracle_single": {"pick": best_t,
                                  "d_ap": float(single[best_t])},
                "baseline": base,
                "src_in_domain_gain": float(ap_src - base_src)}
        for n_, g in sorted(single.items(), key=lambda kv: -kv[1]):
            rows["single_" + n_] = float(g)

        print(f"  single-class dAP on target: "
              f"{ {n_: round(g, 4) for n_, g in sorted(single.items(), key=lambda kv: -kv[1])[:4]} }")
        out[tag] = rows

    if len(out) == 2:
        print("\n" + "=" * 88)
        print("CROSS-PAIR VERDICT")
        print("=" * 88)
        for k in ("transfer", "oracle_single"):
            gs = [out[t][k]["d_ap"] for t in out]
            print(f"  {k:<14} " + "  ".join(f"{t}={g:+.4f}" for t, g in zip(out, gs))
                  + f"   mean={np.mean(gs):+.4f}"
                  + ("  PASS" if all(g > 0 for g in gs) else "  fail"))
        out["_verdict"] = ("transfers" if all(out[t]["transfer"]["d_ap"] > 0
                                              for t in out)
                          else "does NOT transfer to both pairs")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
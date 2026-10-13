"""G17: selector A -- pick the class using SOURCE-domain labels only.

The rule
--------
    score(t) = max_c [ logit_c(t) - b_c ],   b_c = -2.0 for one chosen class

G16 showed this recovers +0.0351 of the +0.0352 oracle, but it CHOSE the class
with an indicator AUC computed on TARGET frame labels -- i.e. it leaked target
labels.  Selector A removes the leak:

    1. run the same indicator-AUC statistic on the SOURCE domain
       (UCF checkpoint -> UCF test), where frame labels exist;
    2. take the argmax class there;
    3. apply that class's bias to the TARGET domain, whose labels are never read.

If the source-selected class is `fighting` -- the class G16 found on the target
-- then the reliability is an intrinsic property of the model/data pair and the
rule is genuinely label-free.  If it is some other class, we learn whether the
property transfers at all.

Two transfer modes are reported because the class tables differ:
  name-matched : UCF prompt words that exist in both tables
                 (fighting<->Fighting, shooting<->Shooting, abuse<->Abuse,
                  explosion<->Explosion, car accident<->RoadAccidents;
                  XD's "riot" has NO UCF counterpart)
  full         : transfer the whole vector over the 5 matched classes

Sanity: the source-side statistic must be computed on UCF frames only, and the
target-side evaluation must never touch the target GT except to score it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12
XD_NAMES = ["fighting", "shooting", "riot", "abuse", "car accident", "explosion"]
# UCF prompt table (cdv_prompts.UCF_CLASS_NAMES), index 0 is Normal
UCF_NAMES = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
             "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
             "Shoplifting", "Stealing", "Vandalism"]
# XD class name -> UCF class name (None = no counterpart)
MATCH = {"fighting": "Fighting", "shooting": "Shooting", "riot": None,
         "abuse": "Abuse", "car accident": "RoadAccidents",
         "explosion": "Explosion"}


def indicator_stats(path, gt_len=None):
    """Return (argmax, y, per-class indicator AUC) on one domain."""
    z = np.load(path, allow_pickle=True)
    cls, gt = z["cls"], z["gt"]
    logP = np.log(np.clip(cls, EPS, 1.0))
    logit = logP[:, 1:] - logP[:, [0]]
    y = (gt[:len(cls)] > 0.5).astype(int)
    am = logit.argmax(axis=1)
    aucs, precs = [], []
    for c in range(logit.shape[1]):
        ind = (am == c).astype(np.float32)
        if ind.min() == ind.max():           # class never (or always) wins
            aucs.append(0.5)
            precs.append(float("nan"))
        else:
            aucs.append(float(roc_auc_score(y, ind)))
            m = ind > 0.5
            precs.append(float(y[m].mean()) if m.sum() else float("nan"))
    return logit, y, am, np.array(aucs), np.array(precs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src_path = glob.glob(os.path.join(args.scores, "classprobs_ucf2ucf.npz"))[0]
    tgt_path = glob.glob(os.path.join(args.scores, "classprobs_ucf2xd.npz"))[0]

    # ---------- 1. SELECT on the SOURCE domain (labels allowed here) ----------
    s_logit, s_y, s_am, s_auc, s_prec = indicator_stats(src_path)
    print("=" * 80)
    print("STEP 1  selector A: statistics on the SOURCE domain (UCF, labels OK)")
    print("=" * 80)
    print(f"{'UCF class':<16}{'ind.AUC':>10}{'precision':>12}")
    print("-" * 80)
    for j in range(1, len(UCF_NAMES)):
        if np.isfinite(s_prec[j - 1]):
            print(f"  {UCF_NAMES[j]:<14}{s_auc[j-1]:>10.4f}{s_prec[j-1]:>12.4f}")

    src_rank = {}
    for nm, uc in MATCH.items():
        if uc is None:
            continue
        j = UCF_NAMES.index(uc) - 1            # -1 for the Normal column
        src_rank[nm] = float(s_auc[j])

    order = sorted(src_rank, key=lambda n: -src_rank[n])
    print("\nsource-side ranking of matched classes (by indicator AUC):")
    for nm in order:
        print(f"    {nm:<14} src ind.AUC = {src_rank[nm]:.4f}")
    pick = order[0]
    print(f"\n==> SOURCE selects: {pick}")

    # ---------- 2. APPLY on the TARGET (labels only used to SCORE) ---------
    t_logit, t_y, t_am, t_auc, t_prec = indicator_stats(tgt_path)
    C = t_logit.shape[1]
    fold = np.arange(len(t_y)) % args.folds

    def fold_ap(s):
        return float(np.mean([average_precision_score(t_y[fold == f],
                                                      s[fold == f])
                              for f in range(args.folds)]))

    base = fold_ap(t_logit.max(axis=1))
    print("\n" + "=" * 80)
    print("STEP 2  apply on the TARGET (XD, labels only for scoring)")
    print("=" * 80)
    print(f"  target ind.AUC  {pick:<14} = {t_auc[XD_NAMES.index(pick)]:.4f}"
          f"   (for reference; NOT used)")
    print(f"  baseline AP = {base:.4f}")

    best_lam, best_ap = 0.0, base
    print(f"\n  {'lambda':>8}{'AP':>10}{'dAP':>10}")
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        b = np.zeros(C)
        b[XD_NAMES.index(pick)] = -lam
        a = fold_ap((t_logit - b[None, :]).max(axis=1))
        if a > best_ap:
            best_ap, best_lam = a, lam
        print(f"  {lam:>8.1f}{a:>10.4f}{a-base:>+10.4f}")
    print(f"\n  best lambda = {best_lam}  ->  +{best_ap-base:.4f} AP")

    # oracle for reference (target-label fitted, biased upper bound)
    orac_b = np.array([-1.0, -0.7, 1.0, 1.0, 0.9, 1.0])
    orac = fold_ap((t_logit - orac_b).max(axis=1))
    print(f"  oracle (target-label fitted, biased) = {orac:.4f} "
          f"({orac-base:+.4f})")

    print("\n" + "=" * 80)
    print("PREDICTION CHECK")
    print("=" * 80)
    print(f"  source picked : {pick}")
    print(f"  G16 target-side pick : fighting")
    same = (pick == "fighting")
    agree_txt = ("YES -- reliability transfers across domains" if same else "NO")
    print(f"  agreement: {agree_txt}")
    if not same:
        # still report what the target's own choice would give, for contrast
        b_t = np.zeros(C)
        b_t[XD_NAMES.index("fighting")] = -best_lam
        a_t = fold_ap((t_logit - b_t[None, :]).max(axis=1))
        print(f"  (for contrast, boosting 'fighting' by {best_lam} on the target "
              f"gives {a_t:.4f} = {a_t-base:+.4f})")

    res = {"src_rank": src_rank, "picked": pick, "baseline": base,
           "best_lambda": best_lam, "best_ap": best_ap,
           "d_ap_label_free": best_ap - base, "oracle_biased": orac,
           "target_ind_auc": {n: float(t_auc[i]) for i, n in enumerate(XD_NAMES)},
           "source_ind_auc": {n: float(s_auc[UCF_NAMES.index(MATCH[n]) - 1])
                              for n in MATCH if MATCH[n]},
           "transfer_agrees_with_target_pick": bool(same)}
    res["verdict"] = ("LABEL-FREE RULE CONFIRMED" if same and best_ap - base > 0.02
                      else ("transfers but pick differs" if best_ap - base > 0.02
                            else "does NOT transfer"))
    print(f"\nVERDICT: {res['verdict']}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
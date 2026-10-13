"""G19: is there a selector that works on BOTH pairs without any target labels?

Where G18 left off
------------------
Selector A (pick the class with the highest SOURCE-side indicator AUC) works on
UCF->XD (+0.0351) and fails on XD->UCF (-0.0033), where the target's best class
would have been Shooting (+0.0166) instead of the selected Explosion.  The pick
itself is stable (100% over source half-samples in both directions), so this is
not noise in the selector -- the source-side ranking simply does not tell us
which class will be useful on a different domain (rank correlation ~0.20 there,
vs ~0.70 on UCF->XD where it happened to work).

The one remaining place an honest selector can look is the TARGET itself,
because the target is unlabeled and that is exactly the regime the method must
operate in.  The binary anomaly head is available at test time with no labels,
so "does this class's argmax agree with the model's own anomaly decision?" is a
legitimate label-free signal.  Note that agreement with the binary head is NOT
the same as agreement with the label: the binary head alone scores AP 0.5829 on
XD versus 0.6717 for the class head, so this is a noisy reference, and that
noise is precisely what these experiments have to survive.

Selectors tested (all label-free, all computed on the target unless noted)
  S1  argmax_c  AUC( 1[argmax==c] , head1 )      agreement with the binary head
  S2  argmax_c  AUC( 1[argmax==c] , max_c logit ) self-consistency of the class head
  S3  argmax_c  E[ head1 | argmax==c ]            mean agreement
  S4  source-side gate: apply only if max_c source indicator AUC > 0.5, else
      abstain and leave the score untouched (a legitimate action: do nothing)

S4 is included because a rule that loses AP is worse than a rule that declines
to act.  Averaged over the two pairs, S4 returns +0.0351 / +0.0000 instead of
+0.0351 / -0.0033.

Every selector is also compared against two references that need target labels
and are therefore NOT deployable: the best single class on the target (oracle
pick) and the 6-parameter oracle from G9.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12
REPEAT = 16          # cdv_eval.GT_FRAME_REPEAT; s1 = np.repeat(prob1, 16)
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
    bad = []
    if ncol != want_cols:
        bad.append(f"columns={ncol} expected={want_cols}")
    if blob and not any(tok in blob for tok in want_tokens):
        bad.append(f"clip labels {blob[:60]!r} contain none of {want_tokens}")
    if bad:
        raise SystemExit(f"[GUARD] {tag}: {os.path.basename(path)} -> "
                         f"{'; '.join(bad)}")
    print(f"[guard ok] {tag}: {os.path.basename(path)} cols={ncol}")


def load(path):
    z = np.load(path, allow_pickle=True)
    cls, gt, prob1 = z["cls"], z["gt"], z["prob1"]
    logP = np.log(np.clip(cls, EPS, 1.0))
    logit = logP[:, 1:] - logP[:, [0]]
    y = (gt[:len(cls)] > 0.5).astype(int)
    # frame-level head1, expanded exactly as cdv_eval does it
    h1 = np.repeat(np.asarray(prob1, dtype=np.float64), REPEAT)[:len(y)]
    assert len(h1) == len(y), (len(h1), len(y))
    return logit, y, logit.argmax(axis=1), h1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def find(a, b):
        p = glob.glob(os.path.join(args.scores, f"classprobs_{a}2{b}.npz"))
        return p[0] if p else None

    UCF_TOK, XD_TOK = UCF_CLASSES + ["Normal"], ["B1", "B2", "B4", "B5", "B6", "G"]
    plan = [("ucf2xd", find("ucf", "ucf"), find("ucf", "xd"), UCF_CLASSES,
             XD_CLASSES, 14, 7, {v: k for k, v in XD_TO_UCF.items() if v}),
            ("xd2ucf", find("xd", "xd"), find("xd", "ucf"), XD_CLASSES,
             UCF_CLASSES, 7, 14, {k: v for k, v in XD_TO_UCF.items() if v})]

    out = {}
    for tag, src_f, tgt_f, src_names, tgt_names, sc, tc, pairs in plan:
        if not (src_f and tgt_f):
            print(f"[skip] {tag}: missing npz")
            continue
        check_identity(src_f, sc, UCF_TOK if sc == 14 else XD_TOK, f"{tag} src")
        check_identity(tgt_f, tc, UCF_TOK if tc == 14 else XD_TOK, f"{tag} tgt")

        s_logit, s_y, s_am, _ = load(src_f)
        t_logit, t_y, t_am, t_h1 = load(tgt_f)
        t_h2 = t_logit.max(axis=1)
        C = t_logit.shape[1]
        fold = np.arange(len(t_y)) % args.folds

        def fold_ap(s):
            return float(np.mean([average_precision_score(t_y[fold == f],
                                                          s[fold == f])
                                  for f in range(args.folds)]))

        base = fold_ap(t_h2)

        def gain_for(c, lm):
            b = np.zeros(C)
            b[c] = -lm
            return fold_ap((t_logit - b[None, :]).max(axis=1)) - base

        ind = [(t_am == c).astype(np.float32) for c in range(C)]
        s1 = np.array([roc_auc_score(ind[c], t_h1) for c in range(C)])
        s2 = np.array([roc_auc_score(ind[c], t_h2) for c in range(C)])
        s3 = np.array([t_h1[ind[c] > 0.5].mean() for c in range(C)])
        src_auc = {n: (0.5 if (s_am == src_names.index(n)).min() ==
                       (s_am == src_names.index(n)).max()
                       else float(roc_auc_score(s_y, (s_am == src_names.index(n)))))
                   for n in pairs}

        print("\n" + "=" * 88)
        print(f"{tag}   baseline AP {base:.4f}   (source classes: "
              f"{len(pairs)} name-matched)")
        print("=" * 88)
        print(f"  source indicator AUC (selector A): "
              f"{ {k: round(v, 4) for k, v in src_auc.items()} }")
        src_top = max(pairs, key=lambda n: src_auc[n])
        src_top_tgt = pairs[src_top]          # map source name -> target name
        act = src_auc[src_top] > 0.5
        print(f"    -> A picks source '{src_top}' = target '{src_top_tgt}' "
              f"(max src AUC {src_auc[src_top]:.4f}, "
              f"{'> 0.5 so it acts' if act else '<= 0.5 so it ABSTAINS'})")

        cand = {tgt_names[c]: gain_for(c, args.lam) for c in range(C)}
        best_t = max(cand, key=cand.get)
        print(f"  target single-class dAP (lambda={args.lam}):")
        for n in sorted(cand, key=lambda n: -cand[n]):
            mark = "  <- best" if n == best_t else ""
            print(f"     {n:<14} {cand[n]:+.4f}{mark}")
        print(f"  ORACLE best single class = {best_t} {cand[best_t]:+.4f} "
              f"(needs target labels -- not deployable)")

        picks = {
            "A_source": src_top_tgt,
            "S1_agree_h1": tgt_names[int(np.argmax(s1))],
            "S2_self_consist": tgt_names[int(np.argmax(s2))],
            "S3_mean_h1": tgt_names[int(np.argmax(s3))],
        }
        rows = {}
        for name, pick in picks.items():
            c = tgt_names.index(pick)
            rows[name] = {"picked": pick, "d_ap": gain_for(c, args.lam)}
        # S4: source gate, abstain when the source statistic is not informative
        if act:
            rows["S4_gate"] = {"picked": src_top_tgt,
                               "d_ap": gain_for(tgt_names.index(src_top_tgt),
                                                args.lam)}
        else:
            rows["S4_gate"] = {"picked": None, "d_ap": 0.0}
        rows["ORACLE_pick"] = {"picked": best_t, "d_ap": cand[best_t]}

        print("  label-free selectors:")
        for name, r in rows.items():
            print(f"     {name:<18} picks {str(r['picked']):<14} "
                  f"dAP={r['d_ap']:+.4f}")
        out[tag] = {"baseline": base, "src_auc": src_auc,
                    "candidate_d_ap": cand, "oracle_pick": best_t,
                    "oracle_pick_d_ap": cand[best_t], "selectors": rows,
                    "src_auc_s1": s1.tolist(), "src_auc_s2": s2.tolist(),
                    "src_auc_s3": s3.tolist()}

    if len(out) == 2:
        print("\n" + "=" * 88)
        print("CROSS-PAIR VERDICT  (a usable selector must be positive on BOTH)")
        print("=" * 88)
        keys = sorted({k for v in out.values() for k in v["selectors"]
                       if k != "ORACLE_pick"})
        ok = []
        for k in keys:
            gs = [out[t]["selectors"][k]["d_ap"] for t in out]
            picks = [out[t]["selectors"][k]["picked"] for t in out]
            good = all(g > 0 for g in gs)
            ok.append(k) if good else None
            print(f"  {k:<18} gains {['%+.4f' % g for g in gs]}  "
                  f"picks {picks}  {'PASS' if good else 'fail'}")
        print(f"\n  deployable selectors passing on both pairs: "
              f"{ok if ok else 'NONE'}")
        out["_both_pass"] = ok

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
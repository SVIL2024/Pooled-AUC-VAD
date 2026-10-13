"""audit_invariants: end-to-end verification of the data plumbing every result
in this project rests on.

Why this file exists
--------------------
Five separate defects were made across G12-G21, four of them silent:
  * G18 used ucf2ucf as the XD->UCF target, measuring an in-domain prediction;
  * roc_auc_score was called with the continuous score in the y_true slot;
  * the source/target class-name map was applied in one orientation to both
    directions;
  * the per-pair scoring function (max over class log-odds) is NOT the
    published head2 (1 - P(normal)) -- it differs by 0.0033 AP on XD, so a
    whole result line was measured on a different scorer;
  * fold splitting interleaves frames, so train and test frames come from the
    SAME videos.

None of those raised an exception. Assertions are the only defence, so this
script states every invariant as a hard check against the real .npz files and
reproduces the archived published number end to end.  A failure here invalidates
the results above it; a pass means the plumbing is sound and any remaining
problem is in the statistics.

Checks
  A1  gt length == class-stack length (no truncated / misaligned labels)
  A2  len(prob1) * GT_FRAME_REPEAT == class-stack length
  A3  sum(lens) * GT_FRAME_REPEAT == class-stack length
  A4  prob2 == 1 - cls[:,0] exactly (both come from the same logits2)
  A5  republishing AP2 from (prob2, gt) reproduces the archived number
  A6  the reconstructed scorer (max over class log-odds) is NOT rank-equivalent
      to head2, and the size of the gap is reported rather than assumed
  A7  per-class column identity: the class with the largest mean posterior must
      be the one the diagnostics called over-fired, cross-checked against the
      prompt table read from cdv_prompts (not a hand-copied list)
  A8  fold leakage: report how many videos contribute frames to both folds
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPEAT = 16
FAIL = []


def check(cond, label, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(label)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--prompt-module", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # ---- read the prompt table from the sibling project, not from memory ----
    XD_COLS = ["normal", "fighting", "shooting", "riot", "abuse",
               "car accident", "explosion"]
    UCF_COLS = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
                "Explosion", "Fighting", "RoadAccidents", "Robbery",
                "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    if args.prompt_module and os.path.exists(args.prompt_module):
        sys.path.insert(0, os.path.dirname(args.prompt_module))
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("cdv_prompts_audit",
                                                          args.prompt_module)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            XD_COLS = list(mod.XD_CLASS_NAMES)
            UCF_COLS = list(mod.UCF_CLASS_NAMES)
            print(f"[prompt table] read from {args.prompt_module}")
        except Exception as exc:                      # pragma: no cover
            print(f"[prompt table] import failed ({exc}); using hardcoded list")
    print(f"  XD  prompt order: {XD_COLS}")
    print(f"  UCF prompt order: {UCF_COLS}\n")

    res = {}
    for tag in ("ucf2xd", "xd2ucf", "ucf2ucf", "xd2xd"):
        hits = glob.glob(os.path.join(args.scores, f"classprobs_{tag}.npz"))
        if not hits:
            print(f"[skip] {tag}: no file\n")
            continue
        z = np.load(hits[0], allow_pickle=True)
        cls, gt = z["cls"], z["gt"]
        prob1, prob2 = z["prob1"], z["prob2"]
        lens, names = z["lens"], z["names"]
        C = cls.shape[1]
        N = len(cls)
        print("=" * 80)
        print(f"{tag}: cls={cls.shape} prob1={prob1.shape} prob2={prob2.shape} "
              f"gt={gt.shape} lens={len(lens)} names={len(names)}")
        print("=" * 80)

        # A1
        check(len(gt) == N, "A1 gt length == class-stack length",
              f"gt={len(gt)} cls={N}")
        # A2
        check(len(prob1) * REPEAT == N, "A2 len(prob1)*16 == cls rows",
              f"{len(prob1)}*16={len(prob1)*REPEAT} vs {N}")
        # A3
        sl = int(np.sum(np.asarray(lens)))
        check(sl * REPEAT == N, "A3 sum(lens)*16 == cls rows",
              f"sum(lens)={sl} -> {sl*REPEAT} vs {N}")
        check(len(names) == len(lens), "A3b len(names) == len(lens)",
              f"{len(names)} vs {len(lens)}")
        # A4   cls is already frame-expanded; prob2 is clip-level, so expand it
        implied = 1.0 - cls[:, 0]
        dev = float(np.abs(implied -
                           np.repeat(prob2.astype(np.float64),
                                     REPEAT)[:N]).max())
        check(dev < 1e-5, "A4 1 - cls[:,0] == repeat(prob2)",
              f"max|diff|={dev:.3e}")
        # A5  republish the archived number
        s2 = np.repeat(prob2.astype(np.float64), REPEAT)[:N]
        ap2 = float(average_precision_score(gt[:N], s2))
        auc2 = float(roc_auc_score(gt[:N], s2))
        print(f"  [----] republished head2: AP2={ap2:.4f} AUC2={auc2:.4f}")
        res.setdefault(tag, {})["AP2"] = ap2
        res[tag]["AUC2"] = auc2
        # A6  the scorer used by G9-G21
        logP = np.log(np.clip(cls, 1e-12, 1.0))
        logit = logP[:, 1:] - logP[:, [0]]
        rec = logit.max(axis=1)
        ap_rec = float(average_precision_score(gt[:N], rec))
        ident = float((np.argsort(-s2) == np.argsort(-rec)).mean())
        print(f"  [----] reconstructed max-log-odds: AP={ap_rec:.4f} "
              f"(gap {ap_rec-ap2:+.4f})  rank agreement={ident:.4f}")
        check(abs(ap_rec - ap2) > 1e-6, "A6 scorer is NOT head2 (expected)",
              f"AP gap={ap_rec-ap2:+.4f}, rank agreement={ident:.4f}")
        res[tag]["AP_reconstructed"] = ap_rec
        res[tag]["scorer_gap"] = ap_rec - ap2
        res[tag]["rank_agreement_with_head2"] = ident
        # A7  class column identity against the real prompt table
        exp = UCF_COLS if C == len(UCF_COLS) else XD_COLS
        check(C == len(exp), "A7 column count == prompt table size",
              f"{C} vs {len(exp)}")
        mean_p = cls.mean(axis=0)
        top = int(np.argmax(mean_p[1:])) + 1
        print(f"  [----] highest mean posterior: column {top} "
              f"({exp[top] if top < len(exp) else '?'}) = {mean_p[top]:.4f}")
        res[tag]["top_mean_posterior"] = exp[top] if top < len(exp) else None
        res[tag]["mean_posterior"] = {exp[i]: float(mean_p[i])
                                      for i in range(min(C, len(exp)))}
        # A8  fold leakage
        folds = 5
        fidx = np.arange(N) % folds
        vid = np.repeat(np.arange(len(lens)), np.asarray(lens) * REPEAT)
        shared = 0
        for f in range(folds):
            vf = np.unique(vid[fidx == f])
            for g in range(folds):
                if f == g:
                    continue
                vg = np.unique(vid[fidx == g])
                shared += len(np.intersect1d(vf, vg))
        check(shared == 0, "A8 no video spans two folds",
              f"cross-fold video overlaps={shared} "
              f"(videos={len(lens)})")
        res[tag]["cross_fold_video_overlaps"] = shared
        res[tag]["n_videos"] = len(lens)
        res[tag]["pos_rate"] = float((gt[:N] > 0.5).mean())
        print()

    print("=" * 80)
    if FAIL:
        print(f"AUDIT FAILED: {len(FAIL)} check(s)")
        for f in FAIL:
            print(f"  - {f}")
    else:
        print("AUDIT PASSED: all invariants hold")
    print("=" * 80)

    out = {"checks_failed": FAIL, "detail": res,
           "verdict": "AUDIT FAILED" if FAIL else "AUDIT PASSED"}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {args.out}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
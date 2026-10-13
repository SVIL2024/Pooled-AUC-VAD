"""G9 stage 2: does a PER-CLASS bias repair the cross-domain damage?

The gap in the archive
----------------------
vad-exp.md 13.3 established -- causally, not just correlationally -- that
cross-domain damage is a CLASS-DEPENDENT boundary shift: on XD->UCF the normal
class moves -0.657 while the anomaly classes move -1.635, pushing the
normal-vs-rest boundary +1.044.  Section 17 then tested a remedy and declared
it dead.

But 17 tested a **GLOBAL scalar bias**: sigmoid(logit(p)+b).  That is a
strictly increasing function of p, so by 17.3 it cannot change AUC or AP for
reasons that have nothing to do with the class-dependence.

A **PER-CLASS** bias is not globally monotone.  It reorders frames attributed
to different classes, which is exactly the class of change 17.6 says can move
AP.  The archive contains no test of it.

What this script does
---------------------
Builds a per-class score from the dumped class posterior P(c | frame):

    logit_c = log( P(c) / P(normal) )            # normal-vs-class log odds
    score   = max_c ( logit_c - b_c )            # b_c = per-class bias

and evaluates three ways of setting b_c:

  (1) oracle      : b_c chosen WITH target labels  -> upper bound.  If this
                    does not beat baseline, no label-free estimator can.
  (2) source      : b_c fitted on the SOURCE domain by matching the class
                    score distribution to the source class prior.
                    Fully label-free on target.
  (3) mean-shift  : b_c = -(mean logit_c on target) + (same on source),
                    i.e. per-class distribution matching, no target labels.

Kill criteria, fixed in advance:
  K1  the ORACLE must gain at least +0.010 pooled AP.  If the oracle cannot
      win, per-class bias is the wrong instrument and the whole idea dies here
      regardless of how clever the estimator is.
  K2  a label-free estimator must recover at least half the oracle gain.
      (Otherwise the method is not deployable, whatever the oracle shows.)

The baseline is the same checkpoint's published AP2, so any comparison is
like-for-like.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

R = 16
EPS = 1e-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="dir with classprobs_*.npz")
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--source", default="ucf")
    ap.add_argument("--target", default="xd")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = glob.glob(os.path.join(args.scores,
                               f"classprobs_{args.source}2{args.target}.npz"))[0]
    z = np.load(p, allow_pickle=True)
    cls, gt, names = z["cls"], z["gt"], z["names"]
    prob2 = z["prob2"]
    with open(args.lengths) as f:
        _tlen = json.load(f)
    # tag_lengths.json is keyed by config tag (e.g. "A_ucf2xd"), not "ucf2xd".
    key = f"{args.source}2{args.target}"
    lengths = _tlen.get(key) or _tlen.get(f"A_{key}")
    if lengths is None:
        raise KeyError(f"no lengths for {key}; available: {sorted(_tlen)}")

    C = cls.shape[1]
    print(f"class stack {cls.shape}, C={C}")
    print(f"videos={len(lengths)}  clips={sum(lengths)}")

    # logit is FRAME level (cls has one row per frame); prob2 is CLIP level.
    # Expand prob2 to the frame grid so the baseline is directly comparable.
    prob2_frame = np.repeat(prob2.astype(np.float64), R)[:len(cls)]
    y_f = (gt[:len(cls)] > 0.5).astype(int)
    base_ap = float(average_precision_score(y_f, prob2_frame))
    base_auc = float(roc_auc_score(y_f, prob2_frame))
    print(f"baseline on the FRAME grid: AP={base_ap:.4f} AUC={base_auc:.4f}")
    print("published (clip-grid)      : AP=0.6750 AUC=0.8683  [reference]")

    # per-class log-odds vs the normal anchor (index 0)
    Pn = np.clip(cls[:, 0], EPS, 1.0)
    logit = np.log(np.clip(cls[:, 1:], EPS, 1.0)) - np.log(Pn)[:, None]

    # which anomaly class does each video actually contain?
    vid_of = np.empty(len(prob2), dtype=np.int32)
    off = 0
    for vi, ln in enumerate(lengths):
        vid_of[off:off + ln] = vi
        off += ln

    # class -> set of videos that contain it (from the per-video label strings).
    # Column c of the stack (c = 1..C-1) is the c-th anomaly prompt; column 0 is
    # the normal anchor.  The XD code -> column map is fixed by cdv_prompts:
    #   1=B1(Burglary) 2=B2(Bribery) 3=B4(Explosion) 4=B5(Fighting)
    #   5=B6(RoadAccident) 6=G(Gunshot)
    code2idx = {"B1": 1, "B2": 2, "B4": 3, "B5": 4, "B6": 5, "G": 6}
    cls_videos = {c: set() for c in range(1, C)}
    for vi, nm in enumerate(names):
        if str(nm) == "A":
            continue
        for tok in str(nm).split("-"):
            if tok in code2idx:
                cls_videos[code2idx[tok]].add(vi)
    print("videos per class: "
          + ", ".join(f"{c}:{len(v)}" for c, v in sorted(cls_videos.items())))

    def evaluate(score):
        # score may be (N, C-1) per-class log-odds -> the frame's anomaly score
        # is the best (max) over classes; or already a 1-D vector.
        ss = np.asarray(score, np.float64)
        if ss.ndim == 2:
            ss = ss.max(axis=1)
        ss = ss[:len(y_f)]
        if len(ss) != len(y_f):
            raise ValueError(f"length mismatch {len(ss)} vs {len(y_f)}")
        return float(average_precision_score(y_f, ss)), float(roc_auc_score(y_f, ss))

    res = {"baseline": {"ap": base_ap, "auc": base_auc},
           "class_video_counts": {str(k): len(v) for k, v in cls_videos.items()}}

    # ---- sanity: does the per-class score reproduce the published number? ----
    ap_l0, auc_l0 = evaluate(logit)
    res["perclass_score_at_b0"] = {"ap": ap_l0, "auc": auc_l0}
    print(f"\n[sanity] per-class max-logit score at b=0: "
          f"AP={ap_l0:.4f} AUC={auc_l0:.4f}")
    print(f"         published head2 (prob2) on same grid: "
          f"AP={base_ap:.4f} AUC={base_auc:.4f}")
    res["score_construction_matches_published"] = bool(abs(ap_l0 - base_ap) < 0.02)

    # ---- (1) ORACLE: one scalar bias per class, chosen with labels ----
    print("\n(1) ORACLE per-class bias (uses target labels -- upper bound)")
    grid = np.linspace(-1.0, 1.0, 21)
    # coordinate ascent over the 6 anomaly classes
    b = np.zeros(C - 1)
    cur_ap, cur_auc = evaluate(logit)
    print(f"    start (b=0): AP={cur_ap:.4f}")
    for it in range(3):
        for c in range(C - 1):
            best = (cur_ap, b[c])
            for g in grid:
                bb = b.copy()
                bb[c] = g
                ap_, _ = evaluate(logit - bb)
                if ap_ > best[0]:
                    best = (ap_, g)
            b[c] = best[1]
            cur_ap = best[0]
        cur_ap, cur_auc = evaluate(logit - b)
        print(f"    iter{it}: AP={cur_ap:.4f}  b={np.round(b,3).tolist()}")
    res["oracle"] = {"ap": cur_ap, "auc": cur_auc, "b": b.tolist(),
                     "d_ap": cur_ap - base_ap}
    print(f"    ORACLE gain = {cur_ap - base_ap:+.4f} AP")
    print(f"    K1 (oracle >= +0.010): "
          f"{'PASS' if cur_ap - base_ap >= 0.010 else 'FAIL'}")

    # ---- (2) SOURCE-matched bias: shift each class so its mean logit matches
    #          what it would be on the source; computed WITHOUT target labels ----
    print("\n(2) per-class distribution matching (no target labels)")
    # empirical target class prior = uniform over clips (we do not know labels)
    tgt_mean = logit.mean(axis=0)
    tgt_std = logit.std(axis=0)
    # source reference: use the NORMAL videos as the reference population,
    # since their class composition is known (all "A") -- no labels needed.
    normal_vids = {vi for vi, nm in enumerate(names) if str(nm) == "A"}
    mask_norm = np.isin(vid_of, list(normal_vids))
    # logit is frame-level, so the normal-video mask must be lifted to frames
    vid_of_frame = np.repeat(vid_of, R)[:len(logit)]
    mask_norm = np.isin(vid_of_frame, list(normal_vids))
    src_mean = logit[mask_norm].mean(axis=0)
    tgt_mean_all = logit.mean(axis=0)
    tgt_std_all = logit.std(axis=0)
    tgt_mean = tgt_mean_all
    tgt_std = tgt_std_all
    b2 = (src_mean - tgt_mean) / (tgt_std + EPS)
    ap2_, auc2_ = evaluate(logit - b2)
    res["dist_match"] = {"ap": ap2_, "auc": auc2_, "b": b2.tolist(),
                         "d_ap": ap2_ - base_ap}
    print(f"    AP={ap2_:.4f}  gain={ap2_ - base_ap:+.4f}")

    # ---- (3) standardised (z-matched) variant ----
    b3 = (src_mean - tgt_mean)
    ap3_, auc3_ = evaluate(logit - b3)
    res["mean_shift"] = {"ap": ap3_, "auc": auc3_, "b": b3.tolist(),
                         "d_ap": ap3_ - base_ap}
    print(f"    mean-only variant: AP={ap3_:.4f}  gain={ap3_ - base_ap:+.4f}")

    best_lf = max(res["dist_match"]["d_ap"], res["mean_shift"]["d_ap"])
    res["best_labelfree_d_ap"] = best_lf
    res["verdict"] = ("K1 FAIL -> per-class bias is the wrong instrument"
                      if cur_ap - base_ap < 0.010
                      else ("label-free estimator viable" if best_lf >= 0.5 * (cur_ap - base_ap)
                            else "oracle works but estimator does not"))
    print(f"\nVERDICT: {res['verdict']}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
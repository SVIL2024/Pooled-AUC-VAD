"""G15: exploit two G14 findings that make the entry point deployable.

G14 established:
  (1) the held-out gain is a PLATEAU at ~+0.035 -- widening the search from
      +-1 to +-3 to +-10 moves the fitted b on 4 of 6 classes but does NOT
      raise AP.  So we do not need to recover one exact vector; any rule that
      lands on the plateau is worth +0.035.
  (2) the instrument is unbiased (G12: +0.0352 +- 0.0001, 5/5 folds,
      zero selection bias).

This script exploits (1) by looking at the plateau's STRUCTURE, which is what
tells us which label-free rules can reach it.

  A. Random-b landscape : sample many random b, measure AP.  How much of the
     space already scores well?  If almost any b works, the "method" is trivial
     and the honest report is that the gain is a property of the *score*, not
     of b.  If only a structured region works, that structure is the target.
  B. Is the gain just head1?  max_c(logit_c - b_c) reorders classes; compare
     against simply using head1 (the class-agnostic binary head).  If head1
     alone reaches +0.035, the "per-class calibration" framing is wrong.
  C. One-class-at-a-time: which single class's bias carries the gain?  This
     localises the mechanism to specific classes and says whether a rule needs
     all six.
  D. Source-anchored scale (label-free, new dimension): compare each class's
     logit spread on the SOURCE domain (UCF->UCF, already dumped) against the
     TARGET (UCF->XD) and use the ratio as b.  G13 tested target-side moments
     only; this is the one remaining label-free signal.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score

EPS = 1e-12
NAMES = ["fighting", "shooting", "riot", "abuse", "car accident", "explosion"]
B_ORACLE = np.array([-1.0, -0.7, 1.0, 1.0, 0.9, 1.0])


def load(path):
    z = np.load(path, allow_pickle=True)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rand", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = load(glob.glob(os.path.join(args.scores,
                                    "classprobs_ucf2xd.npz"))[0])
    cls, gt, prob1 = z["cls"], z["gt"], z["prob1"]
    logP = np.log(np.clip(cls, EPS, 1.0))
    logit = logP[:, 1:] - logP[:, [0]]
    y = (gt[:len(cls)] > 0.5).astype(int)
    N = len(y)
    fold = np.arange(N) % args.folds
    C = logit.shape[1]

    def ap_of(s, mask=None):
        if mask is None:
            return float(average_precision_score(y, s))
        return float(average_precision_score(y[mask], s[mask]))

    base = float(np.mean([ap_of(logit.max(axis=1), fold == f)
                          for f in range(args.folds)]))
    orac = float(np.mean([ap_of((logit - B_ORACLE).max(axis=1), fold == f)
                          for f in range(args.folds)]))
    print(f"baseline {base:.4f}   oracle {orac:.4f}   gain {orac-base:+.4f}")

    res = {"baseline": base, "oracle": orac, "oracle_d_ap": orac - base}
    res["oracle_b"] = B_ORACLE.tolist()

    # ---- A. random-b landscape ----
    rng = np.random.RandomState(0)
    rnd = []
    for _ in range(args.rand):
        b = rng.uniform(-2, 2, C)
        a = float(np.mean([ap_of((logit - b[None, :]).max(axis=1), fold == f)
                           for f in range(args.folds)]))
        rnd.append(a)
    rnd = np.array(rnd)
    res["random_b"] = {"mean": float(rnd.mean()), "max": float(rnd.max()),
                       "p90": float(np.percentile(rnd, 90)),
                       "frac_above_base": float((rnd > base).mean())}
    print(f"\nA. random b in [-2,2]: mean {rnd.mean():.4f} "
          f"p90 {np.percentile(rnd,90):.4f} max {rnd.max():.4f}")
    print(f"   fraction beating baseline: {(rnd > base).mean()*100:.0f}%")

    # ---- B. is it just head1? ----
    head1 = np.repeat(prob1.astype(np.float64), 16)[:N]
    ap_h1 = float(np.mean([ap_of(head1, fold == f) for f in range(args.folds)]))
    res["head1_ap"] = ap_h1
    res["head1_d_ap"] = ap_h1 - base
    print(f"\nB. head1 (class-agnostic) alone: AP={ap_h1:.4f} "
          f"({ap_h1-base:+.4f})")

    # ---- C. one class at a time ----
    print("\nC. single-class bias at the oracle value:")
    per = {}
    for c in range(C):
        b = np.zeros(C)
        b[c] = B_ORACLE[c]
        a = float(np.mean([ap_of((logit - b[None, :]).max(axis=1), fold == f)
                           for f in range(args.folds)]))
        per[NAMES[c]] = {"d_ap": a - base}
        print(f"     {NAMES[c]:<14} b={B_ORACLE[c]:+.1f}  dAP={a-base:+.4f}")
    res["single_class"] = per

    # ---- D. source-anchored scale ----
    print("\nD. source-anchored per-class scale (label free):")
    src = glob.glob(os.path.join(args.scores, "classprobs_ucf2ucf.npz"))
    if not src:
        print("   [skip] source class posteriors not found")
        res["source_anchored"] = None
    else:
        zs = load(src[0])
        # UCF prompt table (14 cols) vs XD (7): match by NAME.
        ucf_names = ["normal", "Abuse", "Arrest", "Arson", "Assault",
                     "Burglary", "Explosion", "Fighting", "RoadAccidents",
                     "Robbery", "Shooting", "Shoplifting", "Stealing",
                     "Vandalism"]
        match = {"fighting": "Fighting", "shooting": "Shooting",
                 "abuse": "Abuse", "explosion": "Explosion",
                 "car accident": "RoadAccidents"}
        lP_s = np.log(np.clip(zs["cls"], EPS, 1.0))
        logit_s = lP_s[:, 1:] - lP_s[:, [0]]
        sd_s = logit_s.std(axis=0)
        sd_t = logit.std(axis=0)
        out = {}
        for c, nm in enumerate(NAMES):
            if nm not in match:
                out[nm] = {"d_ap": None, "note": "no UCF counterpart"}
                continue
            j = ucf_names.index(match[nm])
            ratio = (sd_s[j] + EPS) / (sd_t[c] + EPS)
            for tag, b in (("log_ratio", -np.log(ratio)),):
                bb = np.zeros(C)
                bb[c] = b
                a = float(np.mean([ap_of((logit - bb[None, :]).max(axis=1),
                                         fold == f) for f in range(args.folds)]))
                out[nm] = {"src_sd": float(sd_s[j]), "tgt_sd": float(sd_t[c]),
                           "b": float(b), "d_ap": a - base}
                print(f"     {nm:<14} sd_src={sd_s[j]:.3f} sd_tgt={sd_t[c]:.3f} "
                      f"b={b:+.3f}  dAP={a-base:+.4f}")
        res["source_anchored"] = out

    best_free = max([v for v in res["single_class"].values()
                     if v.get("d_ap") is not None],
                    key=lambda v: v["d_ap"], default=None)
    print(f"\nbest single-class label-free effect: "
          f"{best_free['d_ap']:+.4f}" if best_free else "n/a")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
"""audit_argmax_semantics: check the statistic the G13/G16 mechanism rests on.

The problem
-----------
Every diagnosis from G13 onward computed the class argmax over the ANOMALY
classes only:

    am = (logP[:, 1:] - logP[:, [0]]).argmax(axis=1)      # 6 columns, no normal

so every frame was assigned to some anomaly class, and G13 reported "the model
fires abuse on 22.92% of frames".  But the mean posterior of the normal class
is 0.4286 -- on a majority of those frames the model may well prefer normal.
The argmax that matters is over ALL columns:

    am_full = cls.argmax(axis=1)                            # includes normal

G16's whole mechanism story ("abuse is over-fired; fighting is the one class the
model's decision stays reliable on") was inferred from the anomaly-only number.
If the two disagree, that story was built on a statistic that does not describe
the model's predictions.

What is measured
  * the prediction distribution with and without the normal column;
  * per-class P(GT=1 | argmax == c) under both definitions;
  * the indicator AUC under both;
  * and, to separate "the class rarely wins" from "the class wins on normal
    footage", a breakdown of each anomaly class's wins by GT.

This is a self-audit of a published-in-project claim, so it is written to be
read as a check on the previous conclusions, not as a new experiment.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    TABLES = {7: ["normal", "fighting", "shooting", "riot", "abuse",
                  "car accident", "explosion"],
              14: ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
                   "Explosion", "Fighting", "RoadAccidents", "Robbery",
                   "Shooting", "Shoplifting", "Stealing", "Vandalism"]}

    out = {}
    for tag in ("ucf2xd", "xd2ucf"):
        hits = glob.glob(os.path.join(args.scores, f"classprobs_{tag}.npz"))
        if not hits:
            continue
        z = np.load(hits[0], allow_pickle=True)
        cls, gt = z["cls"], z["gt"]
        y = (gt[:len(cls)] > 0.5).astype(int)
        C = cls.shape[1]
        names = TABLES[C]
        N = len(y)

        logP = np.log(np.clip(cls, EPS, 1.0))
        logit = logP[:, 1:] - logP[:, [0]]
        am_anom = logit.argmax(axis=1)          # anomaly classes only
        am_full = cls.argmax(axis=1)            # all columns, normal included

        print("=" * 92)
        print(f"{tag}   N={N}   positive rate={y.mean():.4f}")
        print("=" * 92)
        share_a = np.bincount(am_anom, minlength=C - 1) / N
        share_f = np.bincount(am_full, minlength=C) / N

        print(f"\n  {'class':<14}{'anom-only argmax':>18}{'full argmax':>14}"
              f"{'mean P':>10}")
        print("  " + "-" * 56)
        print(f"  {'normal':<14}{'(excluded)':>18}{share_f[0]*100:>13.2f}%"
              f"{cls[:,0].mean():>10.4f}")
        for c in range(1, C):
            print(f"  {names[c]:<14}{share_a[c-1]*100:>17.2f}%"
                  f"{share_f[c]*100:>13.2f}%{cls[:,c].mean():>10.4f}")
        print(f"  {'TOTAL anomaly-only':<14}{share_a.sum()*100:>17.2f}%"
              f"{(share_f[1:].sum())*100:>13.2f}%")

        # per-class reliability under both definitions
        rows = {}
        print(f"\n  {'class':<14}{'prec(anom)':>12}{'prec(full)':>12}"
              f"{'lift(anom)':>12}{'lift(full)':>12}")
        print("  " + "-" * 62)
        base = y.mean()
        for c in range(1, C):
            ma, mf = am_anom == (c - 1), am_full == c
            pa = float(y[ma].mean()) if ma.sum() else float("nan")
            pf = float(y[mf].mean()) if mf.sum() else float("nan")
            rows[names[c]] = {
                "share_anom_only": float(share_a[c - 1]),
                "share_full": float(share_f[c]),
                "prec_anom_only": pa, "prec_full": pf,
                "lift_anom_only": pa / base, "lift_full": pf / base,
            }
            print(f"  {names[c]:<14}{pa:>12.4f}{pf:>12.4f}"
                  f"{pa/base:>12.2f}{pf/base:>12.2f}")

        # where do each anomaly class's full-argmax wins actually sit?
        print("\n  breakdown of FULL-argmax wins by GT "
              "(share of that class's wins that are true anomalies):")
        for c in range(1, C):
            mf = am_full == c
            if not mf.sum():
                continue
            print(f"    {names[c]:<14} wins={int(mf.sum()):>8}  "
                  f"P(y=1)={y[mf].mean():.4f}  "
                  f"P(normal wins)={share_f[0]:.4f}")

        # indicator AUC under both definitions
        auc_a = {}
        auc_f = {}
        for c in range(1, C):
            ia = (am_anom == (c - 1)).astype(np.float32)
            iff = (am_full == c).astype(np.float32)
            auc_a[names[c]] = (0.5 if ia.min() == ia.max()
                               else float(roc_auc_score(y, ia)))
            auc_f[names[c]] = (0.5 if iff.min() == iff.max()
                               else float(roc_auc_score(y, iff)))
        print(f"\n  {'class':<14}{'ind.AUC(anom-only)':>22}"
              f"{'ind.AUC(full)':>18}")
        print("  " + "-" * 52)
        for c in range(1, C):
            print(f"  {names[c]:<14}{auc_a[names[c]]:>22.4f}"
                  f"{auc_f[names[c]]:>18.4f}")
        ba = max(auc_a, key=auc_a.get)
        bf = max(auc_f, key=auc_f.get)
        print("\n  best class by indicator AUC: "
              f"anom-only -> {ba}   full -> {bf}   "
              f"{'SAME' if ba == bf else 'DIFFERENT'}")

        out[tag] = {"positive_rate": float(base),
                    "normal_share_full_argmax": float(share_f[0]),
                    "per_class": rows,
                    "ind_auc_anom_only": auc_a, "ind_auc_full": auc_f,
                    "best_anom_only": ba, "best_full": bf,
                    "best_agrees": ba == bf}

    if len(out) == 2:
        same = all(v["best_agrees"] for v in out.values())
        print("\n" + "=" * 92)
        print("  G16's selection under the FULL argmax: "
              + ", ".join(f"{t} -> {v['best_full']}" for t, v in out.items()))
        print(f"  agrees with the anomaly-only version: {'YES' if same else 'NO'}")
        out["_best_full_agrees"] = same

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
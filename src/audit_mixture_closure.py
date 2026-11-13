"""Test whether pooled area is a closed-form mixture of the two conditional areas.

The claim
---------
Pooled area is a pair-averaged win rate over every positive-negative pair. Split
those pairs by whether they fall inside one video or across two, and the pooled
number is the weighted average of the within-video and cross-video areas, with
the weight given by the share of pairs that are within-video:

    pooled = w * AUC_within + (1 - w) * AUC_cross

Every quantity on the right is already computed by the reference implementation
(s2_table.py: w_posneg, auc_within, auc_cross). This script tests the identity
rather than assuming it, and reports the residual per configuration.

Why it matters
--------------
w_posneg is a function of the clip-label geometry alone. It needs no scores and
no model, so a reader can compute it from their own ground truth before
evaluating anything. If the identity holds, the reported number's composition is
known in advance, and two papers reporting pooled areas on benchmarks with
different w are not reporting comparable quantities.

What this does not do
---------------------
It does not explain why AUC_within takes the values it does. That is a property
of the scorer, and the training objective bears on it; see the accompanying note
on the top-fraction objective. This script only tests the algebra.

Reads s2_table.json. Writes the residuals as an archived record.
"""

from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os

RES = os.path.join(REPO_ROOT, "results")
SRC = os.path.join(RES, "s2_table.json")
OUT = os.path.join(RES, "s2_mixture_closure.json")

# The six single-checkpoint configurations the paper's claims rest on. The three
# ensembles are excluded because they are not independent models.
SINGLE = ("A_ucf2xd", "B_xd2ucf", "C_xd2xd", "D_ucf2ucf", "ucf2sht", "xd2sht")


def main() -> int:
    with open(SRC, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]

    print("=== pooled area as a mixture of the two conditional areas ===")
    print("  identity: pooled = w*AUC_within + (1-w)*AUC_cross")
    print("  w = w_posneg, a function of clip-label geometry only\n")
    print(f"  {'config':<12}{'w_posneg':>11}{'AUC_within':>12}{'AUC_cross':>11}"
          f"{'predicted':>11}{'measured':>10}{'|diff|':>9}")
    print("  " + "-" * 76)

    report = {
        "identity": "pooled_auc = w_posneg * auc_within + (1 - w_posneg) * auc_cross",
        "w_source": ("s2_table.py: w_posneg = sum(ps*ns)/(P*Q), the share of "
                     "positive-negative pairs lying inside one video; computed "
                     "from clip-label geometry alone, no scores involved"),
        "configurations": {},
    }
    worst = 0.0
    for tag in SINGLE:
        r = rows.get(tag)
        if r is None:
            print(f"  {tag:<12}  absent from the archive")
            continue
        w = float(r["w_posneg"])
        aw = float(r["auc_within"])
        ac = float(r["auc_cross"])
        pooled = float(r["pooled_auc"])
        pred = w * aw + (1.0 - w) * ac
        diff = abs(pred - pooled)
        worst = max(worst, diff)
        print(f"  {tag:<12}{w:>11.6f}{aw:>12.4f}{ac:>11.4f}"
              f"{pred:>11.4f}{pooled:>10.4f}{diff:>9.4f}")
        report["configurations"][tag] = {
            "w_posneg": w, "auc_within": aw, "auc_cross": ac,
            "pooled_auc": pooled, "predicted": pred, "abs_diff": diff,
            "placebo_within": float(r.get("placebo_within", float("nan"))),
            "placebo_cross": float(r.get("placebo_cross", float("nan"))),
        }

    print(f"\n  largest absolute residual: {worst:.6f}")
    report["max_abs_diff"] = worst
    report["identity_holds_to"] = float(worst)

    print("\n=== what the weight varies across ===")
    ws = [v["w_posneg"] for v in report["configurations"].values()]
    if ws:
        lo, hi = min(ws), max(ws)
        print(f"  w_posneg range: {lo:.6f} to {hi:.6f}   ratio {hi / lo:.1f}x")
        print("  per benchmark:")
        for tag, v in report["configurations"].items():
            tgt = "ShanghaiTech" if tag.endswith("sht") else (
                "UCF-Crime target" if "ucf" in tag else "XD-Violence target")
            print(f"    {tag:<12} w={v['w_posneg']:.6f}  "
                  f"({v['w_posneg'] * 100:.3f}% of pairs)   {tgt}")
        report["w_range"] = {"min": lo, "max": hi, "ratio": hi / lo}

    print("\n=== is AUC_within informative? ===")
    for tag, v in report["configurations"].items():
        p = v["placebo_within"]
        verdict = ("at chance" if abs(p - 0.5) < 0.02 else
                   "above chance" if p > 0.5 else "below chance")
        rel = "above" if v["auc_within"] > p else "below"
        print(f"  {tag:<12} within={v['auc_within']:.4f}  "
              f"placebo={p:.4f} ({verdict})  observed is {rel} the placebo")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

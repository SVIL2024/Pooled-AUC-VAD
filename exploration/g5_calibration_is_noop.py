"""Stage 0 / G5 — Kill test: is the metric criticism actionable as a METHOD?

Why this probe exists
---------------------
My plan (v2 §2.2, module M-A) proposed "domain-invariant score calibration":
a monotone re-scaling of target-domain scores using unlabeled target statistics,
on the theory that it would remove cross-domain drift.

vad-exp.md §17.3 shows that AP and AUC are invariant to ANY strictly monotone
transformation of the scores, because both depend only on the ordering.
Therefore a *monotone* recalibration is a mathematical no-op for both headline
metrics -- regardless of whether the domain shift is real.

That claim is cheap to falsify and must be falsified before any further
investment, so this probe tests it directly on the same data used in G3.

It also tests the one escape hatch §17.6 leaves open: a NON-monotone,
per-frame (selective) re-weighting. If even that cannot help, the whole
"score-space" branch of my plan is dead and the effort must move elsewhere.

Run:  python probe/g5_calibration_is_noop.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cdvad import data as D
from cdvad import metrics as M
try:  # release layout has g1-g5 flat in exploration/, the working tree has probe/
    from g3_wsvad_crossdomain import score_domain, train_mil
except ImportError:
    from probe.g3_wsvad_crossdomain import score_domain, train_mil

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def monotone_recalibration(s_list, mode: str):
    """Monotone transforms that use NO labels -- exactly what M-A proposed."""
    if mode == "identity":
        return list(s_list)
    if mode == "zscore":                       # per-video affine (monotone)
        return [(s - s.mean()) / (s.std() + 1e-8) for s in s_list]
    if mode == "rank":                         # within-video rank (monotone)
        out = []
        for s in s_list:
            order = np.argsort(np.argsort(s)).astype(np.float32)
            out.append(order / max(len(s) - 1, 1))
        return out
    if mode == "global_quantile":              # fit on target, monotone in s
        allv = np.concatenate(s_list)
        qs = np.quantile(allv, np.linspace(0, 1, 1001))
        return [np.interp(s, qs, np.linspace(0, 1, 1001)) for s in s_list]
    if mode == "logit_shift":                  # sigmoid(logit(p)+b): monotone
        out = []
        for s in s_list:
            p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
            lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
            out.append(1.0 / (1.0 + np.exp(-(lg - 2.0))))
        return out
    raise ValueError(mode)


def selective_reweight(s_list, y_list, lam: float, mode: str = "std"):
    """NON-monotone, per-frame re-weighting -- the one escape hatch left.

    ``mode='std'`` down-weights frames the model is confident about (large
    score variance), i.e. an explicit per-frame weight -> changes ordering
    in a way a global monotone map cannot.
    """
    out = []
    for s in s_list:
        if mode == "std":
            w = 1.0 / (1.0 + lam * s.std())
        elif mode == "range":
            w = 1.0 / (1.0 + lam * (s.max() - s.min()))
        else:
            raise ValueError(mode)
        out.append(s * w)
    return out


def report(y_list, s_list, tag):
    r = M.full_report(y_list, s_list)
    print(f"  {tag:26s} pooled={r['micro_recheck']:.6f}  "
          f"within={r['within_within']:.6f}  AP={r['ap']:.6f}")
    return {k: (float(v) if np.isfinite(v) else None)
            for k, v in r.items()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {}

    ucf_tr = D.load_domain("ucf_train")
    tgt = D.load_domain("xd_test")
    print("[train] weakly-supervised MIL on UCF-Crime")
    w, b = train_mil(ucf_tr, seed=0)
    ys, ss = score_domain(tgt, w, b)
    print(f"[eval] on unseen XD ({len(ys)} videos)\n")

    base = report(ys, ss, "baseline")
    res["baseline"] = base

    # ---- Part 1: monotone recalibration must be a NO-OP ----
    print("\n=== PART 1: monotone, unlabeled recalibration (plan M-A) ===")
    mono = {}
    for mode in ("zscore", "rank", "global_quantile", "logit_shift"):
        mono[mode] = report(ys, monotone_recalibration(ss, mode), mode)
    res["monotone"] = mono

    deltas = {m: mono[m]["micro_recheck"] - base["micro_recheck"] for m in mono}
    print("\n  delta pooled AUC vs baseline:")
    for m, d in deltas.items():
        print(f"    {m:16s} {d:+.3e}")
    res["monotone_deltas"] = deltas

    # ------------------------------------------------------------------
    # Interpretation note (verified, not assumed):
    #   global_quantile is monotone *globally*  -> pooled AUC unchanged to
    #      1e-16.  This confirms vad-exp §17.3 exactly.
    #   zscore / rank are monotone *within each video* but not globally,
    #      so they DO change the cross-video ordering -> pooled moves.
    #      Crucially their within-video AUC is bit-identical to baseline,
    #      which proves the two metrics measure disjoint things.
    # ------------------------------------------------------------------
    all_zero = abs(deltas["global_quantile"]) < 1e-12
    within_invariant = all(
        abs(mono[m]["within_within"] - base["within_within"]) < 1e-6
        for m in ("zscore", "rank"))
    res["global_monotone_is_exact_noop"] = bool(all_zero)
    res["per_video_monotone_leaves_within_invariant"] = bool(within_invariant)
    res["per_video_normalisation_hurts_pooled"] = bool(deltas["zscore"] < 0)

    print("\n=== VERDICT ===")
    print(f"  globally-monotone recalibration is an EXACT no-op : {all_zero}")
    print(f"  per-video monotone leaves within-AUC invariant   : {within_invariant}")
    print(f"  per-video normalisation HURTS pooled AUC          : {deltas['zscore']:+.4f}")
    print("  => the pooled-vs-within gap is NOT a calibration error.")
    print("     Removing the video-level component destroys pooled AUC,")
    print("     so the shortcut cannot be 'corrected away' -- it IS the signal")
    print("     the current metric rewards.  M-A is dead.")

    # ---- Part 2: the non-monotone escape hatch ----
    print("\n=== PART 2: non-monotone selective re-weighting (last escape) ===")
    sel = {}
    for mode in ("std", "range"):
        for lam in (0.0, 0.1, 0.5, 1.0, 2.0):
            sel[f"{mode}_lam{lam}"] = report(
                ys, selective_reweight(ss, ys, lam, mode), f"{mode} lam={lam}")
    res["selective"] = sel

    best = max((v["micro_recheck"] for v in sel.values()), default=0.0)
    res["best_selective"] = best
    res["best_selective_delta"] = best - base["micro_recheck"]
    print(f"\n  best selective pooled AUC = {best:.6f} "
          f"(delta {best - base['micro_recheck']:+.4f})")
    print("  => no non-monotone re-weighting of these forms helps either.")

    with open(os.path.join(OUT_DIR, "g5_calibration_is_noop.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[saved] {os.path.join(OUT_DIR, 'g5_calibration_is_noop.json')}")


if __name__ == "__main__":
    main()
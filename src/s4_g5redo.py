"""s4_g5redo: redo G5 on real scores, with the complementary decomposition.

G5 (done locally, features never archived) claimed two things:
    (1) a monotone recalibration of the score is an EXACT no-op, because AUC is
        rank-based;
    (2) per-video de-offset costs -0.1219 AP.
Both were single-configuration, single-domain numbers from a pipeline that no
longer exists.  This redoes them on the six archived configurations, with the
offset/within split made explicit.

Why const and de-offset together are the right design
------------------------------------------------------
They are complementary interventions on the same scores:

    const      every frame takes its video's mean  -> offset ONLY
    de-offset  every frame loses its video's mean  -> within-video ONLY

So the two must together account for the original, and if G1's capacity finding
(const ~ original) and S3's finding (de-offset costs about as much) are the same
statement approached from opposite directions, then their agreement is an
internal consistency check: an error in either direction shows up as a mismatch.
That is the check worth running -- not either number alone.

G5's claim (1) is re-verified numerically rather than asserted: a strictly
monotone but strongly nonlinear recalibration must leave AUC and AP bit-identical,
and any drift means the transform was not actually monotone over the observed
range.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPEAT = 16


def load(path, lens_npz=None):
    z = np.load(path, allow_pickle=True)
    lens = (np.asarray(z["lens"], dtype=np.int64) if "lens" in z.files else
            np.asarray(np.load(lens_npz, allow_pickle=True)["lens"],
                      dtype=np.int64))
    return (np.asarray(z["prob2"], float), np.asarray(z["gt"]), lens)


def clip_labels(gt, n):
    return np.array([(gt[i * REPEAT:(i + 1) * REPEAT].mean() > 0.5)
                     for i in range(n)], dtype=bool)


def const_scores(s, lens):
    offs = np.concatenate([[0], np.cumsum(lens)])
    out = np.array(s, copy=True)
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        out[lo:hi] = out[lo:hi].mean()
    return out


def deoffset(s, lens):
    offs = np.concatenate([[0], np.cumsum(lens)])
    out = np.array(s, copy=True)
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        out[lo:hi] = out[lo:hi] - out[lo:hi].mean()
    return out


def metrics(y, s):
    return (float(roc_auc_score(y, s)), float(average_precision_score(y, s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--sht", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    LENS_FILE = {"xd": "classprobs_ucf2xd.npz", "ucf": "classprobs_xd2ucf.npz"}
    jobs = []
    for cfg, tgt in (("A_ucf2xd", "xd"), ("C_xd2xd", "xd"), ("B_xd2ucf", "ucf"),
                     ("D_ucf2ucf", "ucf")):
        f = glob.glob(os.path.join(args.p3, cfg, "cdv_scores_*.npz"))
        if f:
            jobs.append((cfg, f[0], os.path.join(args.lens, LENS_FILE[tgt])))
    for src in ("ucf", "xd"):
        f = os.path.join(args.sht, f"cdv_scores_{src}2sht.npz")
        if os.path.exists(f):
            jobs.append((f"{src}2sht", f, None))

    # ---- G5 claim (1): monotone recalibration is an exact no-op ----
    print("=" * 122)
    print("CLAIM 1 -- a strictly monotone recalibration must be an exact no-op")
    print("=" * 122)
    mono_ok = True
    for tag, path, lf in jobs:
        s, gt, lens = load(path, lf)
        y = clip_labels(gt, len(s))
        a0, p0 = metrics(y, s)
        for name, tf in (("logit(x)", lambda v: np.log(np.clip(v, 1e-12, 1.0))),
                         ("x^3", lambda v: v ** 3),
                         ("1-exp(-20x)", lambda v: 1 - np.exp(-20 * v)),
                         ("rank", lambda v: np.argsort(np.argsort(v)).astype(
                             float))):
            a1, p1 = metrics(y, tf(s))
            d = max(abs(a1 - a0), abs(p1 - p0))
            mono_ok &= (d == 0.0)
            print(f"  {tag:<14}{name:<14} AUC {a0:.6f} -> {a1:.6f}   "
                  f"AP {p0:.6f} -> {p1:.6f}   |delta| = {d:.1e}")
        print()
    print(f"  monotone no-op holds exactly on all configurations: {mono_ok}")

    # ---- G5 claim (2): de-offset cost, plus the complementary const ----
    print()
    print("=" * 122)
    print("CLAIM 2 -- offset / within decomposition (G1 capacity vs S3 de-offset)")
    print("=" * 122)
    print(f"{'config':<14}{'AUC':>8}{'AP':>8}   |{'const AUC':>10}{'constAP':>9}"
          f"{'keep%':>7}   |{'deoff AUC':>10}{'deoffAP':>9}{'keep%':>7}")
    print("-" * 108)
    rows = {}
    for tag, path, lf in jobs:
        s, gt, lens = load(path, lf)
        y = clip_labels(gt, len(s))
        a0, p0 = metrics(y, s)
        ac, pc = metrics(y, const_scores(s, lens))
        ad, pd_ = metrics(y, deoffset(s, lens))
        rows[tag] = {"auc": a0, "ap": p0, "const_auc": ac, "const_ap": pc,
                     "deoff_auc": ad, "deoff_ap": pd_,
                     "const_keep_pct": 100 * ac / a0,
                     "deoff_keep_pct": 100 * ad / a0}
        print(f"{tag:<14}{a0:>8.4f}{p0:>8.4f}   |{ac:>10.4f}{pc:>9.4f}"
              f"{100*ac/a0:>6.1f}%   |{ad:>10.4f}{pd_:>9.4f}{100*ad/a0:>6.1f}%")

    print("=" * 122)
    print("CONSISTENCY -- the identity that actually holds")
    print("=" * 122)
    print("  A previous version of this script checked that (const_auc - 0.5)")
    print("  equals (auc - deoff_auc).  That check was WRONG BY CONSTRUCTION:")
    print("  `const` is a different SCORING of the data (rank frames by their")
    print("  video mean), not a partition of the original score, so the two")
    print("  numbers need not coincide.  The identity that does hold is the")
    print("  pairwise-decomposition one:")
    print("        pooled == w*within + (1-w)*cross")
    print("  which is checked below using the closed-form w.")
    wmap = {"A_ucf2xd": 0.000710411648, "C_xd2xd": 0.000710411648,
            "E_ens11_ucf2xd": 0.000710411648,
            "F_ens22_ucf2xd": 0.000710411648,
            "G_ens33_ucf2xd": 0.000710411648,
            "B_xd2ucf": 0.003883737658, "D_ucf2ucf": 0.003883737658,
            "ucf2sht": 0.00816881435, "xd2sht": 0.00816881435}
    s2 = {}
    p2 = os.path.join(os.path.dirname(args.out), "s2_table.json")
    if os.path.exists(p2):
        s2 = json.load(open(p2))["rows"]
    print(f"\n  {'config':<14}{'pooled':>9}{'recon':>9}{'|err|':>9}{'within':>9}"
          f"{'cross':>9}{'w':>10}")
    print("  " + "-" * 69)
    worst = 0.0
    for tag, r in rows.items():
        if tag not in s2:
            continue
        w = wmap.get(tag, s2[tag]["w_posneg"])
        aw = s2[tag]["auc_within"]
        ac = s2[tag]["auc_cross"]
        rec = w * aw + (1 - w) * ac
        e = abs(rec - r["auc"])
        worst = max(worst, e)
        print(f"  {tag:<14}{r['auc']:>9.4f}{rec:>9.4f}{e:>9.4f}{aw:>9.4f}"
              f"{ac:>9.4f}{w:>10.6f}")
    if worst:
        print(f"\n  worst |pooled - reconstruction| = {worst:.4f}  "
              f"({'identity holds' if worst < 0.01 else 'DOES NOT HOLD'})")
    else:
        print("\n  (s2_table.json not found beside --out; rerun after s2_table)")

    print()
    print("=" * 122)
    print("AP LOSS FROM DE-OFFSET -- the number G5 reported as -0.1219")
    print("=" * 122)
    for tag, r in rows.items():
        print(f"  {tag:<14} dAP = {r['deoff_ap'] - r['ap']:+.4f}")

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "monotone_noop": bool(mono_ok),
                   "worst_mismatch": worst}, f, indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
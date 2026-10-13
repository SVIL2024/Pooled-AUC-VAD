"""G18: is selector A robust, and does it generalise to a second domain pair?

G17 confirmed the label-free rule on ONE pair:
    select the class with the highest source-side indicator AUC,
    subtract 2.0 from that class's log-odds on the target.
    UCF->XD : source picked `fighting`, target gained +0.0351 AP.

Two weaknesses make that result unsafe to publish as-is, and both are tested
here.

W1  FRAGILE SELECTION.  The source-side top-1/top-2 gap on UCF is 0.0014
    (fighting 0.5141 vs explosion 0.5127).  If that ordering is inside the
    noise, the rule is a coin flip that happened to land correctly.
    Test: resample the SOURCE frames and count how often top-1 is stable; and
    measure what the target gain would be for EACH candidate, not only the one
    that happened to win.

W2  SINGLE PAIR.  Generality needs an independent pair with the roles swapped:
    select on XD (source) and apply on UCF (target).  This is structurally
    harder -- UCF's table has 13 anomaly classes against XD's 6, and XD's
    `riot` has no UCF counterpart at all.

Class correspondence
--------------------
The two prompt tables are named, not positional: UCF column 8 is `Fighting`,
XD column 2 is `fighting`, and the offset between them is not constant.  So the
correspondence is declared by name in XD_CLASS_TO_UCF and then oriented per
direction -- source-name -> target-name -- rather than assumed to be the same
map in both directions.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12
XD_CLASSES = ["fighting", "shooting", "riot", "abuse", "car accident",
              "explosion"]                                    # 6 anomaly cols
UCF_CLASSES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
               "Fighting", "RoadAccidents", "Robbery", "Shooting",
               "Shoplifting", "Stealing", "Vandalism"]         # 13 anomaly cols
# XD name -> UCF name; None where the concept exists in only one dataset.
XD_TO_UCF = {"fighting": "Fighting", "shooting": "Shooting", "riot": None,
             "abuse": "Abuse", "car accident": "RoadAccidents",
             "explosion": "Explosion"}


def load_domain(path):
    z = np.load(path, allow_pickle=True)
    cls, gt = z["cls"], z["gt"]
    logP = np.log(np.clip(cls, EPS, 1.0))
    logit = logP[:, 1:] - logP[:, [0]]
    y = (gt[:len(cls)] > 0.5).astype(int)
    return logit, y, logit.argmax(axis=1)


def ind_auc(y, am, c):
    ind = (am == c).astype(np.float32)
    if ind.min() == ind.max():
        return 0.5
    return float(roc_auc_score(y, ind))


def check_identity(path, want_cols, want_tokens, tag):
    """Guard against using another domain pair's file by accident.

    G18 was first run with ucf2ucf standing in for the XD->UCF target, which
    silently measured an in-domain UCF prediction instead of the cross-domain
    setting.  Nothing in the arithmetic noticed.  This reads the file's own
    metadata -- column count and the clip-label vocabulary -- and refuses to
    continue if they do not match the pair being claimed.

    The two datasets label clips differently: UCF uses the English class name
    ("Fighting"), XD uses its own multi-label codes ("B1-B2-0", "A"), so the
    expected tokens are dataset-specific and matched as substrings.
    """
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
        raise SystemExit(
            f"[GUARD] {tag}: {os.path.basename(path)} is not the expected "
            f"pair -> {'; '.join(bad)}")
    print(f"[guard ok] {tag}: {os.path.basename(path)} "
          f"cols={ncol} labels={blob[:40]!r}")
    return path


def run_direction(tag, src_file, tgt_file, src_names, tgt_names, pairs,
                  lam, folds, n_boot):
    """pairs maps SOURCE class name -> TARGET class name (already oriented)."""
    print("=" * 84)
    print(f"DIRECTION {tag}:  select on SOURCE -> apply on TARGET")
    print("=" * 84)

    # ---------------- SOURCE side ----------------
    s_logit, s_y, s_am = load_domain(src_file)
    auc = {n: ind_auc(s_y, s_am, c) for c, n in enumerate(src_names)}
    paired = {s: auc[s] for s, t in pairs.items()}
    order = sorted(paired, key=lambda n: -paired[n])
    top1, top2 = order[0], order[1]
    margin = paired[top1] - paired[top2]
    print(f"\nsource-side indicator AUC ({len(src_names)}-class source table, "
          f"name-matched classes only):")
    for n in order:
        print(f"    {n:<14} -> {pairs[n]:<14} {paired[n]:.4f}")
    print(f"    TOP1 {top1}  margin over TOP2 ({top2}) = {margin:+.4f}")

    # ---------------- W1: stability under source resampling ----------------
    print(f"\nW1  stability: top-1 pick frequency over {n_boot} "
          f"half-samples of the SOURCE frames")
    N = len(s_y)
    rng = np.random.RandomState(0)
    wins = {n: 0 for n in order}
    for _ in range(n_boot):
        half = rng.permutation(N)[: N // 2]
        hy, ham = s_y[half], s_am[half]
        sc = {n: ind_auc(hy, ham, c) for c, n in enumerate(src_names)}
        wins[max(pairs, key=lambda n: sc[n])] += 1
    freq = {n: wins[n] / n_boot for n in order}
    for n in order:
        print(f"      {n:<14} {freq[n]*100:>5.1f}%  "
              f"{'#' * int(round(freq[n]*40))}")
    stable = freq[top1]

    # ---------------- TARGET side ----------------
    t_logit, t_y, _ = load_domain(tgt_file)
    C = t_logit.shape[1]
    fold = np.arange(len(t_y)) % folds

    def fold_ap(s):
        return float(np.mean([average_precision_score(t_y[fold == f], s[fold == f])
                              for f in range(folds)]))

    base = fold_ap(t_logit.max(axis=1))

    def gain_for(cls_name, lm):
        b = np.zeros(C)
        b[tgt_names.index(cls_name)] = -lm
        return fold_ap((t_logit - b[None, :]).max(axis=1)) - base

    print(f"\nW1b counterfactual -- TARGET dAP for boosting EACH candidate "
          f"at lambda={lam}:")
    cand = {}
    for s in order:
        g = gain_for(pairs[s], lam)
        cand[pairs[s]] = g
        mark = "   <== source top-1" if s == top1 else ""
        print(f"      {s:<14} -> {pairs[s]:<14} {g:+.4f}{mark}")

    print(f"\n    rule gain (boost target '{pairs[top1]}') by lambda:")
    sweep = {lm: gain_for(pairs[top1], lm)
             for lm in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)}
    for lm, g in sweep.items():
        print(f"      lam={lm:<4} {g:+.4f}")

    alt = gain_for(pairs[top2], lam)
    print(f"\n    if the ordering had flipped and we boosted "
          f"'{pairs[top2]}' instead: {alt:+.4f} "
          f"(loses {cand[pairs[top1]] - alt:+.4f})")

    best_t, best_g = None, -9.0
    for n in tgt_names:
        g = gain_for(n, lam)
        if g > best_g:
            best_t, best_g = n, g
    print(f"    best single class on TARGET (oracle pick, biased) = "
          f"{best_t} {best_g:+.4f}")

    return {"tag": tag, "pairs": pairs, "source_auc": paired, "top1": top1,
            "top2": top2, "margin": margin, "stability": freq,
            "stable_freq": stable, "baseline": base,
            "lambda_sweep": {str(k): v for k, v in sweep.items()},
            "candidate_gain": cand, "rank2_gain": alt,
            "oracle_pick": best_t, "oracle_pick_gain": best_g, "lambda": lam}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def find(src, tgt):
        p = glob.glob(os.path.join(args.scores, f"classprobs_{src}2{tgt}.npz"))
        return p[0] if p else None

    ucf2ucf, ucf2xd = find("ucf", "ucf"), find("ucf", "xd")
    xd2xd, xd2ucf = find("xd", "xd"), find("xd", "ucf")
    UCF_TOK = UCF_CLASSES + ["Normal"]
    XD_TOK = ["B1", "B2", "B4", "B5", "B6", "G"]
    out = {}
    if ucf2ucf and ucf2xd:
        # source table = UCF, target table = XD -> UCF name -> XD name
        pairs = {v: k for k, v in XD_TO_UCF.items() if v is not None}
        check_identity(ucf2ucf, 14, UCF_TOK, "UCF->XD source")
        check_identity(ucf2xd, 7, XD_TOK, "UCF->XD target")
        out["ucf2xd"] = run_direction("UCF->XD", ucf2ucf, ucf2xd, UCF_CLASSES,
                                      XD_CLASSES, pairs, args.lam, args.folds,
                                      args.boot)
    if xd2xd and xd2ucf:
        # source table = XD, target table = UCF -> XD name -> UCF name.
        # The target MUST be the XD checkpoint scored on UCF test; using
        # ucf2ucf here would measure an in-domain UCF prediction.
        pairs = {k: v for k, v in XD_TO_UCF.items() if v is not None}
        check_identity(xd2xd, 7, XD_TOK, "XD->UCF source")
        check_identity(xd2ucf, 14, UCF_TOK, "XD->UCF target")
        out["xd2ucf"] = run_direction("XD->UCF", xd2xd, xd2ucf, XD_CLASSES,
                                      UCF_CLASSES, pairs, args.lam, args.folds,
                                      args.boot)
    elif xd2xd:
        print("[skip] XD->UCF: classprobs_xd2ucf.npz missing -- the target "
              "must come from the XD checkpoint on UCF test, not ucf2ucf")

    print("=" * 84)
    print("SUMMARY")
    print("=" * 84)
    unstable, negative = [], []
    for k, v in out.items():
        r = v["lambda_sweep"].get(str(args.lam), 0.0)
        flag = ""
        if v["stable_freq"] < 0.6:
            flag += "   <== UNSTABLE"
        if r <= 0:
            flag += "   <== NO GAIN"
            negative.append(k)
        print(f"  {k:<8} pick={v['top1']:<12} margin={v['margin']:+.4f} "
              f"stability={v['stable_freq']*100:>5.1f}%  "
              f"dAP(lam={args.lam})={r:+.4f}{flag}")
        if v["stable_freq"] < 0.6:
            unstable.append(k)
    # The verdict must require BOTH a stable pick AND a positive gain in every
    # direction.  G18's first version checked stability only and printed
    # "ROBUST" while a direction was losing AP.
    if negative:
        verdict = (f"DOES NOT GENERALISE -- no gain in {negative} "
                   f"(selection itself is stable)")
    elif unstable:
        verdict = f"FRAGILE -- unstable selection in {unstable}"
    else:
        verdict = "ROBUST -- positive gain and stable selection on every pair"
    out["_verdict"] = verdict
    out["_lambda"] = args.lam
    out["_folds"] = args.folds
    print(f"\noverall: {verdict}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
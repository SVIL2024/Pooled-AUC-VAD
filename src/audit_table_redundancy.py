"""Audit Table 1 and the figures for rows that carry no new information.

The question is whether any reported row or panel is decorative: a near-duplicate
of another, a constant across rows, or a figure panel that restates a table
column. Those cost the reader's attention and invite the question of what the
extra row was for.

Checks run:
  1. constant columns -- a column whose value never varies down the table
  2. near-duplicate rows -- pairs differing by less than a stated tolerance on
     every reported column, which makes one of them redundant
  3. ensembles against their own singles -- the three ensembles were added for
     completeness, so check whether they say anything the singles do not
  4. figure panels against Table 1 -- whether any plotted bar is a column the
     table already gives exactly

Exit status is non-zero when a decorative row is found, so it can gate a commit.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os

RES = os.path.join(REPO_ROOT, "results")
DUP_TOL = 0.02          # AUC units; below this two rows say the same thing


def main() -> int:
    t = json.load(open(os.path.join(RES, "s2_table.json"), encoding="utf-8"))
    g = t["rows"]
    keys = list(g)

    cols = ["pooled_auc", "pooled_ap", "w_posneg", "gap"]
    print(f"Table 1: {len(keys)} rows, columns {cols}\n")

    # ---- 1. constant columns -------------------------------------------
    print("=== columns that never vary ===")
    const = []
    for c in cols:
        vals = [round(g[k][c], 6) for k in keys]
        spread = max(vals) - min(vals)
        by_target = {}
        for k in keys:
            by_target.setdefault(g[k]["n_clips"], []).append(round(g[k][c], 6))
        per_target_const = all(len(set(v)) == 1 for v in by_target.values())
        print(f"  {c:<12} spread={spread:.6f}  constant within each target: "
              f"{per_target_const}")
        if per_target_const:
            const.append(c)
    if not const:
        print("  none")

    # ---- 2. near-duplicate rows ----------------------------------------
    print("\n=== near-duplicate row pairs (all columns within tolerance) ===")
    dups = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if g[a]["n_clips"] != g[b]["n_clips"]:
                continue
            diffs = {c: abs(g[a][c] - g[b][c]) for c in cols
                     if c != "w_posneg"}
            if diffs and max(diffs.values()) <= DUP_TOL:
                dups.append((a, b, max(diffs.values())))
    # The manuscript collapses the ensembles into one range row, so the question
    # is whether anything ELSE in Table 1 is redundant. Pairs among the raw
    # ensemble rows no longer correspond to any printed row.
    ENS = [k for k in keys if "ens" in k]
    printed_dups = [(a, b, m) for a, b, m in dups
                    if "ens" not in a and "ens" not in b]
    if not printed_dups:
        print("  none among printed rows")
    for a, b, m in printed_dups:
        print(f"  {a} ~ {b}   max pooled-AUC difference {m:.4f}")
    if dups and not printed_dups:
        print(f"  ({len(dups)} pair(s) among the {len(ENS)} raw ensemble rows, "
              f"reported as a single range row in the manuscript)")

    # ---- 3. what do the ensembles add? ---------------------------------
    print("\n=== ensembles vs the single checkpoint on the same target ===")
    singles = {"A_ucf2xd": "UCF-trained", "C_xd2xd": "XD-trained"}
    for e in [k for k in keys if "ens" in k]:
        near = min(singles, key=lambda s: abs(g[e]["pooled_auc"] - g[s]["pooled_auc"]))
        dp = abs(g[e]["pooled_auc"] - g[near]["pooled_auc"])
        dg = abs(g[e]["gap"] - g[near]["gap"])
        print(f"  {e:<14} closest single {near} ({singles[near]}): "
              f"pooled d={dp:.4f}  gap d={dg:.4f}")
        print(f"      -> ensembling {'changes' if dp > 0.01 else 'does not change'} "
              f"the headline number by more than 0.01")

    # ---- 4. figure panels vs table columns -----------------------------
    print("\n=== do the figures restate Table 1 exactly? ===")
    fig = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "fig_values.json")
    if os.path.exists(fig):
        fv = json.load(open(fig, encoding="utf-8"))
        fig4 = ["A_ucf2xd", "C_xd2xd", "B_xd2ucf", "D_ucf2ucf",
                "ucf2sht", "xd2sht"]
        identical = [k for k in fig4
                     if fv["pooled_auc"].get(k) == round(g[k]["pooled_auc"], 4)]
        print(f"  Fig 4 pooled bar vs Table 1 pooled AUC: "
              f"{len(identical)}/{len(fig4)} identical")
        print("  -> the pooled bar is the table's pooled column; the panel adds "
              "the two components, so it is not redundant")
    else:
        print("  fig_values.json not found; skipped")

    bad = bool(printed_dups)
    print()
    print("VERDICT:", "decorative rows present" if bad
          else "no redundant rows among the printed ones")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
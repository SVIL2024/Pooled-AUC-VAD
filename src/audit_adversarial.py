"""Adversarial checks: look for results that would undercut the paper's claims.

Consistency auditing asks whether the manuscript agrees with itself. This asks
the opposite and more dangerous question: does any stored result contradict a
central claim? Those are the findings that get a paper rejected rather than
revised, because a reviewer who finds one has grounds to distrust the rest.

Claims under test
  C1 sufficiency      the aligned reconstruction should recover the reported
                      figure, and neither control should
  C2 necessity        de-offsetting should not improve on the reported value,
                      since it would mean the offset was not carrying it
  C3 sign reversal     the ShanghaiTech gap should be negative on both
                      configurations if the paper says both reverse
  C4 constancy        the retained share should be high and comparable on both
                      UCF-Crime and XD-Violence
  C5 scope            the interval groups should NOT separate, because the
                      paper says the mechanism is common to all three splits

Any FAIL here is a line the manuscript does not currently state and must.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os

RES = os.path.join(REPO_ROOT, "results")


def main() -> int:
    s6 = json.load(open(os.path.join(RES, "s6_sufficiency.json"), encoding="utf-8"))
    s2 = json.load(open(os.path.join(RES, "s2_table.json"), encoding="utf-8"))["rows"]
    s4 = json.load(open(os.path.join(RES, "s4_g5redo.json"), encoding="utf-8"))["rows"]
    s5 = json.load(open(os.path.join(RES, "s5_ci.json"), encoding="utf-8"))

    fails = []

    print("=== C1 sufficiency: aligned must land near, controls must not ===")
    for k in ("A_ucf2xd", "C_xd2xd", "B_xd2ucf", "D_ucf2ucf"):
        a = s6[k]["arms"]["1.0"]
        dev = a["aligned"] - s6[k]["orig"]
        rec = abs(dev) < 0.07
        bad = max(a["anti"], a["random"])
        ctrl = bad < a["aligned"] - 0.20
        ok = rec and ctrl
        print(f"  {k:<11} orig={s6[k]['orig']:.4f} aligned={a['aligned']:.4f} "
              f"dev={dev:+.4f} anti={a['anti']:.4f} random={a['random']:.4f}  "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"C1 {k}: aligned deviation {dev:+.4f}, controls lose={ctrl}")

    print("\n=== C2 necessity: de-offset must not beat the reported value ===")
    for k in ("A_ucf2xd", "C_xd2xd", "B_xd2ucf", "D_ucf2ucf"):
        better = s6[k]["deoff"] > s6[k]["orig"]
        print(f"  {k:<11} orig={s6[k]['orig']:.4f} deoff={s6[k]['deoff']:.4f}"
              f"  {'FAIL deoff higher' if better else 'ok'}")
        if better:
            fails.append(f"C2 {k}: de-offset exceeds the reported value")

    print("\n=== C3 sign reversal: both ShanghaiTech gaps negative ===")
    for k in ("ucf2sht", "xd2sht"):
        g = s2[k]["gap"]
        print(f"  {k:<11} gap={g:+.4f}  {'FAIL not negative' if g >= 0 else 'ok'}")
        if g >= 0:
            fails.append(f"C3 {k}: gap is not negative")

    print("\n=== C4 retained share high on both mainstream benchmarks ===")
    for k in ("A_ucf2xd", "C_xd2xd", "B_xd2ucf", "D_ucf2ucf"):
        ck = s4[k]["const_keep_pct"]
        ok = 90.0 <= ck <= 100.0
        print(f"  {k:<11} const_keep={ck:.1f}%  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"C4 {k}: retained share {ck:.1f}% outside 90-100")

    print("\n=== C5 scope: interval groups must NOT separate ===")
    groups = {}
    for k, v in s5["rows"].items():
        groups.setdefault(v.get("group", "?"), []).append(v)
    envs = {}
    for gname, items in groups.items():
        lo = min(v["const_ci"][0] for v in items)
        hi = max(v["const_ci"][1] for v in items)
        envs[gname] = (lo, hi)
        print(f"  {gname:<8} n={len(items)} envelope=[{lo:.1f}, {hi:.1f}]")
    names = list(envs)
    overlap = all(
        not (envs[a][1] < envs[b][0] or envs[b][1] < envs[a][0])
        for i, a in enumerate(names) for b in names[i + 1:])
    print(f"  groups overlap: {overlap}  "
          f"{'ok (consistent with the claim)' if overlap else 'FAIL (groups DO separate)'}")
    if not overlap:
        fails.append("C5: interval groups separate, contradicting the paper")

    print("\n=== C6 SHT retained share above 100 is expected, not a failure ===")
    for k in ("ucf2sht", "xd2sht"):
        ck = s4[k]["const_keep_pct"]
        print(f"  {k:<11} const_keep={ck:.1f}%  "
              f"{'ok (pooled was below chance)' if ck > 100 else 'CHECK'}")

    print()
    if fails:
        print(f"{len(fails)} contradiction(s) with a stated claim:")
        for f in fails:
            print(f"  - {f}")
        print("\nVERDICT: a stored result contradicts a central claim")
        return 1
    print("VERDICT: no stored result contradicts any central claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
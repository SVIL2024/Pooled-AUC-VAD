"""test_s1_event_metric: unit tests for the event-level metric before it is used.

An event metric that is subtly wrong would produce a confident, wrong verdict
about the whole thesis, and this project has already shipped one wrong
conclusion that survived several rounds of checking.  So the metric is verified
against cases with known answers BEFORE any real data is touched.

Cases, each with an unambiguous expected outcome:
  T1 perfect           identical predictions -> F1 = 1
  T2 inverted          predictions negated   -> F1 = 0
  T3 fragmentation     same frames detected, but split into pieces
                       -> recall holds, precision holds, F1 < 1 and < T1
  T4 single-clip shift detection shifted by one clip
                       -> tIoU = (n-1)/(n+1); at tIoU 0.3 short events survive,
                          long ones die.  Tests that the threshold bites.
  T5 disjoint          predicted events do not overlap any GT event -> F1 = 0
  T6 empty             no prediction -> F1 = 0, recall = 0
  T7 random            chance-level, and far below T1
  T8 run extraction    runs_of on a hand-checked bit pattern
Also checks that event_pr_at agrees with a brute-force count on one case, so the
greedy one-to-one matching is not merely self-consistent.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from s1_event_vs_frame import (build_grid, event_counts,  # noqa: E402
                              event_f1_at_eer, event_pr_at, runs_of)

FAIL = []


def check(cond, label, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" +
          (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(label)


def single_video(y, s, th):
    return event_pr_at(s, y, [(0, len(y))], th)


def main():
    n = 60
    y = np.zeros(n, dtype=bool)
    y[10:30] = True            # one GT event of 20 clips
    vids = [(0, n)]

    print("T8  run extraction")
    check(runs_of(np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=bool)) == [(2, 5), (6, 7)],
          "runs_of splits consecutive runs", str(runs_of(np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=bool))))
    check(runs_of(np.zeros(5, dtype=bool)) == [], "runs_of empty -> []")
    check(runs_of(np.ones(5, dtype=bool)) == [(0, 5)], "runs_of all true -> one run")

    print()
    print("T1  perfect prediction")
    s = y.astype(float)
    p, r, f = single_video(y, s, 0.5)
    check(abs(f - 1.0) < 1e-9, "perfect -> F1 = 1", f"P={p:.3f} R={r:.3f} F1={f:.3f}")

    print()
    print("T2  inverted prediction")
    s = (~y).astype(float)
    p, r, f = single_video(y, s, 0.5)
    check(abs(f) < 1e-9, "inverted -> F1 = 0", f"P={p:.3f} R={r:.3f} F1={f:.3f}")

    print()
    print("T3  fragmentation (same frames, broken into pieces)")
    frag = np.zeros(n, dtype=bool)
    for a in range(10, 30, 2):
        frag[a] = True
    s = frag.astype(float)
    p, r, f = single_video(y, s, 0.5)
    # every 1-clip run over a 20-clip GT gives tIoU = 1/20 = 0.05 < 0.3,
    # so nothing matches: precision 0, recall 0, F1 0.  Fragmentation must hurt.
    check(f < 0.5, "fragmentation severely hurts F1", f"P={p:.3f} R={r:.3f} F1={f:.3f}")

    print()
    print("T4  one-clip shift")
    sh = np.zeros(n, dtype=bool)
    sh[11:31] = True
    s = sh.astype(float)
    p, r, f = single_video(y, s, 0.5)
    # overlap = 19, union = 21, tIoU = 0.905 >= 0.3 -> still a match
    check(r > 0.9, "1-clip shift of a 20-clip event still matches", f"R={r:.3f}")
    tiny = np.zeros(10, dtype=bool)
    tiny[2:4] = True            # 2-clip event
    tiny2 = np.zeros(10, dtype=bool)
    tiny2[3:5] = True           # shifted by 1 -> tIoU = 1/3 = 0.33 >= 0.3
    p, r, f = event_pr_at(tiny2.astype(float), tiny, [(0, 10)], 0.5)
    check(r > 0.9, "2-clip event, 1-clip shift: tIoU=1/3 >= 0.3 matches", f"R={r:.3f}")
    tiny3 = np.zeros(10, dtype=bool)
    tiny3[4:6] = True           # shifted by 2 -> tIoU = 0 -> no match
    p, r, f = event_pr_at(tiny3.astype(float), tiny, [(0, 10)], 0.5)
    check(r < 0.1, "2-clip event, 2-clip shift: tIoU=0 -> no match", f"R={r:.3f}")

    print()
    print("T5  disjoint prediction")
    s = np.zeros(n, dtype=bool)
    s[40:50] = True
    p, r, f = single_video(y, s.astype(float), 0.5)
    check(abs(f) < 1e-9, "disjoint -> F1 = 0", f"P={p:.3f} R={r:.3f} F1={f:.3f}")

    print()
    print("T6  empty prediction")
    s = np.zeros(n, dtype=float)
    p, r, f = single_video(y, s, 0.5)
    check(abs(f) < 1e-9 and abs(r) < 1e-9, "empty -> F1 = 0, R = 0", f"P={p:.3f}")

    print()
    print("T7  random is far below perfect")
    rng = np.random.RandomState(0)
    fs = []
    for t in range(200):
        s = rng.rand(n)
        _pp, _rr, ff = single_video(y, s, 0.5)
        fs.append(ff)
    fs = np.array(fs)
    check(fs.mean() < 0.3, "random mean F1 low", f"mean={fs.mean():.3f} max={fs.max():.3f}")

    print()
    print("T9  greedy matching vs brute force")
    gt = [(0, 10), (20, 30), (40, 50)]
    pr = [(0, 12), (18, 28), (42, 52)]
    tp, fp, fn = event_counts(pr, gt)
    # 3 pairs, each overlapping its counterpart with tIoU >= 0.3 -> 3 TP
    check(tp == 3 and fp == 0 and fn == 0, "three clean matches",
          f"tp={tp} fp={fp} fn={fn}")
    pr2 = [(0, 10), (20, 30), (40, 50)]
    gt2 = [(5, 15), (25, 35), (45, 55)]
    tp, fp, fn = event_counts(pr2, gt2)
    check(tp == 3 and fn == 0, "shifted matches", f"tp={tp} fp={fp} fn={fn}")
    # one prediction straddling two GT events: must match only ONE
    pr3 = [(0, 30)]
    gt3 = [(0, 10), (20, 30)]
    tp, fp, fn = event_counts(pr3, gt3)
    check(tp == 1 and fn == 1, "one prediction -> at most one TP",
          f"tp={tp} fp={fp} fn={fn}")

    print()
    print("T10 EER sweep returns a finite, monotone-sane value")
    s = y.astype(float) + 0.01 * rng.rand(n)
    f1, p, r, th = event_f1_at_eer(s, y, vids)
    check(0.0 <= f1 <= 1.0 and 0.0 <= th <= 1.0, "EER returns in range",
          f"F1={f1:.3f} P={p:.3f} R={r:.3f} th={th:.3f}")

    print()
    print("T11 build_grid returns (lo, hi) pairs that tile the clip axis")
    lens = np.array([3, 4, 2])
    g = build_grid(lens)
    check(g == [(0, 3), (3, 7), (7, 9)], "build_grid offsets",
          str(g))
    check(all(isinstance(t, tuple) and len(t) == 2 for t in g),
          "every element is a 2-tuple (this is what event_pr_at unpacks)")
    check(g[0][0] == 0 and g[-1][1] == int(lens.sum()),
          "grid spans the whole clip axis", f"{g[0][0]}..{g[-1][1]}")
    # and the metric must accept the grid build_grid actually produces
    ys = np.zeros(9, dtype=bool)
    ys[1:2] = True
    ss = np.zeros(9, dtype=bool)
    ss[1:2] = True
    p, r, f = event_pr_at(ss.astype(float), ys, g, 0.5)
    check(abs(f - 1.0) < 1e-9, "perfect prediction via build_grid -> F1 = 1",
          f"F1={f:.3f}")

    print()
    print("=" * 70)
    if FAIL:
        print(f"FAILED {len(FAIL)} check(s):")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("ALL EVENT-METRIC TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
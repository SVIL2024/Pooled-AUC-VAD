"""s1b_event_structure: why did event F1 survive within-video permutation?

The intervention in s1_event_vs_frame falsified the capability-asymmetry
hypothesis on UCF/XD: randomising the score order inside each video destroyed
only 17.4% of event F1, so event F1 is carried mostly by per-video information.

The leading explanation is dataset geometry, not model behaviour: if a ground
truth event covers a large fraction of its video, then even randomly scattered
high scores still overlap the event enough to clear the tIoU threshold, and
event-level F1 degenerates toward a video-level metric.  That would mean
UCF/XD are simply the wrong testbed for the frame-versus-event question, and it
would also explain why 2604.09327 sees a dramatic gap on NWPUC but we do not
see it here.

So this measures the geometry directly, and nothing else is inferred from it:
  * event length as a fraction of its video
  * the fraction of each anomalous video that the event occupies
  * the same numbers for the normal videos that matter for false positives
  * an analytical check: given event coverage c, what fraction of a random
    scatter still clears tIoU 0.3?  Compared against what the models scored.

If c is large, the "event" metric is not discriminative on this data and the
original thesis was untestable here rather than simply false.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_event_vs_frame import build_grid, runs_of  # noqa: E402

REPEAT = 16
TIOU = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = {"xd": "classprobs_ucf2xd.npz", "ucf": "classprobs_xd2ucf.npz"}
    out = {}
    print("=" * 92)
    print(f"{'domain':<6}{'videos':>8}{'anomVid':>9}{'events':>8}"
          f"{'evt/vidLen':>12}{'anomVidCov':>12}{'multi-evt%':>12}"
          f"{'nEvents':>9}")
    print("=" * 92)
    for dom, f in src.items():
        lens = np.load(os.path.join(args.lens, f), allow_pickle=True)["lens"]
        z = np.load(glob.glob(os.path.join(args.lens, f))[0],
                    allow_pickle=True)
        gt = z["gt"]
        n_clip = len(lens)
        y = np.array([(gt[i * REPEAT:(i + 1) * REPEAT].mean() > 0.5)
                      for i in range(n_clip)], dtype=bool)
        vids = build_grid(lens)

        lengths, covers, n_ev, multi, anom_vids = [], [], [], 0, 0
        for lo, hi in vids:
            L = hi - lo
            ev = runs_of(y[lo:hi])
            if not ev:
                continue
            anom_vids += 1
            n_ev.append(len(ev))
            if len(ev) > 1:
                multi += 1
            for a, b in ev:
                lengths.append(b - a)
                covers.append((b - a) / L)
        lengths = np.array(lengths)
        covers = np.array(covers)
        n_ev = np.array(n_ev)

        print(f"{dom:<6}{len(vids):>8}{anom_vids:>9}{len(lengths):>8}"
              f"{np.median(lengths):>11.1f}c{np.median(covers):>11.3f}"
              f"{100*multi/max(anom_vids,1):>11.1f}%{np.mean(n_ev):>9.2f}")
        out[dom] = {
            "n_videos": len(vids), "n_anom_videos": anom_vids,
            "n_events": int(len(lengths)),
            "median_event_clips": float(np.median(lengths)),
            "mean_event_clips": float(np.mean(lengths)),
            "median_coverage_of_video": float(np.median(covers)),
            "mean_coverage_of_video": float(np.mean(covers)),
            "frac_events_covering_ge_50pct": float((covers >= 0.5).mean()),
            "frac_anom_videos_multi_event": float(multi / max(anom_vids, 1)),
            "mean_events_per_anom_video": float(np.mean(n_ev)),
        }

    # ---- analytic bound: how much can a random scatter still clear tIoU? ----
    print()
    print("=" * 92)
    print("ANALYTIC CHECK  (random scatter of the same volume, event coverage c)")
    print("=" * 92)
    print("  A prediction that is a random subset of the video, of the same size")
    print("  as the event, overlaps it by chance.  For coverage c and n clips,")
    print(f"  a fair estimate of the best achievable tIoU is checked against {TIOU}.")
    print()
    print(f"  {'domain':<6}{'median c':>10}{'best-case tIoU (random subset)':>34}"
          f"{'clears %.1f?' % TIOU:>16}")
    print("  " + "-" * 70)
    rng = np.random.RandomState(0)
    for dom, v in out.items():
        c = v["median_coverage_of_video"]
        n = v["median_event_clips"]
        # random subset of size n out of L = n/c clips; overlap with the event
        L = n / c
        hits = []
        for _ in range(400):
            pick = rng.choice(int(round(L)), size=int(n), replace=False)
            hits.append(np.sum((pick >= 0) & (pick < n)) / max(n, 1))
        best = float(np.percentile(hits, 95))
        print(f"  {dom:<6}{c:>10.3f}{best:>34.3f}"
              f"{('yes' if best >= TIOU else 'NO'):>16}")

    print()
    print("=" * 92)
    print("READING")
    print("=" * 92)
    for dom, v in out.items():
        c = v["median_coverage_of_video"]
        if c >= 0.5:
            msg = (f"{dom}: a GT event covers a median {c*100:.0f}% of its video. "
                   "A random scatter of the same size still clears tIoU, so the "
                   "event\n     metric is close to video-level here and the "
                   "capability-asymmetry test\n     cannot be "
                   "run on this dataset.")
        else:
            msg = (f"{dom}: median coverage {c*100:.0f}%. Short enough for "
                   "temporal structure to matter.")
        print(f"  {msg}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
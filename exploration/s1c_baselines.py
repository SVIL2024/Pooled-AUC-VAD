"""s1c_baselines: the trivial baselines that give event F1 a meaning.

Why this is needed
------------------
s1 reported event F1 around 0.22-0.25 for XD configs and 0.13-0.15 for UCF
configs, and reported that a WITHIN-VIDEO PERMUTATION only drops F1 to ~0.19.
On its own that looks like a bug, because the unit tests show a scattered
prediction scores F1 = 0.

The number is uninterpretable without three reference points:
  FLOOR   random scores                        -- what "no information" scores
  CONST   the video's own mean score            -- the PURE VIDEO-LEVEL scorer,
                                                    i.e. zero within-video
                                                    resolution, the thing the
                                                    permutation is supposed to
                                                    reduce the model to
  ALL     predict every video as one event      -- the trivially-over-predicting
                                                    baseline, which is what a
                                                    broken threshold sweep
                                                    degenerates to
  MODEL   the real scores

If CONST is already close to MODEL, the models are not localising and the
"permutation only costs 17%" reading is exactly right -- the metric simply
cannot separate those two cases. If CONST is near FLOOR while MODEL is far
above, the permutation result IS a bug.

Also recomputes the event geometry, fixing the bug in s1b: the clip axis length
is sum(lens), NOT len(lens).  s1b used len(lens) and therefore saw events in
only 7 of 800 XD videos, so every geometry number it printed was wrong.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_event_vs_frame import (build_grid, event_f1_at_eer,  # noqa: E402
                              runs_of)

REPEAT = 16


def clip_labels(gt, n_clip):
    """GT collapsed to one boolean per clip -- the clip axis is sum(lens)."""
    return np.array([(gt[i * REPEAT:(i + 1) * REPEAT].mean() > 0.5)
                     for i in range(n_clip)], dtype=bool)


def const_scores(scores, lens):
    """Replace every clip's score with its video's mean."""
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, np.int64))])
    out = np.array(scores, dtype=np.float64, copy=True)
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        out[lo:hi] = out[lo:hi].mean()
    return out


def permute_within(scores, lens, rng):
    offs = np.concatenate([[0], np.cumsum(np.asarray(lens, np.int64))])
    out = np.array(scores, dtype=np.float64, copy=True)
    for i in range(len(lens)):
        lo, hi = int(offs[i]), int(offs[i + 1])
        out[lo:hi] = out[lo:hi][rng.permutation(hi - lo)]
    return out


def geometry(y, vids):
    lens_, covs, nev, multi, anom = [], [], [], 0, 0
    for lo, hi in vids:
        L = hi - lo
        ev = runs_of(y[lo:hi])
        if not ev:
            continue
        anom += 1
        nev.append(len(ev))
        multi += len(ev) > 1
        for a, b in ev:
            lens_.append(b - a)
            covs.append((b - a) / L)
    lens_ = np.array(lens_)
    covs = np.array(covs)
    return {"n_anom_videos": anom, "n_events": int(len(lens_)),
            "median_event_clips": float(np.median(lens_)) if len(lens_) else 0.0,
            "median_coverage": float(np.median(covs)) if len(covs) else 0.0,
            "frac_events_cov_ge_30pct": float((covs >= 0.3).mean()) if len(covs) else 0.0,
            "frac_multi_event_videos": float(multi / max(anom, 1)),
            "mean_events_per_anom_video": float(np.mean(nev)) if nev else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = {"xd": "classprobs_ucf2xd.npz", "ucf": "classprobs_xd2ucf.npz"}
    cfg_for = {"xd": ["A_ucf2xd", "C_xd2xd", "E_ens11_ucf2xd"],
               "ucf": ["B_xd2ucf", "D_ucf2ucf"]}
    out = {}

    for dom, f in src.items():
        lens = np.load(os.path.join(args.lens, f), allow_pickle=True)["lens"]
        vids = build_grid(lens)
        n_clip = int(np.sum(lens))          # FIX: clip axis, not video count
        assert vids[-1][1] == n_clip, (vids[-1][1], n_clip)
        print("=" * 96)
        print(f"{dom}: {len(vids)} videos, {n_clip} clips")
        print("=" * 96)
        g = None
        rows = {}
        for cfg in cfg_for[dom]:
            p = glob.glob(os.path.join(args.p3, cfg, "cdv_scores_*.npz"))
            if not p:
                continue
            z = np.load(p[0], allow_pickle=True)
            s, gt = z["prob2"], z["gt"]
            if g is None:
                g = clip_labels(gt, n_clip)
                geo = geometry(g, vids)
                out.setdefault(dom, {})["geometry"] = geo
                print(f"  GEOMETRY: anom videos={geo['n_anom_videos']} "
                      f"events={geo['n_events']} "
                      f"median event={geo['median_event_clips']:.0f} clips "
                      f"({100*geo['median_coverage']:.1f}% of video)")
                print(f"            events covering >=30% of video: "
                      f"{100*geo['frac_events_cov_ge_30pct']:.1f}%   "
                      f"multi-event videos: "
                      f"{100*geo['frac_multi_event_videos']:.1f}%")

            rng = np.random.RandomState(0)
            floor_f1, _, _, _ = event_f1_at_eer(rng.rand(n_clip), g, vids)
            const_f1, _, _, _ = event_f1_at_eer(const_scores(s, lens), g, vids)
            perm_f1, _, _, _ = event_f1_at_eer(
                permute_within(s, lens, np.random.RandomState(0)), g, vids)
            model_f1, _, _, _ = event_f1_at_eer(s, g, vids)
            rows[cfg] = {"floor": floor_f1, "const_video_level": const_f1,
                         "perm": perm_f1, "model": model_f1,
                         "model_over_const": model_f1 - const_f1,
                         "model_over_floor": model_f1 - floor_f1}
            print(f"  {cfg:<18} FLOOR={floor_f1:.4f}  "
                  f"CONST(video-level)={const_f1:.4f}  "
                  f"PERM={perm_f1:.4f}  MODEL={model_f1:.4f}   "
                  f"(model-const={model_f1-const_f1:+.4f})")
        out.setdefault(dom, {})["configs"] = rows

    print()
    print("=" * 96)
    print("READING")
    print("=" * 96)
    for dom, v in out.items():
        rows = v["configs"]
        if not rows:
            continue
        m = np.mean([r["model"] for r in rows.values()])
        c = np.mean([r["const_video_level"] for r in rows.values()])
        fl = np.mean([r["floor"] for r in rows.values()])
        pm = np.mean([r["perm"] for r in rows.values()])
        print(f"\n  {dom}: mean over {len(rows)} configs")
        print(f"    floor {fl:.4f} | const(video-level) {c:.4f} | "
              f"perm {pm:.4f} | model {m:.4f}")
        print(f"    model - const = {m-c:+.4f}  "
              f"(this is the value the models actually add beyond "
              "video-level ranking)")
        if m - c < 0.02:
            print("    => models are close to a pure video-level scorer; "
                  "the event metric on this\n       dataset does not separate "
                  "them, so the permutation result is expected,\n"
                  "       not a bug.  The frame-vs-event question cannot be "
                  "tested here.")
        else:
            print("    => models add real temporal localisation beyond the "
                  "video-level score.")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
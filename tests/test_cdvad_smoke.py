"""Layer-3 runtime smoke test for cdvad: synthetic tensors, no disk, no GPU.

Validates that the decomposition identity and the capacity control behave as
advertised *before* we spend time on real features.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cdvad import metrics as M  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


def main():
    rng = np.random.RandomState(0)
    print("[smoke] decompose")

    # Realistic mix: 4 normal videos (low video-mean) + 4 anomalous videos
    # (higher video-mean) whose positives sit inside a partly-normal video.
    y_list, s_list = [], []
    for v in range(4):
        y = np.zeros(200, dtype=np.float32)
        y_list.append(y)
        s_list.append((0.0 + 0.3 * v + 0.01 * rng.randn(200)).astype(np.float32))
    for v in range(4):
        y = np.zeros(200, dtype=np.float32)
        y[100:140] = 1.0
        y_list.append(y)
        s = 10.0 + v + 0.01 * rng.randn(200)
        s[100:140] += 1.0
        s_list.append(s.astype(np.float32))

    d = M.decompose(y_list, s_list)
    check("micro is finite", np.isfinite(d["micro"]))
    check("within > 0.9 (strong within-video signal)", d["within"] > 0.9,
          f"within={d['within']:.4f}")
    # w: 4 anomalous videos x (40 pos x 160 neg) = 25600 within pairs.
    # total pairs: 160 pos x (4*200 + 4*160) neg = 160 x 1440 = 230400
    #            -> w = 25600/230400 = 0.1111
    check("w == 0.1111", abs(d["w"] - 0.1111111111111111) < 1e-9, f"w={d['w']}")
    # identity: micro*total == within*w_p + cross*(1-w) in pair terms
    check("decomposition identity holds",
          abs(d["micro"] - (d["within"] * d["w"] + d["cross"] * (1 - d["w"]))) < 1e-6)

    print("[smoke] capacity control")
    cap = M.capacity_control(y_list, s_list)
    # Constant-per-video scores destroy all localisation -> within == nan
    # (every video in `keep` has anomalies, so no within-video pairs survive).
    check("constant scorer -> no within-video signal",
          (not np.isfinite(cap["within"])) or abs(cap["within"] - 0.5) < 1e-9,
          f"within={cap['within']}")
    check("constant scorer retains pooled AUC via video ranking",
          cap["micro"] > 0.7, f"micro={cap['micro']:.4f}")

    print("[smoke] video_level_auc")
    va = M.video_level_auc(y_list, s_list)
    check("video_level_auc finite", np.isfinite(va))

    print("[smoke] full_report keys")
    rep = M.full_report(y_list, s_list)
    for k in ("auc", "ap", "fpr95", "within_within", "within_w",
              "const_micro", "video_auc"):
        check(f"key {k} present", k in rep)

    print("[smoke] domain probes")
    ya = np.array([0, 0, 0, 1, 1, 1])
    sa = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])
    check("domain_auc == 1.0 for separable", abs(M.domain_auc(ya, sa) - 1.0) < 1e-9)
    check("within_normal_pair_auc == 1.0",
          abs(M.within_normal_pair_auc([np.zeros(10) + 0.1], [np.zeros(10) + 0.9]) - 1.0) < 1e-9)
    check("within_normal_pair_auc nan when empty",
          np.isnan(M.within_normal_pair_auc([], [np.zeros(3)])))

    print("[smoke] degenerate videos are skipped")
    y_degen = [np.zeros(50, dtype=np.float32)]   # all-normal: no pairs
    d2 = M.decompose(y_degen, [np.ones(50, dtype=np.float32)])
    check("all-normal video -> nan", np.isnan(d2["micro"]))

    print()
    if FAILED:
        print(f"SMOKE FAIL ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
"""G9 stage 1: dump PER-CLASS probabilities on the unseen target.

Why
---
vad-exp.md 13.3 established, causally, that cross-domain damage is a
CLASS-DEPENDENT boundary shift: on XD->UCF the normal class moves -0.657 while
the anomaly classes move -1.635, pushing the normal-vs-rest boundary +1.044.
That is the one mechanism in the whole archive with causal confirmation.

17 then tested a remedy and declared it dead -- but it tested a GLOBAL scalar
bias, which IS a monotone transform of the score, so 17.3 kills it for a
trivial reason that has nothing to do with the class-dependence.

A PER-CLASS bias is not globally monotone: it changes the ordering between
frames attributed to different classes, which is exactly the kind of change
17.6 says can move AP. As far as the archive shows, that was never tested.

To test it we need per-class scores. cdv_cross_eval.py calls
frame_probs(..., stack_logits=False), so the class stack was computed and
thrown away. This script recomputes it and saves it to THIS project's tree.

Isolation: reads the sibling project's modules read-only (same pattern the
sibling itself uses for VadCLIP-main, vad-exp.md 11.2) and writes only under
its own run root, named by VAD_ROOT. No sibling file is modified.

Usage:
    py.bat g9_dump_class_probs.py --source ucf --target xd --out <dir>
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

CDV_SRC = os.environ.get("CDV_ROOT", "crossdomain-vad") + r"\src"
VAD_SRC = os.environ.get("VADCLIP_ROOT", "VadCLIP") + r"\src"
for _p in (VAD_SRC, CDV_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cdv_paths  # noqa: E402

cdv_paths.setup()

from cdv_dataset import ClipVideoDataset, collate  # noqa: E402
from cdv_eval import frame_probs  # noqa: E402
from cdv_lists import load_test_list  # noqa: E402
from cdv_model import build_model  # noqa: E402
from cdv_prompts import prompt_strings  # noqa: E402

SOURCE_ARCH = {
    "xd": {"classes_num": 7, "visual_layers": 1, "attn_window": 64},
    "ucf": {"classes_num": 14, "visual_layers": 2, "attn_window": 8},
}
SHARED = {"embed_dim": 512, "visual_length": 256, "visual_width": 512,
          "visual_head": 1, "prompt_prefix": 10, "prompt_postfix": 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["ucf", "xd"])
    ap.add_argument("--target", required=True, choices=["ucf", "xd"])
    ap.add_argument("--checkpoint", default="official")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    arch = {**SHARED, **SOURCE_ARCH[args.source]}
    test_df = load_test_list(args.target)
    gt = np.load(cdv_paths.gt_path(args.target)).astype(np.float32)
    prompt_text = prompt_strings(args.target, "bare")

    ds = ClipVideoDataset(arch["visual_length"], test_mode=True,
                          dataset=args.target, frame=test_df)
    if args.max_clips:
        ds = torch.utils.data.Subset(ds, range(args.max_clips))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    ckpt = (cdv_paths.OFFICIAL_CHECKPOINTS[args.source]
            if args.checkpoint == "official" else args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch, device)
    model.load_base_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model.to(device)

    # stack_logits=True is the whole point: it keeps logits2's per-class softmax.
    p1, p2, cls, names, lens = frame_probs(
        model, loader, arch["visual_length"], prompt_text, device,
        stack_logits=True)

    # frame_probs returns the class stack as a list of per-batch arrays
    cls_arr = (np.concatenate(cls, axis=0) if isinstance(cls, list)
               else np.asarray(cls))
    print(f"[class stack] shape={None if cls_arr is None else cls_arr.shape}")
    print(f"[names] {names}")
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(
        args.out, f"classprobs_{args.source}2{args.target}.npz")
    np.savez_compressed(out, prob1=p1, prob2=p2, cls=cls_arr,
                        names=np.asarray(names, dtype=object),
                        lens=np.asarray(lens), gt=gt)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
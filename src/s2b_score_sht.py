"""s2b_score_sht: score the ShanghaiTech test split with the existing checkpoints.

Why a bespoke Dataset
---------------------
cdv_dataset.ClipVideoDataset calls ``parse_label(label, dataset)``, and that
table only knows UCF and XD label vocabularies.  ShanghaiTech uses its own
class names, so passing its real labels through would raise -- or worse, be
silently coerced into the wrong binary label.

But ``frame_probs`` does not use the binary label at all: it reads
``item[0]`` (features), ``item[1][0]`` (a name for the record) and ``item[2]``
(the length).  So this Dataset satisfies exactly that contract and returns a
placeholder binary label.  The model forward pass, the window loop, the length
bookkeeping and the score construction are all the SAME code as the UCF/XD runs
that were verified bit-exactly, so SHT scores are directly comparable.

Protocol, checked not assumed
-----------------------------
tools.process_split cuts features into windows of ``visual_length``; that is an
internal detail.  The clip grid is the row count of each feature file, which is
what ``lens`` records.  ``sum(lens) * 16`` must equal ``len(gt_sht)`` -- that
identity is asserted before anything is scored, because a silent misalignment
here is precisely the defect class this project keeps auditing.

Prompts
-------
No ShanghaiTech prompt table is invented.  A cross-domain evaluation applies the
SOURCE domain's prompts to the target, which is the standard zero-shot
protocol; inventing target-specific prompt words would be prompt fishing and
would make SHT incomparable to the UCF/XD results.
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv as csvmod
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CDV_SRC = os.environ.get("CDV_ROOT", "crossdomain-vad") + r"\src"
VAD_SRC = os.environ.get("VADCLIP_ROOT", "VadCLIP") + r"\src"
for _p in (VAD_SRC, CDV_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cdv_paths  # noqa: E402

cdv_paths.setup()

import utils.tools as tools  # noqa: E402
from cdv_dataset import collate  # noqa: E402  (reference collate, unchanged)
from cdv_eval import frame_probs  # noqa: E402
from cdv_model import build_model  # noqa: E402
from cdv_prompts import prompt_strings  # noqa: E402

REPEAT = 16
SOURCE_ARCH = {
    "xd": {"classes_num": 7, "visual_layers": 1, "attn_window": 64},
    "ucf": {"classes_num": 14, "visual_layers": 2, "attn_window": 8},
}
SHARED = {"embed_dim": 512, "visual_length": 256, "visual_width": 512,
          "visual_head": 1, "prompt_prefix": 10, "prompt_postfix": 10}


class PlainTestSet(Dataset):
    """Satisfies frame_probs' contract without cdv_dataset's label parsing."""

    def __init__(self, paths, names, clip_dim):
        self.paths, self.names, self.clip_dim = paths, names, clip_dim

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        feat = np.load(self.paths[i])
        feat, length = tools.process_split(feat, self.clip_dim)
        feat = torch.from_numpy(np.ascontiguousarray(feat)).float()
        return feat, (self.names[i],), int(length), float(0.0)


def read_list(sht_dir):
    """Locate the test list and return absolute feature paths in file order."""
    for cand in ("sht_CLIP_rgbtest.csv", "sht_CLIP_rgb.csv"):
        p = os.path.join(sht_dir, cand)
        if os.path.exists(p):
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csvmod.DictReader(f))
            cols = list(rows[0].keys())
            pk = "path" if "path" in cols else cols[0]
            out = []
            for r in rows:
                rel = str(r[pk]).replace("\\", "/")
                fn = os.path.basename(rel)
                out.append(os.path.join(sht_dir, "test", fn))
            return out, [os.path.basename(str(r[pk])) for r in rows], cols
    # fall back to directory order, which must agree with the GT order
    test_dir = os.path.join(sht_dir, "test")
    files = sorted(os.listdir(test_dir))
    return ([os.path.join(test_dir, f) for f in files], files, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sht", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths, names, cols = read_list(args.sht)
    gt = np.load(os.path.join(args.sht, "gt_sht.npy"))
    total_clips = sum(int(np.load(p).shape[0]) for p in paths)
    print(f"list columns     : {cols}")
    print(f"videos           : {len(paths)}")
    print(f"clips in list    : {total_clips}")
    print(f"len(gt_sht)      : {len(gt)}  -> {len(gt)/REPEAT:.1f} clips implied")
    ok = (total_clips * REPEAT == len(gt))
    print(f"ALIGNMENT sum(lens)*16 == len(gt): {ok}")
    if not ok:
        raise SystemExit("[GUARD] SHT list and GT disagree -- refusing to score")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"[GUARD] {len(missing)} listed features missing, "
                         f"e.g. {missing[:2]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for src in ("ucf", "xd"):
        arch = {**SHARED, **SOURCE_ARCH[src]}
        ds = PlainTestSet(paths, names, arch["visual_length"])
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
        ckpt = cdv_paths.OFFICIAL_CHECKPOINTS[src]
        model = build_model(arch, device)
        model.load_base_state_dict(torch.load(ckpt, map_location="cpu"),
                                   strict=True)
        model.to(device)
        p1, p2, _cls, rnames, lens = frame_probs(
            model, loader, arch["visual_length"], prompt_strings(src, "bare"),
            device, stack_logits=False)
        lens = np.asarray(lens, dtype=np.int64)
        assert int(lens.sum()) * REPEAT == len(gt), (
            int(lens.sum()), len(gt) / REPEAT)
        out = os.path.join(args.out, f"cdv_scores_{src}2sht.npz")
        np.savez_compressed(out, prob1=p1, prob2=p2, gt=gt, lens=lens,
                            names=np.asarray(rnames, dtype=object))
        print(f"[{src}2sht] clips={int(lens.sum())} videos={len(lens)} "
              f"-> {out}")
        results[f"{src}2sht"] = {"n_clips": int(lens.sum()),
                                 "n_videos": int(len(lens))}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("SHT SCORING COMPLETE")


if __name__ == "__main__":
    main()
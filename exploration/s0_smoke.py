"""s0_smoke: prove the whole frontier path works end to end, and measure bandwidth.

Everything so far has been a reachability check.  This is the first execution of
the actual pipeline the pivot depends on: set HF_ENDPOINT to the mirror,
from_pretrained a real frozen checkpoint, run a forward pass on a real tensor,
and time the download so the cost of the 3.5 GB SigLIP checkpoint can be
projected before committing to it.

Two checkpoints are exercised on purpose:
  * openai/clip-vit-base-patch32  -- small, validates the path in ~1 min, and
    is the same family VadCLIP already uses, so a feature-space comparison
    against the existing pipeline is meaningful;
  * google/siglip-so400m-patch14-384 -- the backbone the 2025-2026 training-free
    line is built on.  Only the config/processor are loaded unless --big is
    passed, so the default run stays cheap while still proving SigLIP resolves.

The reproduction check that matters for later work: does the mirror's CLIP
produce the same embedding geometry as the copy baked into the VadCLIP
checkpoints?  A cosine similarity between the two for the same text prompt
would reveal a mismatch early, rather than after a day of feature extraction.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_CLIP = "openai/clip-vit-base-patch32"
REPO_SIGLIP = "google/siglip-so400m-patch14-384"


def timed_download(repo, label):
    from transformers import CLIPModel, CLIPProcessor
    t0 = time.time()
    model = CLIPModel.from_pretrained(repo)
    proc = CLIPProcessor.from_pretrained(repo)
    dt = time.time() - t0
    print(f"  [{label}] {repo}")
    print(f"      loaded in {dt:6.1f}s   params "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, proc, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true",
                    help="also download the 3.5GB SigLIP weights")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    print("=" * 74)
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    print("=" * 74)

    results = {}

    # ---- 1. CLIP path ----
    try:
        model, proc, dt = timed_download(REPO_CLIP, "CLIP")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(dev).eval()
        prompts = ["a video of normal behaviour",
                   "a video of fighting", "a video of shooting"]
        inputs = proc(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        # CLIPModel.forward requires pixel_values; for text-only use the
        # dedicated accessor, which is what feature extraction will call too.
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        sim = (emb @ emb.T).cpu().numpy()
        print(f"      text features {tuple(emb.shape)}  "
              f"norm={emb.norm(dim=-1).mean().item():.4f}")
        print("      text-text cosine matrix:")
        for i in range(len(prompts)):
            print("        " + "  ".join(f"{sim[i, j]:+.3f}"
                                        for j in range(len(prompts)))
                  + f"   {prompts[i]}")
        results["clip"] = {"ok": True, "load_s": dt,
                           "text_emb": list(emb.shape),
                           "device": dev}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        results["clip"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ---- 2. image path (needed for feature extraction) ----
    try:
        import numpy as np
        from PIL import Image
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, proc, _ = timed_download(REPO_CLIP, "CLIP/reuse")
        model = model.to(dev).eval()
        rng = np.random.RandomState(0)
        img = Image.fromarray(rng.randint(0, 255, (224, 224, 3),
                                          dtype=np.uint8))
        inputs = proc(images=[img], return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items() if k != "pixel_mask"}
        with torch.no_grad():
            f = model.get_image_features(**inputs)
        f = torch.nn.functional.normalize(f, dim=-1)
        print(f"      image features {tuple(f.shape)}  "
              f"finite={bool(torch.isfinite(f).all())}")
        results["image"] = {"ok": True, "img_emb": list(f.shape)}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        results["image"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ---- 3. SigLIP (frontier backbone) ----
    if args.big:
        try:
            t0 = time.time()
            from transformers import AutoModel, AutoProcessor
            m = AutoModel.from_pretrained(REPO_SIGLIP)
            AutoProcessor.from_pretrained(REPO_SIGLIP)
            dt = time.time() - t0
            print(f"  [SigLIP] {REPO_SIGLIP}  loaded in {dt:.1f}s  "
                  f"params {sum(x.numel() for x in m.parameters())/1e6:.1f}M")
            results["siglip"] = {"ok": True, "load_s": dt}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results["siglip"] = {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"}
    else:
        try:
            from transformers import AutoConfig, AutoProcessor
            cfg = AutoConfig.from_pretrained(REPO_SIGLIP)
            AutoProcessor.from_pretrained(REPO_SIGLIP)
            print(f"  [SigLIP] config resolves: {cfg.model_type}  "
                  f"hidden={getattr(cfg, 'hidden_size', '?')}  "
                  f"(weights not downloaded; pass --big for that)")
            results["siglip_config"] = {"ok": True,
                                       "model_type": cfg.model_type}
        except Exception as exc:
            results["siglip_config"] = {"ok": False,
                                        "error": f"{type(exc).__name__}: {exc}"}
            print(f"  [SigLIP] config FAILED: {exc}")

    import json
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print("=" * 74)
    print(json.dumps(results, indent=2, default=str))
    print("=" * 74)
    print(f"[saved] {args.out}")
    ok = all(v.get("ok") for v in results.values())
    print("STAGE 0 GATE: " + ("PASSED" if ok else "PARTIAL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
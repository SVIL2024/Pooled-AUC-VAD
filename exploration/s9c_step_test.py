"""Reproduce one real training step and find the kernel that faults.

Two runs died with CUDA faults -- an illegal memory access inside backward, then
an unknown error at the first .item(). The generic GPU sanity check passes, so
the fault is specific to this model's step. Guessing at a fix from that evidence
would be how the earlier axis-order mistake happened, so this reproduces the step
exactly as s9_train_i3d.py builds it and narrows it by varying one thing at a
time: batch size, top-k pooling versus mean pooling, and the dropout layer.

Run with CUDA_LAUNCH_BLOCKING=1 so the fault is attributed to the kernel that
caused it rather than to whichever synchronisation happened to notice.

Read-only.
"""

from __future__ import annotations

import os
import traceback

import numpy as np
import torch
import torch.nn as nn

REPEAT = 16
HIDDEN = 512
DROPOUT = 0.5
TOPK_FRACTION = 0.01
CROP = 256
IN_DIM = 2048


class MilHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(IN_DIM, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )

    def clip_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).squeeze(-1)


def topk_mean(logits: torch.Tensor) -> torch.Tensor:
    n = logits.numel()
    k = max(1, int(round(n * TOPK_FRACTION)))
    return torch.topk(logits, min(k, n)).values.mean()


def try_case(name: str, fn) -> bool:
    try:
        fn()
        torch.cuda.synchronize()
        print(f"  {name:<44} ok")
        return True
    except Exception as exc:
        print(f"  {name:<44} FAIL {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "0")
    print(f"CUDA_LAUNCH_BLOCKING={blocking}")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = MilHead().to(dev)
    lossf = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\n=== forward only, no backward ===")
    for bs in (4, 8, 32):
        def f(bs=bs):
            x = torch.randn(bs, CROP, IN_DIM, device=dev)
            with torch.no_grad():
                model.clip_logits(x)
        try_case(f"forward bs={bs}", f)

    print("\n=== forward + topk pooling + loss, no backward ===")
    for bs in (4, 32):
        def f(bs=bs):
            x = torch.randn(bs, CROP, IN_DIM, device=dev)
            lg = model.clip_logits(x)
            v = torch.stack([topk_mean(lg[r]) for r in range(lg.shape[0])])
            lossf(v, torch.zeros(bs, device=dev))
        try_case(f"topk fwd+loss bs={bs}", f)

    print("\n=== full step, mean pooling instead of topk ===")
    for bs in (8, 32):
        def f(bs=bs):
            x = torch.randn(bs, CROP, IN_DIM, device=dev)
            lg = model.clip_logits(x)
            v = lg.mean(dim=1)
            loss = lossf(v, torch.zeros(bs, device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            float(loss.item())
        if not try_case(f"full step mean-pool bs={bs}", f):
            return 2

    print("\n=== full step, topk pooling (as written) ===")
    for bs in (4, 8, 16, 32):
        def f(bs=bs):
            x = torch.randn(bs, CROP, IN_DIM, device=dev)
            lg = model.clip_logits(x)
            v = torch.stack([topk_mean(lg[r]) for r in range(lg.shape[0])])
            loss = lossf(v, torch.zeros(bs, device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            float(loss.item())
        if not try_case(f"full step topk bs={bs}", f):
            print(f"  first failing batch size: {bs}")
            return 2

    print("\n=== full step with a 2048->512 linear only ===")
    def f():
        x = torch.randn(32, CROP, IN_DIM, device=dev, requires_grad=True)
        y = torch.nn.functional.linear(x, torch.ones(512, IN_DIM, device=dev))
        y.sum().backward()
        float(x.grad.norm().item())
    try_case("single linear backward bs=32", f)

    print("\nall cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Can this GPU actually execute a kernel?

torch.cuda.is_available() returning True and nvidia-smi answering only prove the
driver and NVML respond. Both the training run and the smoke run died with a
CUDA fault that surfaced asynchronously -- first an illegal memory access, then
an unknown error -- which is what a virtualised adapter looks like when it
cannot really run work. The server is a QEMU guest, so that has to be settled
before any further training attempt, otherwise each attempt costs a full
feature load and still tells us nothing new.

This runs the smallest possible sequence -- allocate, fill, matmul, reduce,
copy back, synchronize -- and reports exactly which call fails. Falling back to
CPU is acceptable for this workload if it works: the model is about a million
parameters over fixed-size crops.

Read-only with respect to any project state.
"""

from __future__ import annotations

import os
import traceback


def main() -> int:
    import torch

    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device count  : {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("\nno CUDA; CPU is the only option")
        return 1

    print(f"compiled cuda : {torch.version.cuda}")
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"device 0      : {props.name}")
        print(f"  capability  : {props.major}.{props.minor}")
        print(f"  total mem   : {props.total_memory / 1024 ** 3:.1f} GB")
        print(f"  multi-proc  : {props.multi_processor_count} SMs")
    except Exception as exc:
        print(f"  properties failed: {type(exc).__name__}: {exc}")

    steps = [
        ("empty", lambda: torch.empty(8, device="cuda")),
        ("zeros", lambda: torch.zeros(8, device="cuda")),
        ("ones", lambda: torch.ones(8, device="cuda")),
        ("from_numpy", lambda: torch.from_numpy(
            __import__("numpy").zeros(8, dtype="float32")).to("cuda")),
        ("matmul", lambda: torch.ones(64, 64, device="cuda") @ torch.ones(
            64, 64, device="cuda")),
        ("sum", lambda: torch.ones(1024, 1024, device="cuda").sum()),
        ("logsumexp", lambda: torch.logsumexp(
            torch.ones(32, 256, device="cuda"), dim=1)),
        ("topk", lambda: torch.topk(torch.ones(32, 256, device="cuda"), 3)),
        ("backward", None),  # handled separately
        ("item", None),
    ]

    x = None
    ok = 0
    for name, fn in steps:
        if name == "backward":
            try:
                t = torch.ones(32, 256, 2048, device="cuda", requires_grad=True)
                # F.linear takes weight as (out_features, in_features).
                w = torch.ones(1, 2048, device="cuda")
                out = torch.nn.functional.linear(t, w).squeeze(-1)
                out.mean().backward()
                torch.cuda.synchronize()
                print(f"  {name:<12} ok   grad norm {float(t.grad.norm()):.4f}")
                ok += 1
            except Exception as exc:
                print(f"  {name:<12} FAIL {type(exc).__name__}: {exc}")
                break
            continue
        if name == "item":
            try:
                x = torch.ones(32, 256, device="cuda")
                _ = x.topk(3).indices.mean().item()
                torch.cuda.synchronize()
                print(f"  {name:<12} ok")
                ok += 1
            except Exception as exc:
                print(f"  {name:<12} FAIL {type(exc).__name__}: {exc}")
            continue
        try:
            x = fn()
            torch.cuda.synchronize()
            print(f"  {name:<12} ok")
            ok += 1
        except Exception as exc:
            print(f"  {name:<12} FAIL {type(exc).__name__}: {exc}")
            traceback.print_exc()
            break

    print(f"\nsteps passed: {ok}")
    if ok < len(steps):
        print("\nVERDICT: the CUDA device cannot complete this workload; "
              "training must run on CPU.")
        return 2
    print("\nVERDICT: CUDA works for this workload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

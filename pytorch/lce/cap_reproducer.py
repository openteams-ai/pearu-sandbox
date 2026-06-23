"""Reproducer: chunked auto memory exceeds the reference in the budget regime.

Sweeps ``in_features`` (D) at fixed (num_tokens N, num_classes V) for a bf16
probability target and plots peak CUDA memory for three configs:

  * reference        -- options=None (full F.linear + F.cross_entropy)
  * auto (uncapped)  -- LinearCrossEntropyOptions(); prob resolves to
                        aspect_ratio factor 1, so for D >= V it degenerates to
                        a single chunk that materializes (N, V) PLUS the chunked
                        scratch (incl. the prob-only (N, V) prob_target_buf) and
                        crosses ABOVE reference -- the regression we cap.
  * capped           -- batch_chunk_size = min(aspect_ratio_B,
                        floor_pow2(coeff * N*V/D)) (the proposed
                        min(aspect_ratio_B, floor_tile(N*V/4D)) rule, with
                        floor_pow2 as the conservative default). B_ref =
                        N*V/(4*D) is the largest chunk that stays <= reference;
                        coeff defaults to 0.25 (bf16+compact INDEX fit) and is
                        re-fit here for prob, whose extra prob_target_buf raises
                        the per-row bytes.

Probability targets illustrate the crossing more sharply than index targets
(factor-1 auto + the extra (N, V) buffer). Local-GPU first instance; re-run on
A100 once the cap lands in the op.

    python cap_reproducer.py
    python cap_reproducer.py --num-tokens 4096 --num-classes 8192 \
        --in-features 1024 2048 4096 8192 16384 32768 --coeff 0.25
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import LinearCrossEntropyOptions

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_cap import _aspect_ratio_b, _next_pow2  # reuse the shared formulas

GiB = 1024 ** 3


def floor_pow2(x: float) -> int:
    return 1 << int(math.floor(math.log2(x))) if x >= 1 else 1


def b_cap(n, d, v, coeff, factor):
    asp = _aspect_ratio_b(n, d, v, factor)
    b_ref = floor_pow2(coeff * n * v / d)
    return max(1, min(asp, b_ref, n)), asp, b_ref


def peak_mem(inp0, w0, t, options):
    inp = inp0.detach().requires_grad_(True)
    w = w0.detach().requires_grad_(True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loss = F.linear_cross_entropy(inp, w, t, reduction="mean", options=options)
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del loss, inp, w
    return peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-tokens", type=int, default=4096)
    p.add_argument("--num-classes", type=int, default=8192)
    p.add_argument("--in-features", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--coeff", type=float, default=0.25, help="B_ref = coeff * N*V/D")
    p.add_argument("--factor", type=int, default=1, help="aspect_ratio factor (prob auto = 1)")
    p.add_argument("--target", default="index", choices=["index", "prob"])
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "sweep_out_prob"))
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    dev = "cuda"
    dtype = getattr(torch, args.dtype)
    name = torch.cuda.get_device_name(0)
    N, V = args.num_tokens, args.num_classes
    g = torch.Generator(device=dev).manual_seed(args.seed)

    rows = []
    print(f"# {name}  dtype={args.dtype}  {args.target}-target  N={N} V={V}  coeff={args.coeff}")
    print(f"# {'D':>7} {'ar1_B':>7} {'B_ref':>7} {'B_cap':>7} | {'ref_G':>6} {'auto_G':>6} {'ar1_G':>6} {'cap_G':>6} | ar1>ref?")
    for D in args.in_features:
        inp0 = torch.randn(N, D, device=dev, dtype=dtype, generator=g)
        w0 = torch.randn(V, D, device=dev, dtype=dtype, generator=g) / (D ** 0.5)
        if args.target == "prob":
            t = torch.softmax(torch.randn(N, V, device=dev, dtype=torch.float32, generator=g), dim=1).to(dtype)
        else:
            t = torch.randint(0, V, (N,), device=dev, generator=g)

        bc, asp, bref = b_cap(N, D, V, args.coeff, args.factor)
        ref = peak_mem(inp0, w0, t, None) / GiB
        auto = peak_mem(inp0, w0, t, LinearCrossEntropyOptions()) / GiB
        ar1 = peak_mem(inp0, w0, t, LinearCrossEntropyOptions(acc_policy="compact", chunking_method="aspect_ratio")) / GiB
        cap = peak_mem(inp0, w0, t, LinearCrossEntropyOptions(batch_chunk_size=bc, chunking_method=None)) / GiB
        rows.append((D, asp, bref, bc, ref, auto, ar1, cap))
        flag = "  <-- ar1 REGRESSES" if ar1 > ref else ""
        print(f"  {D:>7} {asp:>7} {bref:>7} {bc:>7} | {ref:>6.3f} {auto:>6.3f} {ar1:>6.3f} {cap:>6.3f} |{flag}")
        del inp0, w0, t
        torch.cuda.empty_cache()

    Ds = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(Ds, [r[4] for r in rows], "o-", color="black", label="reference (full materialization)")
    ax1.plot(Ds, [r[5] for r in rows], "D-", color="tab:gray", label="auto (current default)")
    ax1.plot(Ds, [r[6] for r in rows], "s-", color="tab:red", label="aspect_ratio factor 1 (proposed, uncapped)")
    ax1.plot(Ds, [r[7] for r in rows], "^-", color="tab:green",
             label=f"capped: min(aspect_ratio[1], floor_pow2({args.coeff}*N*V/D))")
    ax1.axvline(V, ls=":", color="gray", lw=1)
    ax1.annotate("D = V (crossing)", (V, ax1.get_ylim()[1]), ha="center", va="top", color="gray", fontsize=8)
    ax1.set_ylabel("peak CUDA memory (GiB)")
    ax1.set_title(f"Factor-1 chunking exceeds reference for D >= V; the N*V/4D cap restores <= reference\n"
                  f"{name}, {args.dtype} {args.target} target, N={N}, V={V}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(Ds, [r[1] for r in rows], "s-", color="tab:red", label="aspect_ratio_B (factor 1, uncapped)")
    ax2.plot(Ds, [r[3] for r in rows], "^-", color="tab:green", label="B_cap")
    ax2.plot(Ds, [r[2] for r in rows], ":", color="tab:blue", label=f"floor_pow2({args.coeff}*N*V/D)")
    ax2.axhline(N, ls="--", color="gray", lw=1, label=f"N={N} (single chunk)")
    ax2.axvline(V, ls=":", color="gray", lw=1)
    ax2.set_xlabel("in_features (D)")
    ax2.set_ylabel("batch_chunk_size B")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = name.replace(" ", "_")
    fpath = out / f"cap_reproducer_{args.dtype}_{slug}_{args.target}.png"
    fig.tight_layout()
    fig.savefig(fpath, dpi=130)
    print(f"\nwrote {fpath}")


if __name__ == "__main__":
    main()

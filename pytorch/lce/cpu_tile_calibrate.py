"""Calibrate the CPU ``batch_tile_size`` model for linear_cross_entropy.

The chunked LCE op's per-row time follows ``per_row = a + b/B`` (B =
batch_chunk_size): ``a`` = compute floor per row, ``b`` = fixed per-chunk
overhead (Python loop + dispatch). per_row is monotone-decreasing in B and
flattens -- expected, and it stays that way until RAM limits (we never hit a
rising tail here). The useful quantity is NOT a periodic tile but the
*decrement granularity* of floor_tile: the knee

    B_tile = b / (alpha * a)        (overhead == alpha * compute floor)

below which per-row degrades. The model predicts how B_tile depends on shape:
if ``a ~ s * K * N`` (3 GEMMs' FLOPs/row) and ``b ~ const`` (shape-independent
overhead), then ``B_tile ~ b / (alpha * s * K * N)`` -- it SHRINKS for large
heads (their big per-chunk compute amortizes the fixed overhead at small B).

This script fits a,b across a (K=in_features, N=num_classes) grid, checks those
scalings, and prints the constants + a pasteable summary so the same run on
different CPUs reveals the variability of B_tile. Single-thread by default
(--threads 1) for a clean, core-architecture-comparable fit.

    python cpu_tile_calibrate.py            # default grid, single-thread
    python cpu_tile_calibrate.py --threads 0 --reps 8   # real (all-core) config
"""
import argparse
import platform
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def fit_ab(D, V, NT, Bs, dt, reps, warmup):
    x = torch.randn(NT, D, dtype=dt, requires_grad=True)
    w = torch.randn(V, D, dtype=dt, requires_grad=True)
    tg = torch.randint(0, V, (NT,))
    per_row = []
    for B in Bs:
        opts = LinearCrossEntropyOptions(chunking_method=None, batch_chunk_size=B)

        def once():
            x.grad = None
            w.grad = None
            F.linear_cross_entropy(x, w, tg, reduction="mean", options=opts).backward()

        for _ in range(warmup):
            once()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            once()
            ts.append((time.perf_counter() - t0) * 1e6)
        per_row.append(statistics.median(ts) * 1e3 / NT)  # ns/row
    Barr = np.array(Bs, float)
    b, a = np.polyfit(1.0 / Barr, per_row, 1)  # per_row ~ a + b*(1/B)
    pred = a + b / Barr
    ss = 1.0 - np.sum((per_row - pred) ** 2) / np.sum((per_row - np.mean(per_row)) ** 2)
    return a, b, ss, per_row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dtype", choices=_DTYPES, default="float32")
    p.add_argument("--num-tokens", type=int, default=2048)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", type=int, default=1, help="0 = torch default (all cores)")
    p.add_argument("--alpha", type=float, default=0.1, help="knee: overhead == alpha*floor")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    dt = _DTYPES[args.dtype]
    NT = args.num_tokens
    Bs = [32, 48, 64, 96, 128, 192, 256, 384, 512]
    # (K, N) grid with repeated K*N products to confirm a depends on the product.
    grid = [(512, 1024), (1024, 1024), (512, 2048), (1024, 2048), (512, 4096), (1024, 4096)]

    rows = []
    for D, V in grid:
        a, b, ss, _ = fit_ab(D, V, NT, Bs, dt, args.reps, args.warmup)
        b_tile = b / (args.alpha * a)
        rows.append((D, V, D * V, a, b, ss, b_tile))

    cpu = platform.processor() or platform.machine() or "CPU"
    print(f"# CPU calibration | {cpu} | torch threads={torch.get_num_threads()} "
          f"| {args.dtype} | NT={NT} | alpha={args.alpha}")
    print(f"{'K':>6}{'N':>7}{'K*N':>10}{'a(ns/row)':>12}{'b(ns/chunk)':>13}"
          f"{'fitR2':>7}{'B_tile':>8}")
    for D, V, KN, a, b, ss, bt in rows:
        print(f"{D:>6}{V:>7}{KN:>10}{a:>12.1f}{b:>13.0f}{ss:>7.3f}{bt:>8.0f}")

    KN = np.array([r[2] for r in rows], float)
    aa = np.array([r[3] for r in rows], float)
    bb = np.array([r[4] for r in rows], float)
    s, c = np.polyfit(KN, aa, 1)  # a ~ s*K*N + c
    a_r2 = 1.0 - np.sum((aa - (s * KN + c)) ** 2) / np.sum((aa - aa.mean()) ** 2)
    b_cv = statistics.pstdev(bb) / statistics.mean(bb)
    print(f"\na ~ {s:.3e} * K*N + {c:.0f}   (R2={a_r2:.3f}; a proportional to K*N if R2~1, c small)")
    print(f"b = {statistics.mean(bb):.0f} +/- {statistics.pstdev(bb):.0f}  "
          f"(CV={b_cv:.0%}; b ~ const if CV small)")
    print(f"=> B_tile(K,N) ~= b / (alpha * s * K*N) = "
          f"{statistics.mean(bb) / (args.alpha * s):.3e} / (K*N)")
    for D, V in [(2048, 32768), (4096, 128256)]:
        print(f"   predicted B_tile for K={D}, N={V}: "
              f"{statistics.mean(bb) / (args.alpha * s * D * V):.0f}")


if __name__ == "__main__":
    main()

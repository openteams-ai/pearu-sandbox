"""Chunk-size sweep for linear_cross_entropy's ``auto*F`` / ``batch_tile_size``.

Two modes:

* default (bare forward GEMM) -- ``input(M, K) @ weight.T -> (M, N)``, M = chunk
  size, K = in_features, N = num_classes, fp32-accumulated. If the kernel tiles
  M at granularity T, GEMM time is a staircase stepping every T rows, so
  time-per-row is a sawtooth (minimal at multiples of T). Reads off the tile T.

* ``--op`` (full chunked LCE op, fwd+bwd over ``--num-tokens`` rows) -- sweeps
  ``batch_chunk_size``. The op's per-row follows ``a + b/B`` (compute floor +
  per-chunk overhead / B), so it falls monotonically with B and flattens at a
  knee. Any *tile* structure is a small periodic ripple on top of that trend,
  so this mode fits and subtracts ``a + b/B`` and plots the residual to expose
  the period.

CUDA uses cuda-event timing; CPU uses perf_counter. Writes an annotated PNG.

Usage:
    python tile_sweep.py --dtype bfloat16                          # A100 GEMM
    python tile_sweep.py --op --device cpu --dtype float32 \\
        --in-features 1024 --num-classes 2048 --num-tokens 4096 \\
        --m-max 1024 --reps 5
"""
import argparse
import platform
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=_DTYPES, default="bfloat16")
    p.add_argument("--in-features", type=int, default=2048, help="K")
    p.add_argument("--num-classes", type=int, default=32768, help="N (num_classes)")
    p.add_argument("--m-min", type=int, default=16)
    p.add_argument("--m-max", type=int, default=1024)
    p.add_argument("--m-step", type=int, default=16)
    p.add_argument("--reps", type=int, default=80)
    p.add_argument("--out-dtype", choices=list(_DTYPES) + ["same"], default="float32")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--op", action="store_true",
                   help="sweep the full chunked LCE op (fwd+bwd) vs the bare GEMM")
    p.add_argument("--num-tokens", type=int, default=4096, help="total batch for --op")
    args = p.parse_args()
    dev = args.device
    if dev == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"

    dt = _DTYPES[args.dtype]
    odt = dt if args.out_dtype == "same" else _DTYPES[args.out_dtype]
    mm_kw = {} if odt == dt else {"out_dtype": odt}  # out_dtype is CUDA-only
    K, N = args.in_features, args.num_classes

    def timed(fn, warmup):
        for _ in range(warmup):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.reps):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e) * 1e3)
            return statistics.median(ts)
        ts = []  # CPU ops are synchronous
        for _ in range(args.reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(ts)

    if args.op:
        NT = args.num_tokens
        x = torch.randn(NT, K, device=dev, dtype=dt, requires_grad=True)
        w = torch.randn(N, K, device=dev, dtype=dt, requires_grad=True)
        tg = torch.randint(0, N, (NT,), device=dev)

        def measure(B):
            opts = LinearCrossEntropyOptions(chunking_method=None, batch_chunk_size=B)

            def once():
                x.grad = None
                w.grad = None
                F.linear_cross_entropy(x, w, tg, reduction="mean", options=opts).backward()

            return timed(once, 4)
        per_row_denom = NT
    else:
        wt = torch.randn(N, K, device=dev, dtype=dt).t().contiguous()

        def measure(M):
            xm = torch.randn(M, K, device=dev, dtype=dt)
            return timed(lambda: torch.mm(xm, wt, **mm_kw), 15)
        per_row_denom = None  # per-row = us / M

    name = (torch.cuda.get_device_name() if dev == "cuda"
            else (platform.processor() or platform.machine() or "CPU"))
    Ms = list(range(args.m_min, args.m_max + 1, args.m_step))
    us = [measure(m) for m in Ms]
    nspr = [u * 1e3 / (per_row_denom or m) for u, m in zip(us, Ms)]

    print(f"# {name} ({dev}) | {args.dtype} | "
          f"{'LCE op fwd+bwd NT=' + str(args.num_tokens) if args.op else 'GEMM'} "
          f"K={K} N={N}")
    print(f"{'B' if args.op else 'M':>6}{'us':>12}{'ns/row':>10}")
    for m, u, nr in zip(Ms, us, nspr):
        print(f"{m:>6}{u:>12.1f}{nr:>10.1f}")

    if args.op:
        # per_row = a + b/B (compute floor + per-chunk overhead). Subtract it;
        # any tile structure shows as a periodic residual.
        Marr, y = np.array(Ms, float), np.array(nspr, float)
        b, a = np.polyfit(1.0 / Marr, y, 1)  # y ~ a + b*(1/M)
        trend = (a + b / Marr).tolist()
        resid = (y - np.array(trend)).tolist()
        rmins = [Ms[i] for i in range(1, len(resid) - 1)
                 if resid[i] <= resid[i - 1] and resid[i] < resid[i + 1]]
        period = (int(statistics.median([q - p for p, q in zip(rmins, rmins[1:])]))
                  if len(rmins) >= 2 else None)
        mono = "monotone-ish" if y[-1] <= min(y[:len(y) // 2]) else "NON-monotone (rises at large B)"
        knee = next(Ms[i] for i in range(len(y)) if y[i] <= 1.10 * y.min())
        print(f"\ntrend per_row ~= {a:.0f} + {b:.0f}/B   knee~={knee}   trend@1024 vs min: {mono}")
        print(f"residual minima: {rmins}")
        print(f"=> residual period ~= {period}")
        _plot(args, name, dev, K, N, Ms, us, nspr, "B = batch_chunk_size",
              knee=knee, trend=trend, resid=resid, rmins=rmins, period=period)
    else:
        troughs = [Ms[i] for i in range(1, len(nspr) - 1)
                   if nspr[i] <= nspr[i - 1] and nspr[i] < nspr[i + 1]]
        tile = int(statistics.median([q - p for p, q in zip(troughs, troughs[1:])])) \
            if len(troughs) >= 2 else None
        bp = [nspr[Ms.index(m) + 1] / nspr[Ms.index(m)] - 1
              for m in troughs if Ms.index(m) + 1 < len(nspr)]
        print(f"\n=> M-tile ~= {tile}   worst per-row penalty ~= {(max(bp) if bp else 0):.0%}")
        _plot(args, name, dev, K, N, Ms, us, nspr, "M = chunk size",
              troughs=troughs, tile=tile, penalty=max(bp) if bp else 0.0)


def _plot(args, name, dev, K, N, Ms, us, nspr, label, *, troughs=None, tile=None,
          penalty=0.0, knee=None, trend=None, resid=None, rmins=None, period=None):
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax2.set_xlabel(label)

    if args.op:
        fig.suptitle(f"{name} ({dev}) | {args.dtype} | chunked LCE op fwd+bwd "
                     f"(NT={args.num_tokens}, V={N}, D={K})")
        ax1.plot(Ms, nspr, marker=".", color="C0", label="time per row")
        ax1.plot(Ms, trend, color="C1", lw=1.2, ls="--", label="a + b/B trend")
        ax1.axvline(knee, color="green", lw=1.0, ls=":", alpha=0.7, label=f"knee ~{knee}")
        ax1.set_ylabel("time per row (ns)")
        ax1.set_title("per-row = overhead trend a + b/B + ripple", fontsize=10)
        ax1.legend(loc="upper right")
        ax2.plot(Ms, resid, marker=".", color="C0")
        ax2.axhline(0, color="gray", lw=0.5)
        if rmins:
            ax2.scatter(rmins, [resid[Ms.index(m)] for m in rmins], color="red",
                        zorder=5, label=f"residual minima (period ~{period})")
            ax2.legend(loc="upper right")
        ax2.set_ylabel("per-row - trend (ns)")
        ax2.set_title("detrended: periodic ripple reveals a chunk-size period", fontsize=10)
        note = (f"per-row fit = a + b/B (overhead); subtracted.\n"
                f"residual minima spacing -> period ~ {period}.")
    else:
        fig.suptitle(f"{name} ({dev}) | {args.dtype} | per-chunk logits GEMM "
                     f"(M, K={K}) @ (K, N={N})")
        ax1.plot(Ms, us, marker=".", color="C0")
        ax1.set_ylabel("GEMM time (us)")
        ax1.set_title("staircase: time steps once per filled M-tile", fontsize=10)
        ax2.plot(Ms, nspr, marker=".", color="C0", label="time per row")
        ax2.set_ylabel("GEMM time per row (ns)")
        ax2.set_title("sawtooth: per-row cost minimal at tile multiples", fontsize=10)
        if troughs and tile:
            ax2.scatter(troughs, [nspr[Ms.index(m)] for m in troughs], color="green",
                        zorder=5, label=f"tile-aligned M (mult. of {tile})")
            for m in range(tile, args.m_max + 1, tile):
                for ax in (ax1, ax2):
                    ax.axvline(m, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax2.legend(loc="upper right")
        note = (f"M-tile T = {tile}\nper-row ~{penalty:.0%} worse just past a boundary\n\n"
                f"auto*F floors chunk sizes DOWN to a multiple of T (green) --\n"
                f"pow2 over-rounds; a 16-tile lands mid-sawtooth")
    ax2.text(0.02, 0.95, note, transform=ax2.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))
    tag = "op" if args.op else "gemm"
    path = os.path.join(
        args.out_dir,
        f"tile_sweep_{tag}_{name.replace(' ', '_').replace('(', '').replace(')', '')}"
        f"_{dev}_{args.dtype}.png",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

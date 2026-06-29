"""Discover the GEMM threadblock M-tile that linear_cross_entropy's
``chunking_method="auto*F"`` should floor chunk sizes to (``_GPU_BLOCK_TILE``).

The per-chunk forward logits GEMM is ``input(M, K) @ weight.T -> (M, N)`` with
M = chunk size (``batch_chunk_size``), K = in_features, N = num_classes, and
the op accumulates in fp32 (``out_dtype``). If the cuBLAS kernel tiles M at
granularity T, GEMM time is a staircase stepping every T rows, so time-per-row
is a sawtooth: minimal at multiples of T, rising just past each boundary.

Sweep M in fine steps and read off the trough spacing == T. On a 2060 (fp16)
T == 64; the A100/bf16 kernel may use a larger tile (128/256) and it is
shape-dependent, so run this per target to set ``_GPU_BLOCK_TILE``. Writes an
annotated PNG (the sawtooth) for the PR motivation.

Usage:
    python tile_sweep.py                          # bf16, K=2048, N=32768
    python tile_sweep.py --dtype float16 --m-step 8
    python tile_sweep.py --in-features 4096 --num-classes 131072
"""
import argparse
import statistics

import torch

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dtype", choices=_DTYPES, default="bfloat16")
    p.add_argument("--in-features", type=int, default=2048, help="GEMM K")
    p.add_argument("--num-classes", type=int, default=32768, help="GEMM N")
    p.add_argument("--m-min", type=int, default=16)
    p.add_argument("--m-max", type=int, default=1024)
    p.add_argument("--m-step", type=int, default=16)
    p.add_argument("--reps", type=int, default=80)
    p.add_argument(
        "--out-dtype", choices=list(_DTYPES) + ["same"], default="float32",
        help="GEMM accumulate/output dtype (the op uses fp32 for bf16/fp16)",
    )
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()
    assert torch.cuda.is_available(), "CUDA required"

    dev = "cuda"
    dt = _DTYPES[args.dtype]
    odt = dt if args.out_dtype == "same" else _DTYPES[args.out_dtype]
    K, N = args.in_features, args.num_classes
    wt = torch.randn(N, K, device=dev, dtype=dt).t().contiguous()  # (K, N)

    def gemm_us(M):
        x = torch.randn(M, K, device=dev, dtype=dt)
        for _ in range(15):
            torch.mm(x, wt, out_dtype=odt)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.reps):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            torch.mm(x, wt, out_dtype=odt)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3)  # us
        return statistics.median(ts)

    Ms = list(range(args.m_min, args.m_max + 1, args.m_step))
    us = [gemm_us(M) for M in Ms]
    nspr = [u * 1e3 / M for u, M in zip(us, Ms)]

    name = torch.cuda.get_device_name()
    print(f"# {name} | {args.dtype} in / {odt} out | GEMM (M,{K}) @ ({K},{N})")
    print(f"{'M':>6}{'us':>10}{'ns/row':>10}")
    for M, u, n in zip(Ms, us, nspr):
        print(f"{M:>6}{u:>10.1f}{n:>10.1f}")

    cv = statistics.pstdev(nspr) / statistics.mean(nspr)
    troughs = [Ms[i] for i in range(1, len(nspr) - 1)
               if nspr[i] <= nspr[i - 1] and nspr[i] < nspr[i + 1]]
    tile = None
    if len(troughs) >= 2:
        sp = [b - a for a, b in zip(troughs, troughs[1:])]
        tile = int(statistics.median(sp))
    # worst per-row penalty just past a tile boundary (the peak immediately
    # after a trough vs that trough). Measured locally -- the global min/max
    # would conflate small-M launch overhead with the tile sawtooth.
    bp = [nspr[Ms.index(m) + 1] / nspr[Ms.index(m)] - 1
          for m in troughs if Ms.index(m) + 1 < len(nspr)]
    penalty = max(bp) if bp else 0.0
    print(f"\nns/row coeff-of-variation = {cv:.1%}  (smooth if small)")
    print(f"troughs (tile-aligned M): {troughs}")
    print(f"=> threadblock M-tile (_GPU_BLOCK_TILE) ~= {tile}   "
          f"worst per-row penalty ~= {penalty:.0%}")

    _plot(args, name, odt, K, N, Ms, us, nspr, troughs, tile, penalty)


def _plot(args, name, odt, K, N, Ms, us, nspr, troughs, tile, penalty):
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        f"{name} | {args.dtype} in / {odt} out | per-chunk logits GEMM "
        f"(M, K={K}) @ (K, N={N})"
    )
    ax1.plot(Ms, us, marker=".", color="C0")
    ax1.set_ylabel("GEMM time (us)")
    ax1.set_title("staircase: time steps once per filled M-tile", fontsize=10)

    ax2.plot(Ms, nspr, marker=".", color="C0", label="time per row")
    if troughs:
        ax2.scatter(
            troughs, [nspr[Ms.index(m)] for m in troughs], color="green",
            zorder=5, label=f"tile-aligned M (mult. of {tile})",
        )
    ax2.set_ylabel("GEMM time per row (ns)")
    ax2.set_xlabel("M = chunk size (batch_chunk_size)")
    ax2.set_title("sawtooth: per-row cost minimal at tile multiples", fontsize=10)
    if tile:
        for m in range(tile, args.m_max + 1, tile):
            for ax in (ax1, ax2):
                ax.axvline(m, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax2.legend(loc="upper right")
    note = (
        f"threadblock M-tile T = {tile}\n"
        f"per-row cost ~{penalty:.0%} worse just past a boundary\n\n"
        f"auto*F floors chunk sizes DOWN to a multiple of T (green) --\n"
        f"pow2 flooring over-rounds; a 16-tile lands mid-sawtooth"
    )
    ax2.text(
        0.02, 0.95, note, transform=ax2.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"),
    )
    path = os.path.join(
        args.out_dir, f"tile_sweep_{name.replace(' ', '_')}_{args.dtype}.png"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

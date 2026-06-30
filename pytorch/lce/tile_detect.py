"""Detector: bare-GEMM tile vs full LCE-op sweetspot, side by side.

Sweeps ONE chunk-size grid ``M`` and at each ``M`` measures BOTH:

* the bare per-chunk logits GEMM ``input(M, K) @ weight.T -> (M, N)``,
  fp32-accumulated -- the dominant per-chunk kernel. Its time-per-row is a
  sawtooth whose period is the threadblock-M tile (minimal at tile multiples):
  2060 fp16 -> 64, A100 bf16 -> 128 measured previously.
* the full chunked LCE op fwd+bwd over ``NT`` rows with ``batch_chunk_size=M``
  -- the real per-row curve, which falls as roughly ``a + b/M`` plus a ripple.

It then reports each curve's structure and, crucially, **whether the op tracks
the GEMM tile**: it detrends both (subtracts the ``a + b/M`` fit) and correlates
the residual ripples, and checks whether the op's per-row minima land on
multiples of the GEMM tile. This is the GPU GEMM<->op connection we want data on
across devices (2060, A100, H100, B200, MI300, ...).

Run on each device; collect the printed SUMMARY block + the PNG. The summary is
one pasteable line group per (device, dtype, shape).

    python tile_detect.py --dtype bfloat16                      # modern CUDA
    python tile_detect.py --device cpu --dtype float32
    python tile_detect.py --in-features 2048 --num-classes 32768 --num-tokens 8192
"""
import argparse
import datetime
import json
import os
import platform
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _fit_trend(Ms, nspr):
    """per_row ~ a + b/M; return (a, b, residual array)."""
    M = np.array(Ms, float)
    y = np.array(nspr, float)
    b, a = np.polyfit(1.0 / M, y, 1)
    return a, b, y - (a + b / M)


def _troughs(Ms, vals):
    """Indices of strict local minima (interior)."""
    return [i for i in range(1, len(vals) - 1)
            if vals[i] <= vals[i - 1] and vals[i] < vals[i + 1]]


def _period(Ms, idxs):
    if len(idxs) < 2:
        return None
    return int(statistics.median([Ms[b] - Ms[a] for a, b in zip(idxs, idxs[1:])]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=_DTYPES, default="bfloat16")
    p.add_argument("--in-features", type=int, default=2048, help="K")
    p.add_argument("--num-classes", type=int, default=32768, help="N")
    p.add_argument("--num-tokens", type=int, default=4096, help="NT total rows for the op")
    p.add_argument("--m-min", type=int, default=16)
    p.add_argument("--m-max", type=int, default=1024)
    p.add_argument("--m-step", type=int, default=16)
    p.add_argument("--gemm-reps", type=int, default=60)
    p.add_argument("--op-reps", type=int, default=6)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--data-dir", default="tile_detect_data",
                   help="JSON records accumulate here for cross-device analysis")
    args = p.parse_args()
    dev = args.device
    if dev == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"

    dt = _DTYPES[args.dtype]
    if dev == "cuda":  # bf16 GEMM needs sm_80+; fp16 needs sm_70+
        major, _ = torch.cuda.get_device_capability()
        if dt == torch.bfloat16 and major < 8:
            print(f"# bf16 GEMM unsupported on sm_{major}x; falling back to float16")
            dt, args.dtype = torch.float16, "float16"
        elif dt == torch.float16 and major < 7:
            print(f"# fp16 GEMM unsupported on sm_{major}x; falling back to float32")
            dt, args.dtype = torch.float32, "float32"
    odt = torch.float32  # logits accumulate in fp32
    mm_kw = {} if odt == dt else {"out_dtype": odt}  # out_dtype is CUDA-only
    K, N, NT = args.in_features, args.num_classes, args.num_tokens

    env = {"torch_version": torch.__version__}
    if dev == "cuda":
        pr = torch.cuda.get_device_properties(0)
        env.update(cuda_version=torch.version.cuda,
                   compute_capability=[pr.major, pr.minor],
                   sm_count=pr.multi_processor_count,
                   total_mem_gb=round(pr.total_memory / 1e9, 1))

    def timed(fn, warmup, reps):
        for _ in range(warmup):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e) * 1e3)
            return statistics.median(ts)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(ts)

    wt = torch.randn(N, K, device=dev, dtype=dt).t().contiguous()  # (K, N)
    x = torch.randn(NT, K, device=dev, dtype=dt, requires_grad=True)
    w = torch.randn(N, K, device=dev, dtype=dt, requires_grad=True)
    tg = torch.randint(0, N, (NT,), device=dev)

    def gemm(M):
        xm = torch.randn(M, K, device=dev, dtype=dt)
        return timed(lambda: torch.mm(xm, wt, **mm_kw), 15, args.gemm_reps)

    def op(M):
        opts = LinearCrossEntropyOptions(chunking_method=None, batch_chunk_size=M)

        def once():
            x.grad = None
            w.grad = None
            F.linear_cross_entropy(x, w, tg, reduction="mean", options=opts).backward()

        return timed(once, 3, args.op_reps)

    name = (torch.cuda.get_device_name() if dev == "cuda"
            else (platform.processor() or platform.machine() or "CPU"))
    Ms = list(range(args.m_min, args.m_max + 1, args.m_step))
    gemm_us = [gemm(m) for m in Ms]
    op_us = [op(m) for m in Ms]
    gemm_nspr = [u * 1e3 / m for u, m in zip(gemm_us, Ms)]
    op_nspr = [u * 1e3 / NT for u in op_us]

    # Structure of each curve.
    g_tro = _troughs(Ms, gemm_nspr)
    T = _period(Ms, g_tro)
    g_a, g_b, g_res = _fit_trend(Ms, gemm_nspr)
    o_a, o_b, o_res = _fit_trend(Ms, op_nspr)
    o_tro = _troughs(Ms, o_res.tolist())
    T_op = _period(Ms, o_tro)
    o_knee = next(Ms[i] for i in range(len(op_nspr))
                  if op_nspr[i] <= 1.10 * min(op_nspr))
    o_argmin = Ms[int(np.argmin(op_nspr))]

    # Connection: correlate the two detrended ripples, and test whether the op
    # is cheaper at GEMM-tile multiples than at mid-tile offsets.
    r = float(np.corrcoef(g_res, o_res)[0, 1]) if len(Ms) > 2 else float("nan")
    align = None
    if T:
        mult = {m for m in Ms if m % T == 0}
        mid = {m for m in Ms if m % T == T // 2}
        if mult and mid:
            on = statistics.mean(o_res[Ms.index(m)] for m in mult)
            off = statistics.mean(o_res[Ms.index(m)] for m in mid)
            align = off - on  # > 0 means op cheaper on tile multiples
    if T and not np.isnan(r) and (r > 0.3 or (align is not None and align > 0)):
        verdict = f"op TRACKS the GEMM tile T={T} (ripple r={r:.2f})"
    elif max(op_nspr) <= 1.15 * min(op_nspr):
        verdict = "op per-row ~flat over the grid; tile not critical"
    else:
        verdict = f"NO clean GEMM<->op link (ripple r={r:.2f}); op knee={o_knee}"

    print(f"\n# SUMMARY | {name} ({dev}) | {args.dtype} | K={K} N={N} NT={NT}")
    print(f"  GEMM: tile T={T}  troughs@{g_tro and [Ms[i] for i in g_tro][:8]}  "
          f"ripple={(max(g_res) - min(g_res)):.0f}ns")
    print(f"  op  : knee={o_knee}  argmin={o_argmin}  ripple_period={T_op}  "
          f"trend={o_a:.0f}+{o_b:.0f}/M")
    print(f"  link: ripple_corr r={r:.2f}  tile-multiple_advantage={align}  "
          f"argmin%T={o_argmin % T if T else None}")
    print(f"  => {verdict}")

    record = {
        "schema": 1,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device_name": name,
        "device_type": dev,
        "dtype": args.dtype,
        "K": K, "N": N, "NT": NT,
        "m_min": args.m_min, "m_max": args.m_max, "m_step": args.m_step,
        "gemm_reps": args.gemm_reps, "op_reps": args.op_reps,
        **env,
        # raw sweep -- kept so future analysis can re-derive metrics
        "Ms": Ms, "gemm_us": gemm_us, "gemm_nspr": gemm_nspr,
        "op_us": op_us, "op_nspr": op_nspr,
        # derived
        "gemm_tile_T": T,
        "gemm_troughs": [Ms[i] for i in g_tro],
        "gemm_ripple_ns": float(max(g_res) - min(g_res)),
        "op_knee": o_knee, "op_argmin": o_argmin, "op_ripple_period": T_op,
        "op_trend_a": float(o_a), "op_trend_b": float(o_b),
        "ripple_corr_r": r,
        "tile_multiple_advantage_ns": align,
        "argmin_mod_T": (o_argmin % T) if T else None,
        "op_tracks_gemm_tile": verdict.startswith("op TRACKS"),
        "verdict": verdict,
    }
    os.makedirs(args.data_dir, exist_ok=True)
    tag = name.replace(" ", "_").replace("(", "").replace(")", "")
    rpath = os.path.join(args.data_dir, f"{tag}_{dev}_{args.dtype}_K{K}_N{N}_NT{NT}.json")
    with open(rpath, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {rpath}")

    _plot(args, name, dev, K, N, NT, Ms, gemm_nspr, op_nspr,
          g_a, g_b, o_a, o_b, o_res, T, g_tro, o_knee, o_argmin, r, verdict)


def _plot(args, name, dev, K, N, NT, Ms, gemm_nspr, op_nspr, g_a, g_b, o_a, o_b,
          o_res, T, g_tro, o_knee, o_argmin, r, verdict):
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = np.array(Ms, float)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"{name} ({dev}) | {args.dtype} | K={K} N={N} NT={NT}\n{verdict}",
                 fontsize=11)
    ax3.set_xlabel("M = chunk size")

    def tile_lines(ax):
        if T:
            for m in range(T, args.m_max + 1, T):
                ax.axvline(m, color="gray", lw=0.5, ls=":", alpha=0.5)

    ax1.plot(Ms, gemm_nspr, marker=".", color="C0")
    if g_tro:
        ax1.scatter([Ms[i] for i in g_tro], [gemm_nspr[i] for i in g_tro],
                    color="green", zorder=5, label=f"GEMM troughs (tile T={T})")
        ax1.legend(loc="upper right")
    tile_lines(ax1)
    ax1.set_ylabel("GEMM ns/row")
    ax1.set_title("bare per-chunk logits GEMM: sawtooth, period = threadblock-M tile",
                  fontsize=10)

    ax2.plot(Ms, op_nspr, marker=".", color="C0", label="op time/row")
    ax2.plot(Ms, (o_a + o_b / M), color="C1", lw=1.2, ls="--", label="a + b/M trend")
    ax2.axvline(o_knee, color="purple", lw=1.0, ls=":", label=f"op knee ~{o_knee}")
    ax2.axvline(o_argmin, color="red", lw=1.0, ls="-", alpha=0.5, label=f"op argmin {o_argmin}")
    tile_lines(ax2)
    ax2.set_ylabel("op ns/row")
    ax2.set_title("full LCE op fwd+bwd; gray = GEMM-tile multiples (do op dips align?)",
                  fontsize=10)
    ax2.legend(loc="upper right")

    ax3.plot(Ms, o_res, marker=".", color="C0")
    ax3.axhline(0, color="gray", lw=0.5)
    tile_lines(ax3)
    ax3.set_ylabel("op per-row - trend (ns)")
    ax3.set_title(f"op detrended ripple; correlation with GEMM ripple r={r:.2f}",
                  fontsize=10)

    tag = name.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(args.out_dir, f"tile_detect_{tag}_{dev}_{args.dtype}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

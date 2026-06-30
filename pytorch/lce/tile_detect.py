"""Detector: bare-GEMM tile vs full LCE-op sweetspot, side by side.

Sweeps ONE chunk-size grid ``M`` and at each ``M`` measures BOTH:

* the bare per-chunk logits GEMM ``input(M, K) @ weight.T -> (M, N)``,
  fp32-accumulated -- the dominant per-chunk kernel. Its time-per-row is a
  sawtooth whose period is the threadblock-M tile (minimal at tile multiples):
  2060 fp16 -> 64, A100 bf16 -> 128 measured previously.
* the full chunked LCE op fwd+bwd over ``NT`` rows with ``batch_chunk_size=M``
  -- the real per-row curve, which falls as roughly ``a + b/M`` plus a ripple.

It then reports each curve's structure and, crucially, **whether the op tracks
the GEMM tile**: it detrends both (subtracts the ``a + b/M`` fit), reads the
GEMM tile from the autocorrelation of its sawtooth (flagging it irregular if the
ripple is weak), and correlates the *differenced* residuals so a shared slow
drift cannot fake a link. ``tracks`` requires a regular GEMM sawtooth that the
op co-moves with. This is the GPU GEMM<->op connection we want data on across
devices (2060, A100, H100, B200, MI300, ...).

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


def _autocorr_period(resid, m_step):
    """Dominant period of a detrended ripple via autocorrelation. Returns
    ``(period_in_M_units, strength)`` where strength is the normalized
    autocorrelation at that lag (0..1; ~>0.25 means a clean repeating ripple,
    low means noise / no single period). Robust where local-minima spacing is
    not -- spurious minima don't move the autocorrelation peak."""
    x = np.asarray(resid, float)
    n = len(x)
    if n < 8:
        return None, 0.0
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[n - 1:]
    if ac[0] == 0:
        return None, 0.0
    ac = ac / ac[0]
    hi = max(3, n // 2)
    # The period is the first LOCAL maximum past lag 0, not the global argmax:
    # near lag 0 the autocorrelation sits on the lag-0 shoulder (adjacent points
    # share the within-tile slope), so argmax would return ~1 grid step. Pick
    # the strongest interior local maximum instead.
    peaks = [k for k in range(1, hi - 1) if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        return None, 0.0
    k = max(peaks, key=lambda j: ac[j])
    return k * m_step, float(ac[k])


def analyze(Ms, gemm_nspr, op_nspr, m_step, m_max):
    """Derive all tile/connection metrics from the raw sweep. Pure function of
    the arrays so live runs and re-derivation from saved records share it."""
    M = np.array(Ms, float)
    _, _, g_res = _fit_trend(Ms, gemm_nspr)
    o_a, o_b, o_res = _fit_trend(Ms, op_nspr)

    # GEMM tile + regularity from the autocorrelation of the sawtooth ripple.
    T, g_strength = _autocorr_period(g_res, m_step)
    gemm_regular = T is not None and g_strength >= 0.25
    trough_Ms = [Ms[i] for i in _troughs(Ms, gemm_nspr)]
    spac = [b - a for a, b in zip(trough_Ms, trough_Ms[1:])]
    trough_period = int(statistics.median(spac)) if spac else None

    o_period, o_strength = _autocorr_period(o_res, m_step)
    # Correlate the SHORT-SCALE ripples only: difference the residuals first so a
    # shared slow drift (curvature the a+b/M fit leaves behind) cannot inflate r.
    # On a tiled device the per-tile sawtooth jumps co-move (high r); on a device
    # whose GEMM is flat (no tile in this M range, e.g. Blackwell) the differences
    # are just noise and r stays low -- the raw-residual correlation was fooled by
    # the shared drift there.
    if len(Ms) > 3:
        r = float(np.corrcoef(np.diff(g_res), np.diff(o_res))[0, 1])
    else:
        r = float("nan")

    # Grid-agnostic tile advantage: op residual near a tile multiple (phase in
    # [0,0.2)U(0.8,1)) vs mid-tile (phase in (0.3,0.7)). >0 => cheaper aligned.
    align = None
    if T:
        phase = (M % T) / T
        near = o_res[(phase < 0.2) | (phase > 0.8)]
        mid = o_res[(phase > 0.3) & (phase < 0.7)]
        if near.size and mid.size:
            align = float(mid.mean() - near.mean())

    o_knee = next(Ms[i] for i in range(len(op_nspr)) if op_nspr[i] <= 1.10 * min(op_nspr))
    o_argmin = Ms[int(np.argmin(op_nspr))]
    # A tile is only "tracked" if the GEMM has a regular sawtooth AND the op's
    # differenced ripple co-moves with it -- no tile, nothing to track.
    tracks = gemm_regular and not np.isnan(r) and r > 0.3
    rs = "nan" if np.isnan(r) else f"{r:.2f}"
    if tracks:
        verdict = f"op TRACKS the GEMM tile T={T} (ripple r={rs}, op period {o_period})"
    elif not gemm_regular:
        verdict = (f"no significant GEMM tile (sawtooth strength {g_strength:.2f}); "
                   f"per-row smooth, chunk size not tile-sensitive")
    elif max(op_nspr) <= 1.15 * min(op_nspr):
        verdict = "op per-row ~flat over the grid; tile not critical"
    else:
        verdict = f"GEMM tile T={T} present but op does not track it (ripple r={rs})"

    return {
        "gemm_tile_T": T,
        "gemm_tile_regular": gemm_regular,
        "gemm_ripple_strength": round(g_strength, 3),
        "gemm_trough_period": trough_period,
        "gemm_troughs": trough_Ms,
        "gemm_ripple_ns": float(max(g_res) - min(g_res)),
        "op_knee": o_knee,
        "op_argmin": o_argmin,
        "op_ripple_period": o_period,
        "op_ripple_strength": round(o_strength, 3),
        "op_trend_a": float(o_a),
        "op_trend_b": float(o_b),
        "ripple_corr_r": None if np.isnan(r) else round(r, 4),
        "tile_multiple_advantage_ns": None if align is None else round(align, 1),
        "argmin_mod_T": (o_argmin % T) if T else None,
        "op_tracks_gemm_tile": bool(tracks),
        "verdict": verdict,
    }


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

    d = analyze(Ms, gemm_nspr, op_nspr, args.m_step, args.m_max)
    print(f"\n# SUMMARY | {name} ({dev}) | {args.dtype} | K={K} N={N} NT={NT}")
    print(f"  GEMM: tile T={d['gemm_tile_T']} regular={d['gemm_tile_regular']} "
          f"(strength {d['gemm_ripple_strength']})  troughs@{d['gemm_troughs'][:8]}")
    print(f"  op  : knee={d['op_knee']} argmin={d['op_argmin']} "
          f"ripple_period={d['op_ripple_period']} (strength {d['op_ripple_strength']})  "
          f"trend={d['op_trend_a']:.0f}+{d['op_trend_b']:.0f}/M")
    print(f"  link: ripple_corr r={d['ripple_corr_r']}  "
          f"tile_advantage={d['tile_multiple_advantage_ns']}ns  "
          f"argmin%T={d['argmin_mod_T']}")
    print(f"  => {d['verdict']}")

    record = {
        "schema": 3,
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
        **d,
    }
    os.makedirs(args.data_dir, exist_ok=True)
    tag = name.replace(" ", "_").replace("(", "").replace(")", "")
    rpath = os.path.join(args.data_dir, f"{tag}_{dev}_{args.dtype}_K{K}_N{N}_NT{NT}.json")
    with open(rpath, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {rpath}")

    plot_record(record, args.out_dir)


def plot_record(rec, out_dir):
    """Render the 3-panel figure from a saved record (live run or any collected
    JSON) -- the raw sweep is stored, so figures regenerate without hardware."""
    Ms, gemm_nspr, op_nspr = rec["Ms"], rec["gemm_nspr"], rec["op_nspr"]
    o_a, o_b, o_res = _fit_trend(Ms, op_nspr)
    g_tro = _troughs(Ms, gemm_nspr)
    _plot(rec["device_name"], rec["device_type"], rec["dtype"], rec["K"], rec["N"],
          rec["NT"], rec["m_max"], out_dir, Ms, gemm_nspr, op_nspr, o_a, o_b, o_res,
          rec["gemm_tile_T"], g_tro, rec["op_knee"], rec["op_argmin"],
          rec["ripple_corr_r"], rec["verdict"])


def _plot(name, dev, dtype, K, N, NT, m_max, out_dir, Ms, gemm_nspr, op_nspr,
          o_a, o_b, o_res, T, g_tro, o_knee, o_argmin, r, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = np.array(Ms, float)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"{name} ({dev}) | {dtype} | K={K} N={N} NT={NT}\n{verdict}",
                 fontsize=11)
    ax3.set_xlabel("M = chunk size")

    def tile_lines(ax):
        if T:
            for m in range(T, m_max + 1, T):
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
    rs = "n/a" if r is None else f"{r:.2f}"
    ax3.set_title(f"op detrended ripple; correlation with GEMM ripple r={rs}",
                  fontsize=10)

    tag = name.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(out_dir, f"tile_detect_{tag}_{dev}_{dtype}_K{K}_N{N}_NT{NT}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

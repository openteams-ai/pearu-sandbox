#!/usr/bin/env python3
"""Stage 1 of the fused-kernel meta-parameter study: map `threads_per_block`
and `tiles_per_stage` at fixed geometry.

A kernel-level tuning tool, NOT a decision artefact -- a new default is adopted
only if the one sweep confirms it (plan: "Kernel meta-parameter tuning study").

Cost note that shapes the grid: `_compile_fused_grad_logits` is keyed on
(dtype pair, threads, tiles) with the chunk shape a runtime symbol, so the run
pays one compile per CONFIG, not per shape. Shapes are nearly free; configs are
not. Hence a 3x3 grid over Fable's full shape set.

`--probe` measures the two unit costs (a cold compile, a per-point timing loop)
and prints the extrapolated wall time for the full run.
"""

import argparse
import csv
import os
import statistics
import time

import torch

from torch._native.ops.linear_cross_entropy.fused_grad_logits_kernel import (
    _compile_fused_grad_logits,
    _DEFAULT_THREADS_PER_BLOCK,
    _DEFAULT_TILES_PER_STAGE,
    fused_grad_logits_into,
)


THREADS = (256, 512, 1024)
TILES = (1, 4, 8)
BCS = (256, 512, 1024, 2048, 4096)
VS = (4096, 8192, 32000, 65536)
# As shipped: the bf16 path keeps fp32 logits, the fp16 path keeps fp16 ones.
LAYOUTS = ((torch.float32, torch.bfloat16), (torch.float16, torch.float16))


def _tensors(bc, v, logits_dtype, g_dtype, dev="cuda"):
    pristine = torch.randn(bc, v, dtype=logits_dtype, device=dev)
    logits = pristine.clone()
    g = logits.view(g_dtype).narrow(1, 0, v)
    term = torch.empty(bc, dtype=torch.float32, device=dev)
    scale = torch.rand(bc, dtype=torch.float32, device=dev) + 0.5
    target = torch.randint(0, v, (bc,), dtype=torch.int64, device=dev)
    return pristine, logits, g, term, scale, target


def _reference(bc, v, logits_dtype, g_dtype):
    """Default config, separate `g`, so the logits survive for comparison."""
    pristine, logits, _, term, scale, target = _tensors(bc, v, logits_dtype, g_dtype)
    g = torch.empty(bc, v, dtype=g_dtype, device="cuda")
    fused_grad_logits_into(g, term, logits, scale, target)
    return pristine, g, term, scale, target


def _time_point(bc, v, logits_dtype, g_dtype, threads, tiles, reps, repeats, ref):
    """Median launch time per repeat, plus the worst deviation from `ref`."""
    pristine, ref_g, ref_term, scale, target = ref
    logits = pristine.clone()
    g = logits.view(g_dtype).narrow(1, 0, v)
    term = torch.empty(bc, dtype=torch.float32, device="cuda")
    meta = dict(threads_per_block=threads, tiles_per_stage=tiles)

    # The aliased layout consumes the logits, so every launch needs its input
    # back. The refill sits outside the event pair.
    logits.copy_(pristine)
    fused_grad_logits_into(g, term, logits, scale, target, **meta)
    torch.cuda.synchronize()
    g_diff = (g.float() - ref_g.float()).abs().max().item()
    term_diff = (term - ref_term).abs().max().item()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    medians = []
    for _ in range(repeats):
        ts = []
        for _ in range(reps):
            logits.copy_(pristine)
            torch.cuda.synchronize()
            start.record()
            fused_grad_logits_into(g, term, logits, scale, target, **meta)
            end.record()
            end.synchronize()
            ts.append(start.elapsed_time(end))
        medians.append(statistics.median(ts))
    return medians, g_diff, term_diff


def _run(args):
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    out = args.out or os.path.join("sweep_out", "data", gpu, "kernel_tuning.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(
            "logits_dtype g_dtype Bc V threads tiles t_ms t_ms_spread"
            " g_absdiff term_absdiff compile_s".split()
        )
    for logits_dtype, g_dtype in LAYOUTS:
        for threads in args.threads:
            for tiles in args.tiles:
                t0 = time.perf_counter()
                _compile_fused_grad_logits(logits_dtype, g_dtype, threads, tiles)
                compile_s = time.perf_counter() - t0
                for bc in BCS:
                    for v in VS:
                        ref = _reference(bc, v, logits_dtype, g_dtype)
                        med, gd, td = _time_point(
                            bc, v, logits_dtype, g_dtype, threads, tiles,
                            args.reps, args.repeats, ref,
                        )
                        w.writerow([
                            logits_dtype, g_dtype, bc, v, threads, tiles,
                            min(med), max(med) - min(med), gd, td, compile_s,
                        ])
                        fh.flush()
                        print(
                            f"{str(logits_dtype).split('.')[1]:>8}/{str(g_dtype).split('.')[1]:<8}"
                            f" Bc={bc:<5} V={v:<6} threads={threads:<5} tiles={tiles:<2}"
                            f" {min(med) * 1e3:8.1f} us  gdiff={gd:.3g}",
                            flush=True,
                        )
                        del ref
                        torch.cuda.empty_cache()
    fh.close()
    print(f"wrote {out}")


def _probe(args):
    """Measure the unit costs and extrapolate the full run."""
    logits_dtype, g_dtype = LAYOUTS[0]
    cold = (max(THREADS), max(TILES))  # least likely to be cached already
    t0 = time.perf_counter()
    _compile_fused_grad_logits(logits_dtype, g_dtype, *cold)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    _compile_fused_grad_logits(logits_dtype, g_dtype, *cold)
    t_hit = time.perf_counter() - t0
    print(f"compile threads={cold[0]} tiles={cold[1]}: first {t_first:.2f}s, in-process hit {t_hit * 1e3:.2f}ms")
    print("(run --probe twice: the second run's 'first' is the DISK-cache hit)")

    t0 = time.perf_counter()
    _compile_fused_grad_logits(
        logits_dtype, g_dtype, _DEFAULT_THREADS_PER_BLOCK, _DEFAULT_TILES_PER_STAGE
    )
    print(f"compile default config: {time.perf_counter() - t0:.2f}s")

    shapes = [(BCS[0], VS[0]), (BCS[2], VS[2]), (BCS[-1], VS[-1])]
    costs = []
    for bc, v in shapes:
        t0 = time.perf_counter()
        ref = _reference(bc, v, logits_dtype, g_dtype)
        med, gd, _ = _time_point(
            bc, v, logits_dtype, g_dtype, _DEFAULT_THREADS_PER_BLOCK,
            _DEFAULT_TILES_PER_STAGE, args.reps, args.repeats, ref,
        )
        wall = time.perf_counter() - t0
        costs.append((bc * v, wall))
        print(f"point Bc={bc:<5} V={v:<6}: wall {wall:6.2f}s, kernel {min(med) * 1e3:8.1f}us, gdiff={gd:.3g}")
        del ref
        torch.cuda.empty_cache()

    # Fit wall = a + b * elements over the smallest and largest point.
    (e0, w0), (e1, w1) = costs[0], costs[-1]
    b = (w1 - w0) / (e1 - e0)
    a = w0 - b * e0
    total_points = sum(bc * v for bc in BCS for v in VS) * len(LAYOUTS)
    n_points = len(BCS) * len(VS) * len(LAYOUTS)
    n_configs = len(THREADS) * len(TILES)
    timing = n_configs * (a * n_points + b * total_points)
    compiles = n_configs * len(LAYOUTS) * t_first
    print(
        f"\nestimate for the full {len(THREADS)}x{len(TILES)} grid:"
        f"\n  {n_configs * len(LAYOUTS)} compiles x {t_first:.1f}s = {compiles / 60:.1f} min"
        f"\n  {n_configs * n_points} timed points = {timing / 60:.1f} min"
        f"\n  total {(compiles + timing) / 60:.1f} min + startup"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true", help="measure unit costs and extrapolate")
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, nargs="+", default=list(THREADS))
    p.add_argument("--tiles", type=int, nargs="+", default=list(TILES))
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    (_probe if args.probe else _run)(args)

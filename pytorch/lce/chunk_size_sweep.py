"""Independent chunk-size sweep: find the throughput-optimal batch_chunk_size
in the budget regime (in_features >= num_classes), decoupled from the budget
heuristic and the ghstack.

Motivation: aspect_ratio is a throughput heuristic (from liger) proven only in
its validated regime (num_classes >> in_features). Above the crossing we are
out of that range with no proven heuristic. This sweep determines the chunk
size by the SAME criterion aspect_ratio optimizes -- throughput -- measured
empirically, accepting the upfront benchmarking cost.

It sweeps EXPLICIT batch_chunk_size (chunking_method=None, the landed #187219
API), so it runs on any build with the chunked op and never references the
budget code. For each (num_tokens, in_features, num_classes) point above the
crossing it sweeps B over powers of two, measuring fwd+bwd median time and peak
memory in an isolated subprocess, plus a full-materialization reference row.

The analysis extracts B_knee = the smallest B on the throughput plateau (within
--tol of the best time): the throughput-optimal chunk size. It then CHECKS (does
not enforce) that B_knee's peak memory stays at or below the reference, so the
resulting heuristic restricts memory growth over reference without being
parameterized by it.

Usage:
    python chunk_size_sweep.py --out sweep_chunk.csv               # measure
    python chunk_size_sweep.py --out sweep_chunk.csv --analyze     # knee + plots
    python chunk_size_sweep.py --out sweep_chunk.csv --smoke       # tiny local set
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch

from lce_benchmark import (
    _CUDAPeakMonitor,
    _chunked_forward_backward,
    _reference_forward_backward,
)
from lce_benchmark_sweep import _apply_log_xticks, _local_device_slug

_THIS_DIR = Path(__file__).resolve().parent

# Budget regime: in_features >= num_classes (above the crossing). The grid
# spans 1x..4x crossing so the knee can be tracked as D pulls away from V.
DEFAULT_GRID = {
    "num_tokens": [4096, 8192, 16384],
    "in_features": [16384, 32768, 65536],
    "num_classes": [8192, 16384, 32000],
}
DEFAULT_FIXED = {"num_tokens": 8192, "in_features": 32768, "num_classes": 16384}

SMOKE_GRID = {
    "num_tokens": [512, 1024],
    "in_features": [2048, 4096],
    "num_classes": [1024, 2048],
}
SMOKE_FIXED = {"num_tokens": 512, "in_features": 4096, "num_classes": 1024}

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _chunk_sizes(num_tokens: int) -> list[int]:
    """Powers of two from 64 up to and including num_tokens (single chunk)."""
    sizes = []
    b = 64
    while b < num_tokens:
        sizes.append(b)
        b *= 2
    sizes.append(num_tokens)
    return sizes


# ---------------------------------------------------------------------------
# Worker: one (shape, B) point in an isolated subprocess
# ---------------------------------------------------------------------------

def _make_inputs(N: int, D: int, V: int, dtype, device):
    input = torch.randn(N, D, dtype=dtype, device=device, requires_grad=True)
    linear_weight = (
        torch.randn(V, D, dtype=dtype, device=device) * (1.0 / (D ** 0.5))
    ).detach().requires_grad_(True)
    target = torch.randint(0, V, (N,), device=device)
    return input, linear_weight, target


def _make_call(payload: dict, input, linear_weight, target):
    reduction = payload["reduction"]
    if payload["mode"] == "reference":
        return lambda: _reference_forward_backward(input, linear_weight, target, reduction)
    from torch.nn import LinearCrossEntropyOptions

    # chunking_method=None disables the heuristic and uses our explicit B,
    # so this never exercises the budget/auto resolution code.
    options = LinearCrossEntropyOptions(
        acc_policy=payload["acc_policy"],
        chunking_method=None,
        batch_chunk_size=payload["batch_chunk_size"],
        acc_dtype=torch.float32,
    )
    return lambda: _chunked_forward_backward(input, linear_weight, target, reduction, options)


def _worker(payload: dict) -> None:
    torch.manual_seed(payload["seed"])
    dtype = _DTYPE_MAP[payload["dtype"]]
    device_type = payload["device_type"]
    device = torch.device("cuda") if device_type == "cuda" else torch.device("cpu")
    N, D, V = payload["num_tokens"], payload["in_features"], payload["num_classes"]

    input, linear_weight, target = _make_inputs(N, D, V, dtype, device)
    call = _make_call(payload, input, linear_weight, target)

    def _clear():
        for t in (input, linear_weight):
            t.grad = None

    for _ in range(payload["warmup"]):
        _clear()
        call()
        if device_type == "cuda":
            torch.cuda.synchronize()

    memory_peak_bytes = 0
    times_ms: list[float] = []
    err: Optional[str] = None
    try:
        if device_type == "cuda":
            _clear()
            with _CUDAPeakMonitor() as mem:
                call()
            memory_peak_bytes = mem.peak_bytes
            events = []
            for _ in range(payload["iters"]):
                _clear()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                call()
                e.record()
                events.append((s, e))
            torch.cuda.synchronize()
            times_ms = [s.elapsed_time(e) for s, e in events]
        else:
            for _ in range(payload["iters"]):
                _clear()
                t0 = time.perf_counter()
                call()
                times_ms.append(1000.0 * (time.perf_counter() - t0))
    except torch.OutOfMemoryError as e:
        err = f"OOM: {e}"
        if device_type == "cuda":
            torch.cuda.empty_cache()

    times_ms.sort()
    median = times_ms[len(times_ms) // 2] if times_ms else float("nan")
    out = {
        "num_tokens": N,
        "in_features": D,
        "num_classes": V,
        "dtype": payload["dtype"],
        "mode": payload["mode"],
        "batch_chunk_size": payload["batch_chunk_size"],
        "acc_policy": payload["acc_policy"],
        "time_ms": median,
        "memory_peak_bytes": memory_peak_bytes if memory_peak_bytes else float("nan"),
    }
    if err:
        out["error"] = err
    print(json.dumps(out))


def _run_point(payload: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", json.dumps(payload)],
        capture_output=True,
        text=True,
    )
    ident = {k: payload[k] for k in (
        "num_tokens", "in_features", "num_classes", "dtype", "mode",
        "batch_chunk_size", "acc_policy",
    )}
    if proc.returncode != 0:
        return {**ident, "error": (proc.stderr or proc.stdout).strip()[-400:] or "failed"}
    try:
        return {**ident, **json.loads(proc.stdout.strip().splitlines()[-1])}
    except Exception as e:
        return {**ident, "error": f"parse failed: {e!r}"}


# ---------------------------------------------------------------------------
# Profile: per-aten-op CUDA self-time, bucketed (GEMM vs softmax/elementwise)
# ---------------------------------------------------------------------------

def _bucket(name: str) -> str:
    low = name.lower()
    if any(g in low for g in ("::mm", "::addmm", "::bmm")):
        return "GEMM"
    if "index" in low or "gather" in low or "scatter" in low:
        return "gather/scatter"
    return "elementwise/reduction"


def _profile_worker(payload: dict) -> None:
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(payload["seed"])
    dtype = _DTYPE_MAP[payload["dtype"]]
    device = torch.device("cuda")
    N, D, V = payload["num_tokens"], payload["in_features"], payload["num_classes"]
    input, linear_weight, target = _make_inputs(N, D, V, dtype, device)
    call = _make_call(payload, input, linear_weight, target)

    def _clear():
        for t in (input, linear_weight):
            t.grad = None

    for _ in range(payload["warmup"]):
        _clear()
        call()
        torch.cuda.synchronize()

    _clear()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(payload["iters"]):
            _clear()
            call()
        torch.cuda.synchronize()

    def dev_us(e) -> float:
        # microseconds; newer torch renames self_cuda_time_total -> self_device_time_total
        return float(getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0) or 0)

    buckets: dict[str, float] = defaultdict(float)
    rows = []
    total = 0.0
    for e in prof.key_averages():
        # Sum CUDA self-time over aten ops only: 'self' excludes children, so
        # summing over the aten layer is the total GPU time with no double count
        # (kernel-name rows would double-count the same time).
        if not e.key.startswith("aten::"):
            continue
        t = dev_us(e)
        if t <= 0:
            continue
        b = _bucket(e.key)
        buckets[b] += t
        total += t
        rows.append((t, e.key, b))

    print(f"shape N{N} D{D} V{V} {payload['dtype']} B={payload['batch_chunk_size']} "
          f"policy={payload['acc_policy']} mode={payload['mode']}  ({payload['iters']} iters)")
    if total <= 0:
        print("no CUDA self-time captured (is this a CUDA build?)")
        return
    print(f"total CUDA self-time over aten ops: {total / 1000:.2f} ms\n")
    print("bucket breakdown:")
    for b, t in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {b:>24}: {t / 1000:>9.3f} ms  {100 * t / total:>5.1f}%")
    print("\ntop aten ops by CUDA self-time:")
    for t, name, b in sorted(rows, reverse=True)[:12]:
        print(f"  {name:>26} [{b:>12}]: {t / 1000:>9.3f} ms  {100 * t / total:>5.1f}%")


def run_profile(args) -> int:
    if not torch.cuda.is_available():
        print("profile mode needs CUDA", file=sys.stderr)
        return 1
    N, D, V = args.profile_shape
    payload = {
        "num_tokens": N, "in_features": D, "num_classes": V,
        "dtype": args.dtypes[0], "device_type": "cuda", "reduction": args.reduction,
        "acc_policy": args.acc_policy, "warmup": args.warmup, "iters": args.iters,
        "seed": args.seed, "mode": "chunked", "batch_chunk_size": args.profile_b,
    }
    proc = subprocess.run(
        [sys.executable, __file__, "--profile-worker", json.dumps(payload)],
        text=True,
    )
    return proc.returncode


# ---------------------------------------------------------------------------
# Measure driver
# ---------------------------------------------------------------------------

_KEYS = [
    "device", "num_tokens", "in_features", "num_classes", "dtype", "mode",
    "batch_chunk_size", "acc_policy", "time_ms", "memory_peak_bytes", "error",
]

# Best-effort SM counts for the per-SM portability check; correct/extend as
# needed (matched by substring against the device slug, uppercased).
SM_COUNTS = {
    "A100": 108,
    "H100": 132,
    "H200": 132,
    "B200": 148,
    "B300": 148,
}


def _sm_count(device: str) -> Optional[int]:
    up = device.upper()
    for key, sm in SM_COUNTS.items():
        if key in up:
            return sm
    return None


def _points(grid: dict, fixed: dict) -> list[dict]:
    """One swept axis at a time at the fixed defaults of the other two."""
    seen = set()
    pts = []
    for axis, values in grid.items():
        for v in values:
            p = dict(fixed)
            p[axis] = v
            key = (p["num_tokens"], p["in_features"], p["num_classes"])
            if key in seen:
                continue
            seen.add(key)
            pts.append(p)
    return pts


def measure(args) -> int:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.device_tag or _local_device_slug()[1]
    grid = SMOKE_GRID if args.smoke else DEFAULT_GRID
    fixed = SMOKE_FIXED if args.smoke else DEFAULT_FIXED
    if args.num_tokens:
        grid["num_tokens"] = args.num_tokens
    if args.in_features:
        grid["in_features"] = args.in_features
    if args.num_classes:
        grid["num_classes"] = args.num_classes

    out = Path(args.out)
    existing = set()
    if out.exists() and not args.force:
        with open(out) as f:
            for r in csv.DictReader(f):
                existing.add((
                    r.get("device", ""),
                    int(r["num_tokens"]), int(r["in_features"]), int(r["num_classes"]),
                    r["dtype"], r["mode"], r["batch_chunk_size"],
                ))

    write_header = not out.exists() or args.force
    mode = "w" if args.force else "a"
    with open(out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=_KEYS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for dtype in args.dtypes:
            for p in _points(grid, fixed):
                N = p["num_tokens"]
                jobs = [{"mode": "reference", "batch_chunk_size": 0}]
                jobs += [{"mode": "chunked", "batch_chunk_size": b} for b in _chunk_sizes(N)]
                for job in jobs:
                    payload = {
                        **p, "dtype": dtype, "device_type": device_type,
                        "reduction": args.reduction, "acc_policy": args.acc_policy,
                        "warmup": args.warmup, "iters": args.iters, "seed": args.seed,
                        **job,
                    }
                    key = (tag, N, p["in_features"], p["num_classes"], dtype,
                           job["mode"], str(job["batch_chunk_size"]))
                    if key in existing:
                        continue
                    row = _run_point(payload)
                    row["device"] = tag
                    w.writerow(row)
                    f.flush()
                    blabel = "ref" if job["mode"] == "reference" else f"B={job['batch_chunk_size']}"
                    t = row.get("time_ms", float("nan"))
                    print(f"  N{N} D{p['in_features']} V{p['num_classes']} {dtype} {blabel}: "
                          f"{t:.2f} ms" + (f"  [{row['error'][:60]}]" if row.get("error") else ""))
    return 0


# ---------------------------------------------------------------------------
# Analyze: throughput knee + memory-vs-reference check + plots
# ---------------------------------------------------------------------------

def _load(out: Path) -> list[dict]:
    rows = []
    with open(out) as f:
        for r in csv.DictReader(f):
            r["device"] = r.get("device") or "unknown"
            for k in ("num_tokens", "in_features", "num_classes"):
                r[k] = int(r[k])
            r["batch_chunk_size"] = int(r["batch_chunk_size"])
            for k in ("time_ms", "memory_peak_bytes"):
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = float("nan")
            rows.append(r)
    return rows


def _knee(chunked: list[dict], tol: float) -> Optional[dict]:
    """Smallest-B row whose time is within (1+tol) of the best time."""
    valid = [r for r in chunked if r["time_ms"] == r["time_ms"]]
    if not valid:
        return None
    best = min(r["time_ms"] for r in valid)
    onplateau = [r for r in valid if r["time_ms"] <= best * (1 + tol)]
    return min(onplateau, key=lambda r: r["batch_chunk_size"])


def _fit_heuristics(summary: list[dict]) -> None:
    """Fit B_knee to candidate closed forms per dtype and rank by misfit.

    Each model is B_knee = c * f(N,D,V); c is the geometric mean of the ratio
    B_knee/f (least squares in log space, the natural metric for a B sampled on
    a power-of-two grid). The headline metric is log2 RMSE -- the RMS number of
    octaves the model misses by -- since a real heuristic would snap to a
    power of two anyway.
    """
    models = [
        ("const", lambda s: 1.0),
        ("N", lambda s: s["N"]),
        ("V", lambda s: s["V"]),
        ("D", lambda s: s["D"]),
        ("N*V/D", lambda s: s["N"] * s["V"] / s["D"]),
    ]
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for s in summary:
        by_group[(s["dtype"], s["device"])].append(s)
    const_by_device: dict[str, float] = {}
    for (dt, dev), pts in sorted(by_group.items()):
        if len(pts) < 2:
            continue
        print(f"\nHeuristic fit (dtype={dt}, device={dev}, {len(pts)} shapes), "
              "B_knee = c * f(N,D,V):")
        print(f"  {'model':>10} {'c':>12} {'log2 RMSE':>10} {'rel RMSE':>9}  (lower is better)")
        results = []
        for name, f in models:
            logc = sum(math.log(s["B_knee"] / f(s)) for s in pts) / len(pts)
            c = math.exp(logc)
            log2err = [math.log2(c * f(s) / s["B_knee"]) for s in pts]
            l2 = (sum(e * e for e in log2err) / len(log2err)) ** 0.5
            rel = [(c * f(s) - s["B_knee"]) / s["B_knee"] for s in pts]
            relrmse = (sum(e * e for e in rel) / len(rel)) ** 0.5
            results.append((l2, name, c, relrmse))
        results.sort()
        for l2, name, c, relrmse in results:
            print(f"  {name:>10} {c:>12.4g} {l2:>10.3f} {relrmse:>9.3f}")
            # Always record the constant model so the per-SM portability check
            # below can run, even when 'const' is not this device's top fit.
            if name == "const":
                const_by_device[dev] = c
        l2, name, c, _ = results[0]
        print(f"  -> best: B_knee ~= {c:.4g} * {name}  "
              f"(log2 RMSE {l2:.3f} = within ~{2 ** l2:.2f}x)")
        print("  log2 RMSE below ~0.5 => the model lands within one power-of-two "
              "sweep step of the measured knee.")

    # Cross-device portability: is the constant invariant, or invariant per SM?
    if len(const_by_device) >= 2:
        print("\nCross-device constant (is the saturation B portable?):")
        print(f"  {'device':>28} {'const B':>9} {'SM':>5} {'B/SM':>8}")
        for dev, c in sorted(const_by_device.items()):
            sm = _sm_count(dev)
            bsm = f"{c / sm:>8.2f}" if sm else f"{'?':>8}"
            print(f"  {dev:>28} {c:>9.4g} {(sm if sm else '?'):>5} {bsm}")
        print("  If 'const B' differs across devices but 'B/SM' is ~flat, the "
              "portable heuristic is B = (B/SM) * SM_count.")


def analyze(args) -> int:
    paths = [Path(p) for p in (args.inputs or [args.out])]
    rows: list[dict] = []
    for p in paths:
        rows += _load(p)
    by_shape: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        shape = (r["num_tokens"], r["in_features"], r["num_classes"], r["dtype"], r["device"])
        if r["mode"] == "reference":
            by_shape[shape]["ref"] = r
        else:
            by_shape[shape].setdefault("chunked", []).append(r)

    gib = 1024 ** 3
    print(f"\nThroughput knee (within {args.tol:.0%} of best time), per shape:")
    print(f"{'N':>6} {'D':>7} {'V':>7} {'dt':>4} {'device':>22} | {'B_knee':>7} {'B/N':>6} "
          f"{'t_knee':>8} {'t_best':>8} | {'mem_knee':>9} {'mem_ref':>9} {'<=ref?':>7}")
    summary = []
    for shape in sorted(by_shape):
        N, D, V, dt, dev = shape
        rec = by_shape[shape]
        chunked = rec.get("chunked", [])
        knee = _knee(chunked, args.tol)
        if knee is None:
            continue
        valid = [r["time_ms"] for r in chunked if r["time_ms"] == r["time_ms"]]
        t_best = min(valid)
        ref = rec.get("ref", {})
        mem_knee = knee["memory_peak_bytes"] / gib
        mem_ref = ref.get("memory_peak_bytes", float("nan")) / gib
        ok = "yes" if mem_knee <= mem_ref else "NO"
        print(f"{N:>6} {D:>7} {V:>7} {dt:>4} {dev[:22]:>22} | {knee['batch_chunk_size']:>7} "
              f"{knee['batch_chunk_size']/N:>6.3f} {knee['time_ms']:>8.2f} {t_best:>8.2f} | "
              f"{mem_knee:>9.3f} {mem_ref:>9.3f} {ok:>7}")
        summary.append({
            "N": N, "D": D, "V": V, "dtype": dt, "device": dev,
            "B_knee": knee["batch_chunk_size"], "mem_knee": mem_knee, "mem_ref": mem_ref,
        })

    if summary:
        knees = [s["B_knee"] for s in summary]
        print(f"\nB_knee: min={min(knees)} max={max(knees)} "
              f"median={sorted(knees)[len(knees)//2]}")
        print("If B_knee is ~flat across shapes, a near-constant chunk size is the "
              "throughput heuristic; a clear slope means it scales with that axis.")
        _fit_heuristics(summary)

    if not args.no_plot:
        _plot(by_shape, paths[0], args.tol)
    return 0


def _plot(by_shape: dict, out: Path, tol: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}", file=sys.stderr)
        return

    gib = 1024 ** 3
    shapes = sorted(by_shape)
    ncols = min(len(shapes), 4)
    nrows = math.ceil(len(shapes) / ncols)
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=(4.5 * ncols, 5 * nrows), squeeze=False)
    for i, shape in enumerate(shapes):
        N, D, V, dt, dev = shape
        rec = by_shape[shape]
        chunked = sorted(rec.get("chunked", []), key=lambda r: r["batch_chunk_size"])
        if not chunked:
            continue
        bs = [r["batch_chunk_size"] for r in chunked]
        ts = [r["time_ms"] for r in chunked]
        ms = [r["memory_peak_bytes"] / gib for r in chunked]
        knee = _knee(chunked, tol)
        ref = rec.get("ref", {})
        mem_ref = ref.get("memory_peak_bytes", float("nan")) / gib

        rax = (i // ncols) * 2
        cax = i % ncols
        ax_t = axes[rax][cax]
        ax_m = axes[rax + 1][cax]
        ax_t.plot(bs, ts, marker="o")
        if knee is not None:
            ax_t.axvline(knee["batch_chunk_size"], color="green", ls="--", lw=1, label="knee")
            ax_m.axvline(knee["batch_chunk_size"], color="green", ls="--", lw=1)
        ax_t.set_title(f"N{N} D{D} V{V} {dt}\n{dev[:24]}", fontsize=7)
        ax_t.set_ylabel("time (ms)")
        ax_t.set_xscale("log")
        _apply_log_xticks(ax_t, bs)
        ax_t.legend(fontsize=6)
        ax_m.plot(bs, ms, marker="o", color="tab:red")
        if mem_ref == mem_ref:
            ax_m.axhline(mem_ref, color="black", ls=":", lw=1, label="reference")
        ax_m.set_ylabel("peak mem (GiB)")
        ax_m.set_xlabel("batch_chunk_size")
        ax_m.set_xscale("log")
        _apply_log_xticks(ax_m, bs)
        ax_m.legend(fontsize=6)

    fig.suptitle(f"chunk-size sweep: throughput knee vs memory (reference dotted) "
                 f"[plateau tol={tol:.0%}]")
    fig.tight_layout()
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=120)
    print(f"\nwrote {png}")


# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", default=None, help="(internal)")
    p.add_argument("--profile-worker", default=None, help="(internal)")
    p.add_argument("--profile", action="store_true",
                   help="profile one fwd+bwd and bucket CUDA self-time by aten op")
    p.add_argument("--profile-shape", nargs=3, type=int, default=[8192, 32768, 16384],
                   metavar=("N", "D", "V"), help="shape to profile (budget regime)")
    p.add_argument("--profile-b", type=int, default=2048, help="batch_chunk_size to profile")
    p.add_argument("--out", default=str(_THIS_DIR / "chunk_size_sweep.csv"))
    p.add_argument("--device-tag", default=None,
                   help="stamp this label into the 'device' column; defaults to the "
                   "auto-detected GPU slug. Lets multiple GPUs share/merge a CSV.")
    p.add_argument("--inputs", nargs="+", default=None,
                   help="(analyze) one or more CSVs to merge and fit together; "
                   "defaults to --out")
    p.add_argument("--dtypes", nargs="+", default=["bfloat16"])
    p.add_argument("--acc-policy", default="compact",
                   help="budget-regime CUDA auto pick is compact; sweep that path")
    p.add_argument("--reduction", default="mean")
    p.add_argument("--num-tokens", nargs="+", type=int, default=None)
    p.add_argument("--in-features", nargs="+", type=int, default=None)
    p.add_argument("--num-classes", nargs="+", type=int, default=None)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=0.03, help="throughput-plateau tolerance")
    p.add_argument("--smoke", action="store_true", help="tiny grid for local validation")
    p.add_argument("--force", action="store_true", help="overwrite CSV instead of resuming")
    p.add_argument("--analyze", action="store_true", help="knee + plots from existing CSV")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse()
    if args.worker is not None:
        _worker(json.loads(args.worker))
        return 0
    if args.profile_worker is not None:
        _profile_worker(json.loads(args.profile_worker))
        return 0
    if args.profile:
        return run_profile(args)
    if args.analyze:
        return analyze(args)
    return measure(args)


if __name__ == "__main__":
    sys.exit(main())

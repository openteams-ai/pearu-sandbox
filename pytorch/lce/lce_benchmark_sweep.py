"""Sweep driver around lce_benchmark.py.

Sweeps one axis at a time (num_tokens, in_features, num_classes) at the
defaults of the other two, per-device + per-dtype, writing one CSV per
(point, dtype, device, reduction, retain-graph) tuple. Then plots a 4-row
figure per (dtype, device_name) grouping: time_ms, memory_peak_gb,
grad_linear_weight_error, grad_input_error.

Defaults are device-specific (CUDA sweeps are much larger than CPU sweeps).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import torch


_THIS_DIR = Path(__file__).resolve().parent
_INNER = _THIS_DIR / "lce_benchmark.py"


# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------

AXIS_NAMES = ("num_tokens", "in_features", "num_classes")

# CUDA: full sweep
CUDA_AXES: dict[str, list[int]] = {
    "num_tokens": [1024, 2048, 4096, 8192, 16384],
    "in_features": [1024, 2048, 4096, 8192],
    "num_classes": [4096, 8192, 16384, 32000, 65536],
}
CUDA_DEFAULTS: dict[str, int] = {
    "num_tokens": 4096,
    "in_features": 4096,
    "num_classes": 32000,
}

# CPU: smaller sweep to keep wall-time manageable
CPU_AXES: dict[str, list[int]] = {
    "num_tokens": [128, 256, 512, 1024],
    "in_features": [256, 512, 1024, 2048],
    "num_classes": [1024, 2048, 4096, 8192],
}
CPU_DEFAULTS: dict[str, int] = {
    "num_tokens": 512,
    "in_features": 1024,
    "num_classes": 4096,
}


# ---------------------------------------------------------------------------
# Local device slug
# ---------------------------------------------------------------------------

def _local_device_slug() -> tuple[str, str, str]:
    """Return (device_name, slug, device_type)."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        return name, slug, "cuda"

    # CPU
    name = "cpu"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return name, slug, "cpu"


# ---------------------------------------------------------------------------
# CSV path
# ---------------------------------------------------------------------------

def _csv_path(
    out_dir: Path,
    point: dict,
    dtype: str,
    device_slug: str,
    allow_retain_graph: bool = False,
    reduction: str = "mean",
) -> Path:
    parts = [
        f"N{point['num_tokens']}",
        f"D{point['in_features']}",
        f"V{point['num_classes']}",
        dtype,
    ]
    if reduction != "mean":
        parts.append(f"reduction-{reduction}")
    if allow_retain_graph:
        parts.append("retain-graph")
    stem = "_".join(parts)
    return out_dir / device_slug / f"{stem}.csv"


# ---------------------------------------------------------------------------
# Run a single inner benchmark and write its CSV
# ---------------------------------------------------------------------------

def _run_point(
    *,
    point: dict,
    dtype: str,
    device_type: str,
    device_slug: str,
    reduction: str,
    allow_retain_graph: bool,
    include_acc_none: bool,
    out_dir: Path,
    warmup: int,
    iters: int,
    force: bool,
) -> Path:
    csv_path = _csv_path(out_dir, point, dtype, device_slug, allow_retain_graph, reduction)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and not force:
        return csv_path
    if csv_path.exists():
        csv_path.unlink()

    cmd = [
        sys.executable,
        str(_INNER),
        "--num-tokens", str(point["num_tokens"]),
        "--in-features", str(point["in_features"]),
        "--num-classes", str(point["num_classes"]),
        "--dtype", dtype,
        "--device", device_type,
        "--reduction", reduction,
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--out", str(csv_path),
    ]
    if allow_retain_graph:
        cmd.append("--allow-retain-graph")
    if include_acc_none:
        cmd.append("--include-acc-none")
    subprocess.run(cmd, check=False)
    return csv_path


# ---------------------------------------------------------------------------
# Load + plot
# ---------------------------------------------------------------------------

def _format_axis_tick(v: float) -> str:
    """K-suffix integer formatting for the log x-axis ticks."""
    iv = int(round(v))
    if iv >= 1024 and iv % 1024 == 0:
        return f"{iv // 1024}K"
    if iv >= 1000:
        return f"{iv / 1024:.1f}K"
    return str(iv)


def _apply_log_xticks(ax, xvals) -> None:
    """Pin major ticks to the swept x-values; suppress log-decade minor ticks."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    xs = sorted(set(xvals))
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _format_axis_tick(v)))

def _load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            # Skip rows missing axis identifiers (legacy error-only rows).
            if not (r.get("num_tokens") and r.get("in_features") and r.get("num_classes")):
                continue
            for k in ("num_tokens", "in_features", "num_classes"):
                r[k] = int(r[k])
            for k in ("time_ms", "memory_peak_bytes", "grad_input_error", "grad_linear_weight_error"):
                v = r.get(k)
                if v in (None, ""):
                    r[k] = float("nan")
                else:
                    try:
                        r[k] = float(v)
                    except Exception:
                        r[k] = float("nan")
            mb = r["memory_peak_bytes"]
            r["memory_peak_gb"] = mb / (1024 ** 3) if mb == mb else float("nan")
            rows.append(r)
    return rows


YFIELDS = [
    ("time_ms", "time (ms)", False),
    ("memory_peak_gb", "peak memory (GiB)", False),
    ("grad_linear_weight_error", "grad weight rel err", True),
    ("grad_input_error", "grad input rel err", True),
]


def _row_is_fp32_or_baseline(r: dict) -> bool:
    """True iff this row should be shown when --include-acc-none is off."""
    return r.get("label") in ("reference", "liger") or r.get("acc_dtype") == "float32"


def _plot(
    out_dir: Path,
    csv_paths: list[Path],
    group_label: str,
    *,
    include_acc_none: bool = False,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}", file=sys.stderr)
        return

    all_rows: list[dict] = []
    for p in csv_paths:
        all_rows.extend(_load_rows(p))
    if not all_rows:
        print(f"no data for {group_label}", file=sys.stderr)
        return

    if not include_acc_none:
        all_rows = [r for r in all_rows if _row_is_fp32_or_baseline(r)]
        if not all_rows:
            print(f"no fp32-acc rows for {group_label}; pass --include-acc-none", file=sys.stderr)
            return

    # Infer per-axis defaults (mode across non-swept axes)
    counts: dict[str, Counter] = {a: Counter() for a in AXIS_NAMES}
    for r in all_rows:
        for a in AXIS_NAMES:
            counts[a][r[a]] += 1
    defaults = {a: counts[a].most_common(1)[0][0] for a in AXIS_NAMES}

    # sharex='col' ties x-axis (scale, limits, ticks) across rows in the
    # same column, so a subplot with missing y-data still aligns with its
    # peers above/below.
    fig, axes = plt.subplots(
        len(YFIELDS), len(AXIS_NAMES), figsize=(15, 12), sharex="col"
    )
    if len(YFIELDS) == 1:
        axes = [axes]

    for col, axis in enumerate(AXIS_NAMES):
        rows = [
            r
            for r in all_rows
            if all(r[a] == defaults[a] for a in AXIS_NAMES if a != axis)
        ]
        series: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            series[r["label"]].append(r)
        for label, xs in series.items():
            xs.sort(key=lambda r: r[axis])
        # Collect column-wide x-values so every row in this column gets
        # the same ticks/limits (sharex='col' propagates the locator).
        col_xvals = sorted({r[axis] for r in rows})
        for row_idx, (yfield, ylabel, log) in enumerate(YFIELDS):
            ax = axes[row_idx][col]
            any_positive = False
            for label, xs in sorted(series.items()):
                ys = [x.get(yfield, float("nan")) for x in xs]
                xvals = [x[axis] for x in xs]
                # Pin liger to black + star marker so it's visually
                # distinct from the auto-cycled colors / round markers
                # used for our chunked configs.
                if label == "liger":
                    style = {"color": "black", "marker": "*", "markersize": 9}
                else:
                    style = {"marker": "o"}
                ax.plot(xvals, ys, label=label, linewidth=1, **style)
                if any(isinstance(y, (int, float)) and y == y and y > 0 for y in ys):
                    any_positive = True
            ax.set_xlabel(axis)
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            _apply_log_xticks(ax, col_xvals)
            # sharex='col' hides tick numerals on non-bottom rows by default;
            # re-enable so each row carries its own axis numbers.
            ax.tick_params(labelbottom=True)
            # Skip log y-scale when no positive data exists in this subplot
            # (e.g. all grad-error fields are NaN because the fp64 reference
            # budget-check skipped on VRAM-constrained cards).
            if log and any_positive:
                ax.set_yscale("log")
            if row_idx == 0 and col == 0:
                ax.legend(fontsize=6, loc="best")

    fig.suptitle(f"linear_cross_entropy benchmark sweep: {group_label}")
    fig.tight_layout()
    png = out_dir / f"sweep_{group_label.replace(' ', '_').replace('/', '_')}.png"
    fig.savefig(png, dpi=120)
    print(f"wrote {png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--dtypes", nargs="+", default=["float16", "bfloat16"])
    p.add_argument("--reduction", default="mean")
    p.add_argument("--allow-retain-graph", action="store_true")
    p.add_argument(
        "--include-acc-none",
        action="store_true",
        help="for fp16/bf16, also include configs with acc_dtype=None. By "
        "default only the _fp32 (mixed-precision) variants are emitted.",
    )
    p.add_argument("--out-dir", default=str(_THIS_DIR / "sweep_out"))
    p.add_argument("--data-subdir", default="data")
    p.add_argument("--axes", nargs="+", default=list(AXIS_NAMES))
    p.add_argument(
        "--num-tokens",
        nargs="+",
        type=int,
        default=None,
        help="override the swept values for num_tokens (defaults to "
        "CUDA_AXES / CPU_AXES depending on --device)",
    )
    p.add_argument(
        "--in-features",
        nargs="+",
        type=int,
        default=None,
        help="override the swept values for in_features",
    )
    p.add_argument(
        "--num-classes",
        nargs="+",
        type=int,
        default=None,
        help="override the swept values for num_classes",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--force", action="store_true", help="re-run even if CSV exists")
    p.add_argument("--plot-only", action="store_true", help="skip benchmarking, just plot")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    data_dir = out_dir / args.data_subdir
    data_dir.mkdir(parents=True, exist_ok=True)

    device_name, device_slug, device_type_local = _local_device_slug()
    if args.device == "auto":
        device_type = device_type_local
    else:
        device_type = args.device

    if device_type == "cuda":
        axes_table = dict(CUDA_AXES)
        defaults = dict(CUDA_DEFAULTS)
    else:
        axes_table = dict(CPU_AXES)
        defaults = dict(CPU_DEFAULTS)

    # CLI overrides update both the swept axis values AND the default
    # used when that axis is NOT the swept one (otherwise non-swept axes
    # would still pull from CUDA_DEFAULTS / CPU_DEFAULTS even when the
    # user expected a uniform shift).
    for key, override in (
        ("num_tokens", args.num_tokens),
        ("in_features", args.in_features),
        ("num_classes", args.num_classes),
    ):
        if override is not None:
            axes_table[key] = override
            # Preserve the original default if it's still in the override
            # list; otherwise pick the median of the new range.
            if defaults[key] not in override:
                defaults[key] = sorted(override)[len(override) // 2]

    csv_paths: list[Path] = []
    for dtype in args.dtypes:
        for axis in args.axes:
            for v in axes_table[axis]:
                point = dict(defaults)
                point[axis] = v
                if args.plot_only:
                    csv_paths.append(
                        _csv_path(data_dir, point, dtype, device_slug, args.allow_retain_graph, args.reduction)
                    )
                else:
                    path = _run_point(
                        point=point,
                        dtype=dtype,
                        device_type=device_type,
                        device_slug=device_slug,
                        reduction=args.reduction,
                        allow_retain_graph=args.allow_retain_graph,
                        include_acc_none=args.include_acc_none,
                        out_dir=data_dir,
                        warmup=args.warmup,
                        iters=args.iters,
                        force=args.force,
                    )
                    csv_paths.append(path)

    # Group plots by (dtype, device, reduction). reduction is part of the
    # key so a reduction='none' sweep doesn't clobber the 'mean' figure and
    # the two reductions never share a plot (their CSVs are separate files).
    grouped: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for p in csv_paths:
        if not p.exists():
            continue
        rows = _load_rows(p)
        if not rows:
            continue
        dtype = rows[0].get("dtype", "?")
        reduction = rows[0].get("reduction") or "mean"
        grouped[(dtype, device_name, reduction)].append(p)

    for (dtype, dev_name, reduction), paths in grouped.items():
        # Keep the 'mean' figure name unchanged; suffix the others.
        group_label = f"{dtype}_{dev_name}" + ("" if reduction == "mean" else f"_{reduction}")
        _plot(out_dir, paths, group_label, include_acc_none=args.include_acc_none)

    return 0


if __name__ == "__main__":
    sys.exit(main())

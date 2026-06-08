"""Focused auto-dispatch plotter.

Reads the CSVs produced by lce_benchmark_sweep.py and produces per-(dtype, device)
3-row figures showing only the labels relevant to the auto-dispatch story:

  * 'reference' (full-materialization F.linear + F.cross_entropy)
  * 'liger' (if present)
  * 'auto' (acc_policy='auto')
  * the two AR neighbors of auto's resolved chunking method (e.g. for
    'compact_aspect_ratio:2', neighbors are 'compact_aspect_ratio' and
    'compact_aspect_ratio:4')

Rows:
  1. time_ms
  2. memory_peak_gb
  3. merged gradient errors (grad_linear_weight_error + grad_input_error)
     with marker shape distinguishing the two metrics per series
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


AXIS_NAMES = ("num_tokens", "in_features", "num_classes")


# Match labels like "compact_aspect_ratio", "compact_aspect_ratio:2",
# and their "_fp32" variants (acc_dtype=torch.float32).
_LABEL_RE = re.compile(
    r"^(?P<policy>accurate|balanced|compact)_aspect_ratio(?::(?P<factor>\d+))?(?P<fp32>_fp32)?$"
)


def _label_to_policy_factor(label: str) -> Optional[tuple[str, int, bool]]:
    m = _LABEL_RE.match(label)
    if m is None:
        return None
    return m.group("policy"), int(m.group("factor") or "1"), bool(m.group("fp32"))


def _policy_factor_to_label(policy: str, factor: int, fp32: bool = False) -> str:
    base = f"{policy}_aspect_ratio" if factor == 1 else f"{policy}_aspect_ratio:{factor}"
    return base + ("_fp32" if fp32 else "")


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


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
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


# ---------------------------------------------------------------------------
# Pick the AR neighbors of the auto label observed in the data
# ---------------------------------------------------------------------------

def _pick_focus_labels(rows: list[dict]) -> list[str]:
    """Return the set of labels to plot for this dtype/device group."""
    have = {r["label"] for r in rows}
    focus: list[str] = []

    if "reference" in have:
        focus.append("reference")
    if "liger" in have:
        focus.append("liger")
    if "auto" in have:
        focus.append("auto")
    if "auto_fp32" in have:
        focus.append("auto_fp32")

    # Find the (policy, factor, fp32) that each auto* variant resolves to
    # and pull in its AR neighbors. auto is one of the non-fp32 chunked
    # rows, auto_fp32 is one of the _fp32 rows.
    def _neighbors_for(auto_label: str, want_fp32: bool) -> None:
        auto_rows = [r for r in rows if r["label"] == auto_label]
        if not auto_rows:
            return
        chunked = [
            r
            for r in rows
            if (pf := _label_to_policy_factor(r["label"])) is not None
            and pf[2] == want_fp32
        ]
        if not chunked:
            return
        auto_time = sum(r.get("time_ms", 0.0) for r in auto_rows) / len(auto_rows)
        by_label: dict[str, list[float]] = defaultdict(list)
        for r in chunked:
            by_label[r["label"]].append(r.get("time_ms", 0.0))
        avg = {lbl: sum(ts) / len(ts) for lbl, ts in by_label.items()}
        if not avg:
            return
        closest = min(avg, key=lambda lbl: abs(avg[lbl] - auto_time))
        pf = _label_to_policy_factor(closest)
        if pf is None:
            return
        policy, factor, fp32 = pf
        for f in (factor // 2, factor * 2):
            if f < 1:
                continue
            lbl = _policy_factor_to_label(policy, f, fp32)
            if lbl in have and lbl not in focus:
                focus.append(lbl)

    _neighbors_for("auto", False)
    _neighbors_for("auto_fp32", True)
    return focus


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

YROWS = [
    ("time_ms", "time (ms)", False, None),
    ("memory_peak_gb", "peak memory (GiB)", False, None),
    (None, "gradient rel error", True, ("grad_linear_weight_error", "grad_input_error")),
]


def _plot_group(
    out_dir: Path,
    rows: list[dict],
    group_label: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}", file=sys.stderr)
        return

    if not rows:
        return

    labels = _pick_focus_labels(rows)
    rows = [r for r in rows if r["label"] in labels]
    if not rows:
        return

    # Infer defaults
    counts: dict[str, Counter] = {a: Counter() for a in AXIS_NAMES}
    for r in rows:
        for a in AXIS_NAMES:
            counts[a][r[a]] += 1
    defaults = {a: counts[a].most_common(1)[0][0] for a in AXIS_NAMES}

    fig, axes = plt.subplots(
        len(YROWS), len(AXIS_NAMES), figsize=(15, 10), sharex="col"
    )

    for col, axis in enumerate(AXIS_NAMES):
        slab = [r for r in rows if all(r[a] == defaults[a] for a in AXIS_NAMES if a != axis)]
        by_label: dict[str, list[dict]] = defaultdict(list)
        for r in slab:
            by_label[r["label"]].append(r)
        for lbl, xs in by_label.items():
            xs.sort(key=lambda r: r[axis])
        col_xvals = sorted({r[axis] for r in slab})
        for row_idx, (yfield, ylabel, log, merged) in enumerate(YROWS):
            ax = axes[row_idx][col]
            any_positive = False
            def _track(ys: list) -> None:
                nonlocal any_positive
                if any(isinstance(y, (int, float)) and y == y and y > 0 for y in ys):
                    any_positive = True
            if merged is None:
                for lbl in labels:
                    xs = by_label.get(lbl, [])
                    if not xs:
                        continue
                    ys = [x.get(yfield, float("nan")) for x in xs]
                    xvals = [x[axis] for x in xs]
                    # Liger pinned to black + star marker.
                    if lbl == "liger":
                        style = {"color": "black", "marker": "*", "markersize": 9}
                    else:
                        style = {"marker": "o"}
                    ax.plot(
                        xvals,
                        ys,
                        label=lbl,
                        linewidth=1,
                        **style,
                    )
                    _track(ys)
            else:
                # Merged: per-(series, metric) markers; same color per
                # series so the two metrics of one config are visibly tied.
                markers = ("o", "s")
                for lbl in labels:
                    xs = by_label.get(lbl, [])
                    if not xs:
                        continue
                    color = "black" if lbl == "liger" else None
                    for mi, metric in enumerate(merged):
                        ys = [x.get(metric, float("nan")) for x in xs]
                        # Skip series that are entirely NaN
                        if all(y != y for y in ys):  # NaN check
                            continue
                        xvals = [x[axis] for x in xs]
                        line, = ax.plot(
                            xvals,
                            ys,
                            marker=markers[mi],
                            linestyle="-" if mi == 0 else "--",
                            label=f"{lbl} ({metric})",
                            linewidth=1,
                            color=color,
                        )
                        # Lock subsequent metric in same color as the first.
                        if mi == 0 and color is None:
                            color = line.get_color()
                        _track(ys)
            ax.set_xlabel(axis)
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            _apply_log_xticks(ax, col_xvals)
            ax.tick_params(labelbottom=True)
            # Skip log y-scale when no positive data exists in this subplot.
            if log and any_positive:
                ax.set_yscale("log")
            if row_idx == 0 and col == 0:
                ax.legend(fontsize=6, loc="best")
            # Marker legend for the merged grad-error row (col 0 only).
            if merged is not None and col == 0:
                from matplotlib.lines import Line2D

                marker_handles = [
                    Line2D(
                        [0], [0],
                        color="gray", marker=markers[mi],
                        linestyle="-" if mi == 0 else "--",
                        label=metric.replace("grad_", "").replace("_error", "") + " grad",
                    )
                    for mi, metric in enumerate(merged)
                ]
                ax.legend(
                    handles=marker_handles, fontsize=6, loc="lower right",
                    title="metric", title_fontsize=6,
                )

    fig.suptitle(f"auto dispatch: {group_label}")
    fig.tight_layout()
    out_path = out_dir / f"auto_dispatch_{group_label.replace(' ', '_').replace('/', '_')}.png"
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "sweep_out" / "data"))
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "sweep_out"))
    p.add_argument(
        "--newer-than-today",
        action="store_true",
        help="filter CSVs by mtime today (best-effort)",
    )
    p.add_argument(
        "--include-acc-none",
        action="store_true",
        help="also include rows with acc_dtype != float32. By default only "
        "the _fp32 (mixed-precision) variants plus reference/liger are "
        "plotted.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(data_dir.rglob("*.csv"))

    if args.newer_than_today:
        now = time.time()
        cutoff = now - 24 * 3600
        csv_paths = [p for p in csv_paths if p.stat().st_mtime >= cutoff]

    def _keep(r: dict) -> bool:
        if args.include_acc_none:
            return True
        return r.get("label") in ("reference", "liger") or r.get("acc_dtype") == "float32"

    # Group by (dtype, device, reduction). reduction is part of the key so
    # globbing all CSVs doesn't mix 'mean' and 'none' rows in one figure.
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for p in csv_paths:
        rows = _load_rows(p)
        for r in rows:
            if not _keep(r):
                continue
            dtype = r.get("dtype", "?")
            # use device subdir name as device label
            device_name = p.parent.name
            reduction = r.get("reduction") or "mean"
            grouped[(dtype, device_name, reduction)].append(r)

    for (dtype, device_name, reduction), rows in grouped.items():
        group_label = f"{dtype}_{device_name}" + ("" if reduction == "mean" else f"_{reduction}")
        _plot_group(out_dir, rows, group_label)

    return 0


if __name__ == "__main__":
    sys.exit(main())

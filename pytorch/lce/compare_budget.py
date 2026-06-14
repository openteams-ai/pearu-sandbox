"""Before/after overlay for the budget auto-switch PR (index targets).

Like the materialization-fix comparison, this reads two sweeps measured on
the same build differing only by the budget PR:

  sweep_out_budget/<before-subdir>/<device-slug>/*.csv  (main: auto == aspect_ratio)
  sweep_out_budget/<after-subdir>/<device-slug>/*.csv   (PR:   auto switches to budget)

and overlays them per axis: BEFORE dashed, AFTER solid, same color per label.
``reference`` and ``liger`` are untouched by the PR, so their before/after
lines must coincide -- they are the controls. ``auto_fp32`` is the line the
PR changes: it coincides below the crossing region (num_classes >
in_features) and, above it, the AFTER line bounds the per-chunk accumulator
(budget) that the BEFORE line (aspect_ratio) lets grow quadratically.

Rows: peak memory, time, gradient rel error vs fp64. The grad-error row
carries the accuracy story (chunked keeps fp32-acc logits, so it stays more
accurate than the reference at every shape, which is why auto switches
method rather than falling back to the reference).

Usage:
    python compare_budget.py [--before-subdir data-before] [--after-subdir data] \
        [--dtype float16] [--device-slug SLUG]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from lce_benchmark_sweep import (
    AXIS_NAMES,
    _apply_log_xticks,
    _load_rows,
    _local_device_slug,
)

_THIS_DIR = Path(__file__).resolve().parent

# Labels to overlay. reference/liger are the unchanged controls; auto_fp32 is
# the line the PR moves (aspect_ratio below the crossing, budget above).
FOCUS = ["reference", "liger", "auto_fp32"]
YROWS = [
    ("memory_peak_gb", "peak memory (GiB)", False),
    ("time_ms", "time (ms)", False),
    ("grad_input_error", "grad input rel error", True),
]


def _load_index_rows(data_dir: Path, dtype: str) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(data_dir.rglob("*.csv")):
        rows += [
            r
            for r in _load_rows(p)
            if r.get("prob_target") not in (True, "True")  # index targets only
            and r.get("dtype") == dtype
        ]
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(_THIS_DIR / "sweep_out_budget"))
    p.add_argument("--before-subdir", default="data-before")
    p.add_argument("--after-subdir", default="data")
    p.add_argument("--dtype", default="float16")
    p.add_argument(
        "--device-slug",
        default=None,
        help="device subdir to read; defaults to the local device slug "
        "(an unscoped rglob would mix rows from other machines' subdirs)",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    slug = args.device_slug or _local_device_slug()[1]
    runs = {}
    for tag, sub in (("before", args.before_subdir), ("after", args.after_subdir)):
        rows = _load_index_rows(out_dir / sub / slug, args.dtype)
        if not rows:
            print(f"no {args.dtype} index rows under {out_dir / sub / slug}", file=sys.stderr)
            return 1
        runs[tag] = rows

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = {a: Counter() for a in AXIS_NAMES}
    for r in runs["after"]:
        for a in AXIS_NAMES:
            counts[a][r[a]] += 1
    defaults = {a: counts[a].most_common(1)[0][0] for a in AXIS_NAMES}

    fig, axes = plt.subplots(
        len(YROWS), len(AXIS_NAMES), figsize=(15, 11), sharex="col", squeeze=False
    )
    colors: dict[str, str] = {}
    for col, axis in enumerate(AXIS_NAMES):
        col_xvals = set()
        for row_idx, (yfield, ylabel, log) in enumerate(YROWS):
            ax = axes[row_idx][col]
            any_positive = False
            for tag, ls, mk in (("before", "--", "x"), ("after", "-", "o")):
                slab = [
                    r for r in runs[tag]
                    if all(r[a] == defaults[a] for a in AXIS_NAMES if a != axis)
                ]
                by_label: dict[str, list[dict]] = defaultdict(list)
                for r in slab:
                    by_label[r["label"]].append(r)
                for label in FOCUS:
                    xs = sorted(by_label.get(label, []), key=lambda r: r[axis])
                    if not xs:
                        continue
                    ys = [x.get(yfield, float("nan")) for x in xs]
                    xvals = [x[axis] for x in xs]
                    col_xvals.update(xvals)
                    line, = ax.plot(
                        xvals, ys,
                        label=f"{label} ({tag})" if (row_idx, col) == (0, 0) else None,
                        linestyle=ls, marker=mk, linewidth=1.3,
                        color=colors.get(label),
                    )
                    colors.setdefault(label, line.get_color())
                    if any(isinstance(y, (int, float)) and y == y and y > 0 for y in ys):
                        any_positive = True
            ax.set_xlabel(axis)
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            _apply_log_xticks(ax, sorted(col_xvals))
            ax.tick_params(labelbottom=True)
            if log and any_positive:
                ax.set_yscale("log")
            if (row_idx, col) == (0, 0):
                ax.legend(fontsize=7, loc="best")

    name = _local_device_slug()[0] if args.device_slug is None else args.device_slug
    fig.suptitle(
        f"budget auto-switch: {args.dtype} {name} (index targets; dashed=before, "
        "solid=after)\nreference / liger are unchanged controls; auto_fp32 is "
        "the line the PR bounds above the crossing"
    )
    fig.tight_layout()
    png = out_dir / f"budget_before_after_{args.dtype}_{slug}.png"
    fig.savefig(png, dpi=120)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

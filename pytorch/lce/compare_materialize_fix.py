"""Before/after overlay for the unused-output grad-materialization fix.

The chunked scalar-reduction op returns (loss, grad_input,
grad_linear_weight, grad_linear_bias); without ctx.set_materialize_grads(False)
every backward materializes zero-filled gradients for the three unused
outputs -- (N, F) + (C, F) + (C,) at input dtype -- inflating the peak.

Reads two sweeps measured on the same build differing only by the fix:

  sweep_out/<before-subdir>/<device-slug>/*.csv   (fix reverted)
  sweep_out/<after-subdir>/<device-slug>/*.csv    (fix applied)

and overlays peak memory and time per axis: BEFORE dashed, AFTER solid,
same color per label. ``reference`` is the control -- its path is
untouched by the fix, so its dashed/solid lines must coincide.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from lce_benchmark_sweep import AXIS_NAMES, _apply_log_xticks, _load_rows, _local_device_slug

_THIS_DIR = Path(__file__).resolve().parent

FOCUS = ["reference", "auto_fp32", "accurate_aspect_ratio:2_fp32",
         "balanced_aspect_ratio:2_fp32", "compact_aspect_ratio:2_fp32"]
YROWS = [("memory_peak_gb", "peak memory (GiB)"), ("time_ms", "time (ms)")]
# PR-figure mode: memory only, auto dispatch vs reference.
SLIM_FOCUS = ["reference", "auto_fp32"]


def _load_dir(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(data_dir.rglob("*.csv")):
        rows += [r for r in _load_rows(p)
                 if r.get("prob_target") not in (True, "True")]  # index targets
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--before-subdir", default="data-before-fix")
    p.add_argument("--after-subdir", default="data")
    p.add_argument("--out-dir", default=str(_THIS_DIR / "sweep_out"))
    p.add_argument("--dtype", default="float16")
    p.add_argument(
        "--device-slug",
        default=None,
        help="device subdir to read; defaults to the local device slug "
        "(an unscoped rglob would mix rows from other machines' subdirs)",
    )
    p.add_argument(
        "--slim",
        action="store_true",
        help="auto dispatch vs reference only; the PR-figure variant",
    )
    args = p.parse_args()
    focus = SLIM_FOCUS if args.slim else FOCUS
    yrows = YROWS

    out_dir = Path(args.out_dir)
    slug = args.device_slug or _local_device_slug()[1]
    runs = {}
    for tag, sub in (("before", args.before_subdir), ("after", args.after_subdir)):
        rows = [r for r in _load_dir(out_dir / sub / slug) if r.get("dtype") == args.dtype]
        if not rows:
            print(f"no {args.dtype} index rows under {out_dir / sub}", file=sys.stderr)
            return 1
        runs[tag] = rows

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Per-axis defaults from the AFTER run (mode over non-swept axes).
    counts = {a: Counter() for a in AXIS_NAMES}
    for r in runs["after"]:
        for a in AXIS_NAMES:
            counts[a][r[a]] += 1
    defaults = {a: counts[a].most_common(1)[0][0] for a in AXIS_NAMES}

    fig, axes = plt.subplots(
        len(yrows), len(AXIS_NAMES),
        figsize=(15, 8), sharex="col", squeeze=False,
    )
    colors: dict[str, str] = {}
    for col, axis in enumerate(AXIS_NAMES):
        col_xvals = set()
        for row_idx, (yfield, ylabel) in enumerate(yrows):
            ax = axes[row_idx][col]
            for tag, style in (("before", "--"), ("after", "-")):
                slab = [r for r in runs[tag]
                        if all(r[a] == defaults[a] for a in AXIS_NAMES if a != axis)]
                by_label = defaultdict(list)
                for r in slab:
                    by_label[r["label"]].append(r)
                for label in focus:
                    xs = sorted(by_label.get(label, []), key=lambda r: r[axis])
                    if not xs:
                        continue
                    xvals = [x[axis] for x in xs]
                    col_xvals.update(xvals)
                    line, = ax.plot(
                        xvals, [x.get(yfield, float("nan")) for x in xs],
                        linestyle=style, marker="o" if tag == "after" else "x",
                        linewidth=1, color=colors.get(label),
                        label=f"{label} ({tag})" if (row_idx, col) == (0, 0) else None,
                    )
                    colors.setdefault(label, line.get_color())
            ax.set_xlabel(axis)
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            _apply_log_xticks(ax, col_xvals)
            ax.tick_params(labelbottom=True)
            if (row_idx, col) == (0, 0):
                ax.legend(fontsize=6, loc="best")

    fig.suptitle(
        f"index-target sweep, {args.dtype}: unused-output grad materialization fix"
        " (dashed=before, solid=after; reference is the unaffected control)"
    )
    fig.tight_layout()
    png = out_dir / f"materialize_fix_index_{args.dtype}{'_slim' if args.slim else ''}.png"
    fig.savefig(png, dpi=120)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

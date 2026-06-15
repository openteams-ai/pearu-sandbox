"""Focused single-panel budget justification: in_features vs peak memory.

Shows the crossing the budget method fixes, on one N/V slice, without liger
(which would compress the y-axis). Two modes:

  --mode before-after  (default): reference + auto-before (main, aspect_ratio,
      dashed) + auto-after (budget, solid). Needs both data/ and data-before/.
      Used for the index-target #187271 figure.

  --mode methods: reference + compact_aspect_ratio:2 + compact_budget, all from
      one build's data/. Used when there is no before build to diff against
      (e.g. the probability-target case, whose build always has budget); the
      explicit method lines carry the comparison directly.

The story: aspect_ratio rises ABOVE reference for large in_features (chunked
losing to full materialization); budget keeps the chunked peak at the
reference floor. Modest by design (budget bounds the per-chunk buffer to ~one
input footprint) -- a robustness fix, not a large win.
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _series(data_dir, slug, dtype, label, prob, N, V):
    pts = []
    for p in glob.glob(f"{data_dir}/{slug}/*{dtype}*.csv"):
        for r in csv.DictReader(open(p)):
            if not r.get("num_classes"):
                continue
            if (r.get("prob_target") in (True, "True")) != prob:
                continue
            if r["label"] != label or int(r["num_tokens"]) != N or int(r["num_classes"]) != V:
                continue
            m = _f(r["memory_peak_bytes"])
            if m == m:
                pts.append((int(r["in_features"]), m / 2**30))
    return sorted(pts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(_THIS / "sweep_out_budget"))
    p.add_argument("--mode", choices=("before-after", "methods"), default="before-after")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-slug", required=True)
    p.add_argument("--prob", action="store_true", help="probability targets")
    p.add_argument("--num-tokens", type=int, default=4096)
    p.add_argument("--num-classes", type=int, default=32000)
    args = p.parse_args()

    out = Path(args.out_dir)
    after = str(out / "data")
    before = str(out / "data-before")
    N, V, slug, dt, prob = args.num_tokens, args.num_classes, args.device_slug, args.dtype, args.prob

    if args.mode == "before-after":
        plots = [
            (after, "reference", "reference (full materialization)", dict(color="C0", marker="o")),
            (before, "auto_fp32", "auto BEFORE = aspect_ratio (main)", dict(color="C1", marker="x", ls="--")),
            (after, "auto_fp32", "auto AFTER = budget (#187271)", dict(color="C2", marker="o")),
        ]
    else:
        plots = [
            (after, "reference", "reference (full materialization)", dict(color="C0", marker="o")),
            (after, "compact_aspect_ratio:2_fp32", "aspect_ratio (crosses reference)", dict(color="C1", marker="x", ls="--")),
            (after, "compact_budget_fp32", "budget (bounded)", dict(color="C2", marker="o")),
        ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    any_data = False
    for data_dir, label, disp, style in plots:
        s = _series(data_dir, slug, dt, label, prob, N, V)
        if s:
            any_data = True
            ax.plot([d for d, _ in s], [m for _, m in s], label=disp, linewidth=1.6, **style)
    if not any_data:
        print("no data found; check --device-slug / --prob / data dirs", file=sys.stderr)
        return 1

    kind = "probability-target" if prob else "index-target"
    ax.set_xscale("log")
    ax.set_xlabel(f"in_features  (num_classes={V}, num_tokens={N})")
    ax.set_ylabel("peak memory (GiB)")
    ax.set_title(
        f"{dt} {slug}, {kind}: budget keeps chunked peak at the reference floor\n"
        "(aspect_ratio crosses ABOVE reference for large in_features)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    tag = "prob" if prob else "index"
    png = out / f"budget_justification_{tag}_{dt}_{slug}.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

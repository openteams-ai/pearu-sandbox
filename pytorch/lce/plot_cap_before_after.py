"""Before/after figure for the N*V/4D chunking-cap PR (index target).

Reads an in_features sweep produced by lce_benchmark_sweep.py (fixed N, V;
D crossing V) and draws a 2-row figure -- peak memory and time -- comparing,
on the SAME build, five configs expressed via ``options``:

  reference  : options=None (full materialization)
  liger      : liger fused op (index targets only)
  before     : aspect_ratio:2  (the old auto default for cuda index)
  uncapped   : aspect_ratio    (factor 1, NO cap -- what the new default would
               be without the memory cap; shows what the cap prevents)
  after      : auto            (factor 1 + the N*V/4D cap -- the new default)

Accuracy is intentionally omitted (the ULP caps are unchanged across platforms).

    python plot_cap_before_after.py            # bf16, N=4096 V=8192
    python plot_cap_before_after.py --dtype float16 --num-tokens 4096 --num-classes 8192
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GiB = 1024 ** 3

# (label-in-csv, legend, color, linestyle, marker). bf16/fp16 emit the
# acc_dtype=fp32 variants, so the chunked labels carry a _fp32 suffix.
SERIES = [
    ("reference", "reference (full materialization)", "black", "-", "o"),
    ("liger", "liger (index-only)", "tab:gray", "-", "*"),
    ("compact_aspect_ratio:2{s}", "before: aspect_ratio:2", "tab:orange", "-", "s"),
    ("compact_aspect_ratio{s}", "uncapped: aspect_ratio (factor 1)", "tab:red", "--", "v"),
    ("auto{s}", "after: auto (factor 1 + N*V/4D cap)", "tab:green", "-", "^"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "sweep_out" / "data"))
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "sweep_out"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--x-axis", default="in_features", choices=["in_features", "num_classes"])
    p.add_argument("--num-tokens", type=int, default=4096)
    p.add_argument("--num-classes", type=int, default=8192, help="fixed V (when x-axis=in_features)")
    p.add_argument("--in-features", type=int, default=4096, help="fixed D (when x-axis=num_classes)")
    args = p.parse_args()
    suffix = "_fp32" if args.dtype in ("bfloat16", "float16") else ""

    # in_features sweep: N, V fixed, D varies. num_classes sweep: N, D fixed, V varies.
    if args.x_axis == "in_features":
        pattern = f"N{args.num_tokens}_D*_V{args.num_classes}_{args.dtype}.csv"
        xlabel, fixed_label = "in_features (D)", f"V={args.num_classes}"
    else:
        pattern = f"N{args.num_tokens}_D{args.in_features}_V*_{args.dtype}.csv"
        xlabel, fixed_label = "num_classes (V)", f"D={args.in_features}"
    files = glob.glob(str(Path(args.data_dir) / "**" / pattern), recursive=True)
    files = [f for f in files if "reduction-none" not in f and "prob" not in f]
    if not files:
        raise SystemExit(f"no CSVs match {pattern} under {args.data_dir}")

    device = ""
    by_x: dict[int, dict[str, dict]] = {}
    for f in files:
        rows = {r["label"]: r for r in csv.DictReader(open(f))}
        ref = rows.get("reference")
        if ref is None:
            continue
        device = ref["device_type"]
        by_x[int(ref[args.x_axis])] = rows
    xs_all = sorted(by_x)

    fig, (ax_mem, ax_t) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for key, legend, color, ls, marker in SERIES:
        label = key.format(s=suffix)
        mem, tms, xs = [], [], []
        for x in xs_all:
            r = by_x[x].get(label)
            if r is None:
                continue
            xs.append(x)
            mem.append(int(r["memory_peak_bytes"]) / GiB)
            tms.append(float(r["time_ms"]))
        if not xs:
            continue
        ax_mem.plot(xs, mem, marker=marker, color=color, ls=ls, label=legend)
        ax_t.plot(xs, tms, marker=marker, color=color, ls=ls, label=legend)

    # Budget regime (in_features >= num_classes): the cap binds, trading
    # throughput for <=reference memory -- atypical for index (vocab heads have
    # V >> D), so the time cost there is the intended tradeoff, not a regression.
    # It sits at the high-D end of an in_features sweep, the low-V end of a
    # num_classes sweep.
    lo, hi = min(xs_all), max(xs_all)
    if args.x_axis == "in_features":
        span = (max(args.num_classes, lo), hi) if hi >= args.num_classes else None
    else:
        span = (lo, min(args.in_features, hi)) if lo <= args.in_features else None
    for ax in (ax_mem, ax_t):
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        if span:
            ax.axvspan(*span, color="gray", alpha=0.10)
    if span:
        ax_t.annotate("budget regime (D>=V): cap trades throughput\n"
                      "for <=ref memory; atypical for index",
                      (span[0], ax_t.get_ylim()[1]), ha="left", va="top",
                      color="dimgray", fontsize=7)
    ax_mem.set_ylabel("peak memory (GiB)")
    ax_t.set_ylabel("fwd+bwd time (ms)")
    ax_t.set_xlabel(xlabel)
    ax_mem.set_title(f"factor-1 + N*V/4D cap, index target ({device}, {args.dtype}, "
                     f"N={args.num_tokens}, {fixed_label})\n"
                     f"after (auto) maximizes throughput in the vocab regime, "
                     f"stays <= reference everywhere")
    ax_mem.legend(fontsize=8)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"V{args.num_classes}" if args.x_axis == "in_features" else f"D{args.in_features}_Vsweep"
    fpath = out / f"cap_before_after_{args.dtype}_{device}_N{args.num_tokens}_{tag}.png"
    fig.tight_layout()
    fig.savefig(fpath, dpi=130)
    print(f"wrote {fpath}")


if __name__ == "__main__":
    main()

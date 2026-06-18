"""Summary figure for the cross-GPU chunk-size throughput sweep.

Reads chunk_size_sweep_<gpu>.csv files, extracts B_knee per (device, shape) at a
fixed throughput-plateau tolerance, and shows the headline result:
B_knee = k * SM_count, with k set by tensor-core generation -- Ampere/Hopper one
class, Blackwell ~1.25x higher. Left: per-device geomean B_knee vs SM count with
the two class lines through the origin. Right: B/SM per device (the constant).
"""
from __future__ import annotations

import glob
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chunk_size_sweep import _knee, _load, _sm_count

_THIS = Path(__file__).resolve().parent
TOL = 0.05


def _arch_class(dev: str) -> str:
    up = dev.upper()
    return "Blackwell" if ("B200" in up or "B300" in up) else "Ampere/Hopper"


def _short(dev: str) -> str:
    for k in ("A100", "H100", "H200", "B200", "B300"):
        if k in dev.upper():
            return k
    return dev


def main() -> None:
    files = sorted(glob.glob(str(_THIS / "chunk_size_sweep_*.csv")))
    knees: dict[str, list[int]] = defaultdict(list)
    dev_sm: dict[str, int] = {}
    for f in files:
        rows = _load(Path(f))
        by_shape: dict[tuple, dict] = defaultdict(dict)
        for r in rows:
            dev = r["device"]
            dev_sm.setdefault(dev, r["sm_count"] or _sm_count(dev) or 0)
            sh = (dev, r["num_tokens"], r["in_features"], r["num_classes"])
            if r["mode"] == "reference":
                by_shape[sh]["ref"] = r
            else:
                by_shape[sh].setdefault("chunked", []).append(r)
        for sh, rec in by_shape.items():
            kn = _knee(rec.get("chunked", []), TOL)
            if kn:
                knees[sh[0]].append(kn["batch_chunk_size"])

    devs = sorted(knees, key=lambda d: (_arch_class(d), dev_sm[d]))
    # per-device geomean B_knee + B/SM
    const = {d: math.exp(sum(math.log(b) for b in knees[d]) / len(knees[d])) for d in devs}
    bsm = {d: const[d] / dev_sm[d] for d in devs}
    # class slopes = mean B/SM over the class
    cls_k: dict[str, float] = {}
    for c in ("Ampere/Hopper", "Blackwell"):
        members = [d for d in devs if _arch_class(d) == c]
        cls_k[c] = sum(bsm[d] for d in members) / len(members)

    colors = {"Ampere/Hopper": "tab:blue", "Blackwell": "tab:red"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: B_knee vs SM with class lines through origin
    xmax = max(dev_sm.values()) * 1.12
    for c, k in cls_k.items():
        axL.plot([0, xmax], [0, k * xmax], "--", color=colors[c], lw=1.2,
                 label=f"{c}: B = {k:.1f}*SM")
    # Group devices that land on the same (SM, const) point -- H100=H200 and
    # B200=B300 coincide, which IS the memory/SKU-invariance result; label them
    # jointly so the overlap reads as a feature, not a collision.
    groups: dict[tuple, list[str]] = defaultdict(list)
    for d in devs:
        groups[(dev_sm[d], round(const[d]))].append(d)
    for d in devs:
        c = _arch_class(d)
        axL.scatter([dev_sm[d]] * len(knees[d]), knees[d], color=colors[c],
                    alpha=0.35, s=22)
        axL.scatter([dev_sm[d]], [const[d]], color=colors[c], s=130,
                    edgecolor="black", zorder=5, marker="D")
    for (sm, cst), members in groups.items():
        c = _arch_class(members[0])
        label = "=".join(_short(m) for m in members)
        axL.annotate(f"{label}\n{sm} SM", (sm, cst),
                     textcoords="offset points", xytext=(9, -2), fontsize=8,
                     color=colors[c])
    axL.set_xlabel("SM count")
    axL.set_ylabel(f"B_knee  (geomean; tol={TOL:.0%})")
    axL.set_title("Throughput-saturating chunk size scales with SM count\n"
                  "(diamond = per-device geomean; dots = per-shape knees)")
    axL.set_xlim(left=0)
    axL.set_ylim(bottom=0)
    axL.legend(loc="upper left", fontsize=9)
    axL.grid(alpha=0.3)

    # Right: B/SM bars
    order = devs
    axR.bar(range(len(order)), [bsm[d] for d in order],
            color=[colors[_arch_class(d)] for d in order], edgecolor="black")
    for i, d in enumerate(order):
        axR.text(i, bsm[d] + 0.2, f"{bsm[d]:.1f}", ha="center", fontsize=9)
    axR.set_xticks(range(len(order)))
    axR.set_xticklabels([_short(d) for d in order])
    axR.set_ylabel("B/SM (rows per SM)")
    axR.set_title(f"k = B_knee / SM_count  (tol={TOL:.0%})\n"
                  "memory tier (H200=H100) and SKU (B200=B300) do not move it")
    axR.grid(axis="y", alpha=0.3)

    fig.suptitle("LCE budget-regime chunking: throughput-optimal B is a per-arch SM-constant",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = _THIS / "chunk_size_summary.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print("per-device B/SM:", {_short(d): round(bsm[d], 2) for d in devs})
    print("class k:", {c: round(k, 2) for c, k in cls_k.items()})


if __name__ == "__main__":
    main()

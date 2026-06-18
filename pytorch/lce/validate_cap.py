"""Validate the min(aspect_ratio_B, k*SM) cap from existing sweep CSVs -- on main.

The chunk_size_sweep CSVs already measured time+memory at every power-of-two B
per shape (explicit batch_chunk_size, chunking_method=None -- the landed #187219
API). So the cap is testable on plain main with no heuristic implemented: for
each shape read off the measured time/memory at aspect_ratio's uncapped chunk
and at the capped chunk B_cap = min(aspect_ratio_B, round_pow2(k*SM)), and check
the cap is a Pareto move (>= throughput, <= memory) and where it engages.

k*SM is taken per device as the geomean B_knee at --tol (the measured saturation
constant). aspect_ratio uses factor 2 (the auto index-target CUDA default).
"""
from __future__ import annotations

import glob
import math
from collections import defaultdict
from pathlib import Path

from chunk_size_sweep import _knee, _load, _sm_count

_THIS = Path(__file__).resolve().parent
TOL = 0.05
FACTOR = 2  # auto index-target CUDA default is aspect_ratio:2


def _round_pow2(x: float) -> int:
    return 1 << round(math.log2(x)) if x > 0 else 1


def _next_pow2(x: int) -> int:
    v = 1
    while v < x:
        v *= 2
    return v


def _aspect_ratio_b(N: int, D: int, V: int, factor: int) -> int:
    inc = -(-V // D)               # ceil(V/D)
    return max(1, _next_pow2(-(-N // inc)) // factor)   # next_pow2(ceil(N/inc))/factor


def _liger_b(N: int, D: int, V: int) -> int:
    # liger's own chunk: next_pow2(ceil(BT / ceil(V/H))), factor 1.
    return _aspect_ratio_b(N, D, V, 1)


# Representative LLM-region (V >> D) shapes -- the cap engages only if
# aspect_ratio's chunk already exceeds k*SM there.
_LLM_SHAPES = [
    ("Llama3 head, 8k tok", 8192, 4096, 128256),
    ("Llama3 head, 32k tok", 32768, 4096, 128256),
    ("Gemma head, 8k tok", 8192, 3072, 256000),
    ("GPT2-ish, 16k tok", 16384, 1024, 50257),
    ("huge batch, 128k tok", 131072, 4096, 128256),
]


def _llm_engagement(cap: int) -> None:
    print(f"\nLLM-region cap engagement (cap = round_pow2(k*SM) = {cap}; "
          "cap engages iff aspect_ratio_B > cap -- pure arithmetic, no sweep):")
    print(f"  {'shape':>22} {'N':>7} {'D':>6} {'V':>7} | {'aspB:2':>7} {'aspB:1':>7} {'liger':>6} | {'engaged?':>9}")
    for lbl, N, D, V in _LLM_SHAPES:
        b2, b1, lg = _aspect_ratio_b(N, D, V, 2), _aspect_ratio_b(N, D, V, 1), _liger_b(N, D, V)
        eng = "yes" if max(b2, b1) > cap else "no"
        print(f"  {lbl:>22} {N:>7} {D:>6} {V:>7} | {b2:>7} {b1:>7} {lg:>6} | {eng:>9}")
    print(f"  Cap engages only above N ~ cap*factor*ceil(V/D); for cap={cap}, factor<=2, V/D>=8 "
          f"that is N > ~{cap*2*8}. Typical LLM training N (<= ~32768) is well below -> cap INERT in the LLM region.")


def _bref_fit(dev: str, ref: dict, chunked: dict) -> None:
    """B_ref = largest swept B with chunked(B) <= reference, per shape; fit c*N*V/D.

    This is the memory cap: the most efficient chunk that does not exceed
    reference memory. Memory is device-independent, so c should match across
    GPUs (unlike the per-arch throughput k).
    """
    rows = []
    for sh in sorted(chunked):
        if sh not in ref:
            continue
        N, D, V = sh
        fit = [b for b, (_, m) in chunked[sh].items() if m == m and m <= ref[sh]]
        if fit:
            rows.append((N, D, V, max(fit)))
    if not rows:
        return
    ratios = [b / (N * V / D) for N, D, V, b in rows]
    c = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    l2 = (sum(math.log2(c * (N * V / D) / b) ** 2 for N, D, V, b in rows) / len(rows)) ** 0.5
    print(f"  B_ref = largest B with chunked <= reference;  fit B_ref ~= {c:.3f} * N*V/D "
          f"(log2 RMSE {l2:.3f})")
    print(f"  {'N':>6} {'D':>7} {'V':>7} | {'B_ref':>6} {'N*V/D':>8} {'B_ref/(NV/D)':>12}")
    for N, D, V, b in rows:
        print(f"  {N:>6} {D:>7} {V:>7} | {b:>6} {N * V / D:>8.0f} {b / (N * V / D):>12.4f}")


def main() -> None:
    gib = 1024 ** 3
    for f in sorted(glob.glob(str(_THIS / "chunk_size_sweep_*.csv"))):
        rows = _load(Path(f))
        chunked: dict[tuple, dict] = defaultdict(dict)   # shape -> {B: (t, mem)}
        ref: dict[tuple, float] = {}                     # shape -> reference mem
        knees: list[int] = []
        dev, sm = "?", 0
        for r in rows:
            sh = (r["num_tokens"], r["in_features"], r["num_classes"])
            if r["mode"] == "reference":
                ref[sh] = r["memory_peak_bytes"]
                continue
            if r["mode"] != "chunked":
                continue
            dev = r["device"]
            sm = r["sm_count"] or _sm_count(dev) or 0
            chunked[sh][r["batch_chunk_size"]] = (r["time_ms"], r["memory_peak_bytes"])
        for sh, bmap in chunked.items():
            kn = _knee([{"batch_chunk_size": b, "time_ms": t}
                        for b, (t, _) in bmap.items()], TOL)
            if kn:
                knees.append(kn["batch_chunk_size"])
        if not knees:
            continue
        const = math.exp(sum(math.log(b) for b in knees) / len(knees))
        cap = _round_pow2(const)

        print(f"\n{dev}  (SM={sm}; k*SM~={const:.0f} -> cap=round_pow2={cap}; tol={TOL:.0%}, aspect_ratio:{FACTOR})")
        print(f"  {'N':>6} {'D':>7} {'V':>7} | {'aspect_B':>9} {'B_cap':>6} {'engaged':>8} "
              f"| {'t_cap/t_asp':>11} {'mem_cap/mem_asp':>15}")
        for sh in sorted(chunked):
            N, D, V = sh
            asp = _aspect_ratio_b(N, D, V, FACTOR)
            bcap = min(asp, cap)
            bmap = chunked[sh]
            if asp not in bmap or bcap not in bmap:
                continue
            t_asp, m_asp = bmap[asp]
            t_cap, m_cap = bmap[bcap]
            eng = "yes" if bcap < asp else "no"
            tr = t_cap / t_asp if t_asp == t_asp and t_asp else float("nan")
            mr = m_cap / m_asp if m_asp == m_asp and m_asp else float("nan")
            print(f"  {N:>6} {D:>7} {V:>7} | {asp:>9} {bcap:>6} {eng:>8} "
                  f"| {tr:>11.3f} {mr:>15.3f}")
        print("  (cap trades <= tol throughput for memory: t_cap/t_asp is the cost, "
              "mem_cap/mem_asp < 1.0 the saving.)")
        print()
        _bref_fit(dev, ref, chunked)
        _llm_engagement(cap)


if __name__ == "__main__":
    main()

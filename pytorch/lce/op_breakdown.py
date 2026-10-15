#!/usr/bin/env python3
"""Per-operation breakdown of the chunked LCE call, eager vs the override.

The chunked loop is four operations per chunk: three cuBLAS GEMMs (logits,
grad_input, grad_weight) and the auxiliary work between them -- a row-shifted
softmax and the gradient-of-logits transform, which eager walks the (Bc, V)
buffer several times to do and the override does in one kernel. The GEMMs are
identical in both arms (same chunk size, same shapes, same output dtypes), so
their ratio is a CONTROL: it should read 1.0, and a deviation means something
other than the kernel changed. Auxiliary is total minus the three GEMMs, so
nothing can hide between categories.

**Why this instrument is minimal, learned the hard way.** A previous version
created two `torch.cuda.Event` objects per timed call and also bracketed every
auxiliary pointwise op. At small chunk sizes eager issues ~19 device calls per
chunk and the loop is host-bound, so that overhead landed INSIDE the outer
total and inflated eager's arm 2.5x against the uninstrumented sweep -- turning
a real 1.56x into a reported 2.94x. Two rules follow: events are pre-created
and reused, never allocated per call; and only the three GEMMs plus the whole
call are bracketed. Per-call intervals also span host launch latency whenever
the stream is starved, which is why there is no attempt here to split the
auxiliary into device and idle time -- that split needs nsys, or launch counts
times a separately measured per-launch cost.

`--analyze` cross-checks every instrumented total against the same point's
uninstrumented row in the sweep CSVs and flags any arm outside a 5% band. A
flagged point is excluded from the headline medians: at those shapes this tool
cannot measure what it is asked to measure.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys

# Sites classified per iteration: the same site set recurs, so the mapping is
# stable, but it is recomputed rather than cached to keep the worker stateless.


SHAPES = [
    (1024, 4096, 32000), (2048, 4096, 32000), (4096, 4096, 32000),
    (8192, 4096, 32000), (16384, 4096, 32000),
    (4096, 1024, 32000), (4096, 2048, 32000), (4096, 8192, 32000),
    (4096, 4096, 4096), (4096, 4096, 8192), (4096, 4096, 16384),
    (4096, 4096, 65536),
]
STAGES = ("off", "fused_inplace")
CATEGORIES = ("logits_gemm", "grad_input_gemm", "grad_weight_gemm", "aux", "other")


def _worker(payload):
    import torch
    import torch.nn.functional as F
    from torch.nn import LinearCrossEntropyOptions

    torch.manual_seed(payload["seed"])
    dtype = getattr(torch, payload["dtype"])
    N, D, V = payload["num_tokens"], payload["in_features"], payload["num_classes"]
    dev = "cuda"

    input = torch.randn(N, D, dtype=dtype, device=dev, requires_grad=True)
    weight = (torch.randn(V, D, dtype=dtype, device=dev) * (1.0 / D ** 0.5)).detach()
    weight.requires_grad_(True)
    target = torch.randint(0, V, (N,), device=dev)
    options = LinearCrossEntropyOptions(
        acc_policy="auto", chunking_method="auto", acc_dtype=torch.float32
    )

    events = []
    sites = {}
    # Pre-created and reused: allocating events per call is what broke v1.
    pool = [torch.cuda.Event(enable_timing=True) for _ in range(1024)]
    pool_next = [0]

    def _pair():
        i = pool_next[0]
        if i + 2 > len(pool):
            pool.extend(torch.cuda.Event(enable_timing=True) for _ in range(len(pool)))
        pool_next[0] = i + 2
        return pool[i], pool[i + 1]

    def _wrap(fn, out_of=None, fixed_tag=None):
        """Time every call, tagging by CALL SITE rather than by shape.

        D == V at one sweep point, so the logits and grad_input GEMMs cannot be
        told apart by their output shape; their call sites always differ, and
        the site that runs first in a chunk is the logits one.
        """
        def wrapper(*args, **kwargs):
            start, end = _pair()
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            if fixed_tag is not None:
                tag = fixed_tag
            else:
                frame = sys._getframe(1)
                tag = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
                if tag not in sites:
                    written = out_of(args, kwargs)
                    sites[tag] = (len(events), tuple(written.shape) if written is not None else None)
            events.append((tag, start, end))
            return out
        return wrapper

    torch.mm = _wrap(torch.mm, out_of=lambda a, k: k.get("out"))
    torch.addmm = _wrap(torch.addmm, out_of=lambda a, k: k.get("out"))
    torch.Tensor.addmm_ = _wrap(torch.Tensor.addmm_, out_of=lambda a, k: a[0])
    try:
        from torch._native.ops.linear_cross_entropy import fused_grad_logits_kernel as fk
        fk.fused_grad_logits_into = _wrap(fk.fused_grad_logits_into, fixed_tag="aux_kernel")
    except ImportError:
        pass

    def _categories():
        """site -> category, from what each site writes and when it first ran."""
        out = {}
        rest = []
        for tag, (first, shape) in sites.items():
            if shape is None:
                out[tag] = "other"
            elif shape[0] == V:
                out[tag] = "grad_weight_gemm"
            else:
                rest.append((first, tag))
        for i, (_, tag) in enumerate(sorted(rest)):
            out[tag] = "logits_gemm" if i == 0 else "grad_input_gemm"
        return out

    def call():
        for t in (input, weight):
            t.grad = None
        F.linear_cross_entropy(
            input, weight, target, reduction="mean", options=options
        ).backward()

    for _ in range(payload["warmup"]):
        call()
    torch.cuda.synchronize()

    per_iter = []
    for _ in range(payload["iters"]):
        events.clear()
        pool_next[0] = 0
        total_start, total_end = torch.cuda.Event(True), torch.cuda.Event(True)
        total_start.record()
        call()
        total_end.record()
        torch.cuda.synchronize()
        category = _categories()
        sums = {}
        for tag, start, end in events:
            key = category.get(tag, tag)
            sums[key] = sums.get(key, 0.0) + start.elapsed_time(end)
        sums["total"] = total_start.elapsed_time(total_end)
        sums["n_chunks"] = sum(
            1 for tag, _, _ in events if category.get(tag) == "logits_gemm"
        )
        # How many device launches the call makes, so the unattributed
        # remainder can be checked against a per-launch gap.
        sums["n_bracketed"] = len(events)
        per_iter.append(sums)

    keys = {k for s in per_iter for k in s}
    result = {k: statistics.median([s.get(k, 0.0) for s in per_iter]) for k in keys}
    gemms = sum(result.get(k, 0.0) for k in CATEGORIES[:3])
    result["aux"] = result["total"] - gemms
    result["n_chunks"] = int(result["n_chunks"])
    result["batch_chunk_size"] = -(-N // result["n_chunks"]) if result["n_chunks"] else 0
    result.update({k: payload[k] for k in ("num_tokens", "in_features", "num_classes", "dtype", "native_stage")})
    print(json.dumps(result))


def _run(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lce_benchmark import native_stage_env

    import torch  # parent only: for the device slug
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    out = args.out or os.path.join("sweep_out", "data", gpu, "op_breakdown.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = [
        "num_tokens", "in_features", "num_classes", "dtype", "native_stage",
        "n_chunks", "batch_chunk_size", "total", "logits_gemm",
        "grad_input_gemm", "grad_weight_gemm", "aux", "aux_kernel", "n_bracketed",
        "error",
    ]
    write_header = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for N, D, V in SHAPES:
            for dtype in args.dtypes:
                for stage in STAGES:
                    payload = dict(
                        num_tokens=N, in_features=D, num_classes=V, dtype=dtype,
                        native_stage=stage, warmup=args.warmup, iters=args.iters,
                        seed=0,
                    )
                    proc = subprocess.run(
                        [sys.executable, os.path.abspath(__file__), "--worker",
                         json.dumps(payload)],
                        capture_output=True, text=True,
                        env=native_stage_env(os.environ, stage),
                    )
                    if proc.returncode != 0:
                        row = {**payload, "error": (proc.stderr.strip() or "failed")[-300:]}
                    else:
                        row = json.loads(proc.stdout.strip().splitlines()[-1])
                    w.writerow(row)
                    fh.flush()
                    print(f"N{N} D{D} V{V} {dtype:9} {stage:14}"
                          f" total={row.get('total', float('nan')):8.3f}"
                          f" aux={row.get('aux', float('nan')):8.3f}"
                          f" {row.get('error', '')}", flush=True)
    print(f"wrote {out}")


def _sweep_totals(data_dir):
    """Uninstrumented totals from the sweep CSVs, for the self-check."""
    import glob
    out = {}
    for f in glob.glob(os.path.join(data_dir, "N*_D*_V*.csv")):
        base = os.path.basename(f)[:-4]
        if "op_breakdown" in base:
            continue
        dtype = "bfloat16" if base.endswith("_bfloat16") else "float16"
        point = base.replace("_bfloat16", "").replace("_float16", "")
        for r in csv.DictReader(open(f)):
            if r.get("label") != "auto_fp32" or not r.get("time_ms"):
                continue
            out[(point, dtype, r["native_stage"])] = float(r["time_ms"])
    return out


def _analyze(args):
    rows = list(csv.DictReader(open(args.out)))
    sweep = _sweep_totals(os.path.dirname(os.path.abspath(args.out)))
    by = {}
    for r in rows:
        if r.get("error"):
            continue
        key = (int(r["num_tokens"]), int(r["in_features"]), int(r["num_classes"]), r["dtype"])
        by.setdefault(key, {})[r["native_stage"]] = r
    labels = [
        ("logits_gemm", "logits GEMM"),
        ("aux", "auxiliary"),
        ("grad_input_gemm", "grad_input GEMM"),
        ("grad_weight_gemm", "grad_weight GEMM"),
        ("total", "TOTAL"),
    ]
    summary = []
    for key in sorted(by, key=lambda k: (k[3], k[1], k[2], k[0])):
        arms = by[key]
        if set(STAGES) - set(arms):
            continue
        e, o = arms["off"], arms["fused_inplace"]
        point = f"N{key[0]}_D{key[1]}_V{key[2]}"
        # Self-check: this tool perturbs host-bound loops, so compare its own
        # totals with the sweep's uninstrumented ones before trusting a row.
        infl = {}
        for stage, row in (("off", e), ("fused_inplace", o)):
            ref = sweep.get((point, key[3], stage))
            if ref:
                infl[stage] = float(row["total"]) / ref
        flagged = any(abs(v - 1.0) > args.band for v in infl.values())
        note = ""
        if infl:
            note = ("  [instrument vs sweep: eager "
                    f"{infl.get('off', float('nan')):.2f}x, override "
                    f"{infl.get('fused_inplace', float('nan')):.2f}x"
                    f"{' -- FLAGGED, excluded from medians' if flagged else ''}]")
        print(f"\nN={key[0]} D={key[1]} V={key[2]} {key[3]}"
              f"  Bc={e['batch_chunk_size']} ({e['n_chunks']} chunks){note}")
        print(f"  {'operation':18} {'eager ms':>9} {'override ms':>12} {'speedup':>8} {'eager share':>12}")
        for field, label in labels:
            te, to = float(e[field]), float(o[field])
            share = te / float(e["total"])
            print(f"  {label:18} {te:9.3f} {to:12.3f} {te / to:8.3f} {share:11.1%}")
        if o.get("aux_kernel"):
            print(f"  {'(of which kernel)':18} {'':9} {float(o['aux_kernel']):12.3f}")
        f_aux = float(e["aux"]) / float(e["total"])
        s_aux = float(e["aux"]) / float(o["aux"])
        predicted = 1.0 / ((1 - f_aux) + f_aux / s_aux)
        measured = float(e["total"]) / float(o["total"])
        print(f"  Amdahl from the parts: {predicted:.3f}, measured total: {measured:.3f}")
        summary.append((key, measured, s_aux, predicted,
                        [float(e[g]) / float(o[g]) for g in CATEGORIES[:3]],
                        flagged))
    print("\n\nSummary -- speedup of the override over eager, per point")
    print(f"{'point':30} {'dt':9} {'total':>7} {'aux':>7} {'predicted':>10} {'logits':>7} {'gradin':>7} {'gradW':>7}  flag")
    for key, measured, s_aux, predicted, gemms, flagged in summary:
        pt = f"N{key[0]}_D{key[1]}_V{key[2]}"
        print(f"{pt:30} {key[3]:9} {measured:7.3f} {s_aux:7.3f} {predicted:10.3f}"
              f" {gemms[0]:7.3f} {gemms[1]:7.3f} {gemms[2]:7.3f}"
              f"  {'PERTURBED' if flagged else ''}")
    clean = [s for s in summary if not s[5]]
    print(f"\n{len(clean)} of {len(summary)} points within the {args.band:.0%} band;"
          f" medians below use only those")
    tot = [s[1] for s in clean]; aux = [s[2] for s in clean]
    allg = [g for s in clean for g in s[4]]
    print(f"\nmedian total {statistics.median(tot):.3f} (min {min(tot):.3f} max {max(tot):.3f})")
    print(f"median aux   {statistics.median(aux):.3f} (min {min(aux):.3f} max {max(aux):.3f})")
    print(f"GEMM control: median {statistics.median(allg):.3f}"
          f" (min {min(allg):.3f} max {max(allg):.3f}) over {len(allg)} measurements")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", default=None)
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16"])
    p.add_argument(
        "--shapes", nargs="+", default=None,
        help="restrict to these 'N,D,V' points (default: the sweep's 12)",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument(
        "--band", type=float, default=0.05,
        help="how far the instrumented total may sit from the sweep's total "
        "before the point is flagged as perturbed and dropped from medians",
    )
    args = p.parse_args()
    if args.shapes:
        wanted = {tuple(int(x) for x in s.split(",")) for s in args.shapes}
        SHAPES[:] = [s for s in SHAPES if s in wanted]
    if args.worker:
        _worker(json.loads(args.worker))
    elif args.analyze:
        _analyze(args)
    else:
        _run(args)

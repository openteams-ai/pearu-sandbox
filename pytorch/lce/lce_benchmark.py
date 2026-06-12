"""Inner benchmark for linear_cross_entropy.

Measures time, memory, and (optionally) gradient errors for a single
(num_tokens, in_features, num_classes, dtype, device, reduction, acc_policy,
chunking_method) point. Writes one CSV row per (acc_policy, chunking_method)
config plus a 'reference' row (non-chunked F.linear + F.cross_entropy) and,
when available on CUDA, a 'liger' row.

Used by lce_benchmark_sweep.py. Can also be invoked directly for a single
point.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Liger detection (optional dep)
# ---------------------------------------------------------------------------

def _liger_callable():
    try:
        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )
    except Exception:
        return None

    def _liger(input, linear_weight, target, linear_bias=None, reduction="mean"):
        # Positional order: _input, weight, target, bias, ce_weight,
        # ignore_index, lse_square_scale, label_smoothing, reduction.
        return LigerFusedLinearCrossEntropyFunction.apply(
            input, linear_weight, target, linear_bias, None, -100, 0.0, 0.0, reduction
        )

    return _liger


# ---------------------------------------------------------------------------
# CPU device name (best-effort, used to slugify CPU device subdirs)
# ---------------------------------------------------------------------------

def _cpu_device_name() -> str:
    # /proc/cpuinfo "model name" line
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# CUDA peak memory tracking (absolute)
# ---------------------------------------------------------------------------

class _CUDAPeakMonitor:
    """Records ``torch.cuda.max_memory_allocated()`` over the context body
    as an absolute peak, not a transient one. To make the number
    comparable across configurations, the caller must clear any in-flight
    gradients before entering so the peak isn't biased by warmup-populated
    state (which would otherwise bloat the caching-allocator baseline
    and let the in-context allocations slot into the freed blocks
    invisibly).
    """

    def __init__(self):
        self.peak = 0

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.peak = torch.cuda.max_memory_allocated()

    @property
    def peak_bytes(self) -> int:
        return self.peak


# ---------------------------------------------------------------------------
# CPU RSS peak monitor (total RSS, not transient)
# ---------------------------------------------------------------------------

class _RSSPeakMonitor:
    """Polls /proc/self/status VmHWM in a background thread."""

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.peak = 0
        self._stop = False
        self._thread = None

    def _read_rss_kb(self) -> int:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    def __enter__(self):
        import threading

        self.peak = self._read_rss_kb()
        self._stop = False

        def _loop():
            while not self._stop:
                cur = self._read_rss_kb()
                if cur > self.peak:
                    self.peak = cur
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def peak_bytes(self) -> int:
        return self.peak * 1024


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    label: str
    acc_policy: Optional[str]  # None for reference / liger
    chunking_method: Optional[str]
    use_chunked: bool = True
    acc_dtype: Optional[str] = None  # "float32" forces mixed-precision path


def _build_configs(dtype: str, include_acc_none: bool = False) -> list[Config]:
    """Build the config matrix.

    For fp16/bf16 input the default is to emit only the ``_fp32`` variants
    that pin ``acc_dtype=torch.float32`` (the realistic training setup,
    exercising the cuBLAS out_dtype= / explicit-cast mixed-precision path).
    Pass ``include_acc_none=True`` to also include the input-dtype-only
    variants (acc_dtype=None resolves to the input dtype).

    For fp32 input, acc_dtype=fp32 collapses to the input dtype, so only
    the non-fp32 variants are emitted regardless of ``include_acc_none``.
    """
    cfgs: list[Config] = [Config("reference", None, None, use_chunked=False)]
    fp32_acc = dtype in ("float16", "bfloat16")
    emit_acc_none = include_acc_none or not fp32_acc
    for policy in ("accurate", "balanced", "compact"):
        for cm in ("aspect_ratio", "aspect_ratio:2", "aspect_ratio:4"):
            if emit_acc_none:
                cfgs.append(Config(f"{policy}_{cm}", policy, cm))
            if fp32_acc:
                cfgs.append(
                    Config(f"{policy}_{cm}_fp32", policy, cm, acc_dtype="float32")
                )
    # chunking_method="auto" so the auto rows track the real zero-arg
    # default resolution, which is target-kind dependent since
    # pytorch/pytorch#187053 (prob targets on CUDA resolve to
    # aspect_ratio factor 1, index targets to :2) -- hardcoding ":2"
    # would silently measure a non-default config for prob targets.
    if emit_acc_none:
        cfgs.append(Config("auto", "auto", "auto"))
    if fp32_acc:
        cfgs.append(Config("auto_fp32", "auto", "auto", acc_dtype="float32"))
    return cfgs


# ---------------------------------------------------------------------------
# Reference path (full materialization)
# ---------------------------------------------------------------------------

def _reference_forward_backward(input, linear_weight, target, reduction, linear_bias=None):
    logits = F.linear(input, linear_weight, linear_bias)
    loss = F.cross_entropy(logits, target, reduction=reduction)
    if reduction == "none":
        loss.sum().backward()
    else:
        loss.backward()
    return loss


def _chunked_forward_backward(
    input, linear_weight, target, reduction, options, linear_bias=None
):
    loss = F.linear_cross_entropy(
        input, linear_weight, target, linear_bias=linear_bias,
        reduction=reduction, options=options,
    )
    if reduction == "none":
        loss.sum().backward()
    else:
        loss.backward()
    return loss


# ---------------------------------------------------------------------------
# Single-config measurement (runs in subprocess to isolate memory)
# ---------------------------------------------------------------------------

def _measure_in_subprocess(
    *,
    num_tokens: int,
    in_features: int,
    num_classes: int,
    dtype: str,
    device_type: str,
    reduction: str,
    prob_target: bool,
    bias: bool,
    config: Config,
    allow_retain_graph: bool,
    warmup: int,
    iters: int,
    grad_error_check: bool,
    seed: int,
) -> dict:
    payload = {
        "num_tokens": num_tokens,
        "in_features": in_features,
        "num_classes": num_classes,
        "dtype": dtype,
        "device_type": device_type,
        "reduction": reduction,
        "prob_target": prob_target,
        "bias": bias,
        "label": config.label,
        "acc_policy": config.acc_policy,
        "chunking_method": config.chunking_method,
        "use_chunked": config.use_chunked,
        "acc_dtype": config.acc_dtype,
        "allow_retain_graph": allow_retain_graph,
        "warmup": warmup,
        "iters": iters,
        "grad_error_check": grad_error_check,
        "seed": seed,
    }

    proc = subprocess.run(
        [sys.executable, __file__, "--worker", json.dumps(payload)],
        capture_output=True,
        text=True,
    )

    # Identifier fields that pin this row to its point even on subprocess crash.
    ident = {
        "label": config.label,
        "acc_policy": config.acc_policy,
        "chunking_method": config.chunking_method,
        "acc_dtype": config.acc_dtype,
        "num_tokens": num_tokens,
        "in_features": in_features,
        "num_classes": num_classes,
        "dtype": dtype,
        "device_type": device_type,
        "reduction": reduction,
        "prob_target": prob_target,
        "bias": bias,
        "allow_retain_graph": allow_retain_graph,
    }
    if proc.returncode != 0:
        return {
            **ident,
            "error": proc.stderr.strip() or proc.stdout.strip() or "subprocess failed",
        }
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        return {**ident, **parsed}
    except Exception as e:
        return {**ident, "error": f"parse failed: {e!r}\nstdout: {proc.stdout}"}


def _worker_main(payload: dict) -> None:
    """Runs inside the subprocess; prints a single JSON line on stdout."""
    torch.manual_seed(payload["seed"])

    device_type = payload["device_type"]
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[payload["dtype"]]
    device = torch.device("cuda") if device_type == "cuda" else torch.device("cpu")

    N = payload["num_tokens"]
    D = payload["in_features"]
    V = payload["num_classes"]
    reduction = payload["reduction"]
    label = payload["label"]

    input = torch.randn(N, D, dtype=dtype, device=device, requires_grad=True)
    linear_weight = torch.randn(V, D, dtype=dtype, device=device, requires_grad=True) * (1.0 / (D ** 0.5))
    linear_weight = linear_weight.detach().requires_grad_(True)
    prob_target = payload.get("prob_target", False)
    if prob_target:
        # Soft-label target: (N, V) probabilities at the input dtype
        # (chunked-path requirement). Note the target itself is an (N, V)
        # tensor, so it contributes to every config's memory floor.
        target = torch.softmax(torch.randn(N, V, dtype=dtype, device=device), dim=1)
    else:
        target = torch.randint(0, V, (N,), device=device)
    bias = payload.get("bias", False)
    linear_bias = (
        (torch.randn(V, dtype=dtype, device=device) * (1.0 / (D ** 0.5)))
        .detach()
        .requires_grad_(True)
        if bias
        else None
    )

    def _clear_grads():
        for t in (input, linear_weight, linear_bias):
            if t is not None and t.grad is not None:
                t.grad = None

    if label == "liger":
        liger = _liger_callable()
        if liger is None:
            print(json.dumps({"label": label, "error": "liger not installed"}))
            return

        def _liger_fwd_bwd():
            # Liger's Function.apply returns (loss, z_loss, token_accuracy);
            # we only need the loss.
            out = liger(input, linear_weight, target, linear_bias, reduction=reduction)
            loss = out[0] if isinstance(out, tuple) else out
            if reduction == "none":
                loss.sum().backward()
            else:
                loss.backward()
            return loss

        call = _liger_fwd_bwd
    elif payload["use_chunked"]:
        from torch.nn import LinearCrossEntropyOptions
        acc_dtype = dtype_map[payload["acc_dtype"]] if payload.get("acc_dtype") else None
        options = LinearCrossEntropyOptions(
            acc_policy=payload["acc_policy"],
            chunking_method=payload["chunking_method"],
            acc_dtype=acc_dtype,
            allow_retain_graph=payload["allow_retain_graph"],
        )
        call = lambda: _chunked_forward_backward(
            input, linear_weight, target, reduction, options, linear_bias
        )
    else:
        call = lambda: _reference_forward_backward(
            input, linear_weight, target, reduction, linear_bias
        )

    # Warmup
    for _ in range(payload["warmup"]):
        _clear_grads()
        call()
        if device_type == "cuda":
            torch.cuda.synchronize()

    # Optional grad-error check vs reference. Done before timing so its tensors
    # are freed before the timing-loop peak measurement. Skip with NaN if the
    # fp64 reference wouldn't fit in available VRAM, or if any OOM occurs.
    grad_input_error = float("nan")
    grad_linear_weight_error = float("nan")
    grad_linear_bias_error = float("nan")
    if payload["grad_error_check"]:
        try:
            if device_type == "cuda":
                free_bytes, _ = torch.cuda.mem_get_info()
                # Peak fp64 footprint of the reference call: ref_input
                # (N*D) + ref_weight (V*D) + logits (N*V) from F.linear +
                # grad_logits/softmax temp (N*V) from F.cross_entropy
                # backward + ref_input.grad (N*D) + ref_weight.grad (V*D),
                # times 8 (fp64). A probability target adds its own fp64
                # copy (N*V). 1.5x leaves headroom for small temps the
                # allocator may carry; tighter caused false skips on
                # small-V configs that would fit.
                nv_terms = 3 if prob_target else 2
                needed = (2 * N * D + 2 * V * D + nv_terms * N * V) * 8 * 1.5
                if needed > free_bytes:
                    raise torch.OutOfMemoryError(
                        f"grad-error check would need ~{needed / (1024 ** 3):.1f} GiB, "
                        f"have ~{free_bytes / (1024 ** 3):.1f} GiB free"
                    )

            # Run chunked call FIRST so its peak transient memory doesn't
            # overlap with the held fp64 reference grads.
            _clear_grads()
            call()
            gi = input.grad.detach().to(torch.float64) if input.grad is not None else None
            gw = linear_weight.grad.detach().to(torch.float64) if linear_weight.grad is not None else None
            gb = (
                linear_bias.grad.detach().to(torch.float64)
                if linear_bias is not None and linear_bias.grad is not None
                else None
            )

            ref_input = input.detach().to(torch.float64).requires_grad_(True)
            ref_weight = linear_weight.detach().to(torch.float64).requires_grad_(True)
            ref_bias = (
                linear_bias.detach().to(torch.float64).requires_grad_(True)
                if linear_bias is not None
                else None
            )
            # fp64 target for a probability target (lossless upcast);
            # index targets stay int64.
            ref_target = target.to(torch.float64) if prob_target else target
            ref_loss = F.cross_entropy(
                F.linear(ref_input, ref_weight, ref_bias), ref_target, reduction=reduction
            )
            if reduction == "none":
                ref_loss.sum().backward()
            else:
                ref_loss.backward()
            ref_gi = ref_input.grad
            ref_gw = ref_weight.grad
            ref_gb = ref_bias.grad if ref_bias is not None else None

            if gi is not None and ref_gi is not None:
                grad_input_error = float((gi - ref_gi).norm() / (ref_gi.norm() + 1e-30))
            if gw is not None and ref_gw is not None:
                grad_linear_weight_error = float((gw - ref_gw).norm() / (ref_gw.norm() + 1e-30))
            if gb is not None and ref_gb is not None:
                grad_linear_bias_error = float((gb - ref_gb).norm() / (ref_gb.norm() + 1e-30))

            # ref_target included: for a probability target it is an
            # (N, V) fp64 tensor that would otherwise stay alive (as a
            # function-scope local) through the memory phase, inflating
            # every config's measured peak by N*V*8 bytes.
            del ref_input, ref_weight, ref_bias, ref_target, ref_loss
            del ref_gi, ref_gw, ref_gb, gi, gw, gb
        except torch.OutOfMemoryError:
            pass
        finally:
            if device_type == "cuda":
                torch.cuda.empty_cache()

    # Time + memory. Memory and timing are measured in separate phases:
    # the memory phase runs one fresh iter under the peak monitor (no
    # event overhead, clean peak), the timing phase queues N iters back-
    # to-back with cuda.Event start/end and a single sync at the end
    # (matches the typical training loop overlap and gives GPU-side time).
    times_ms: list[float] = []
    memory_peak_bytes = 0
    timing_error: Optional[str] = None

    if device_type == "cuda":
        # Memory phase: clear warmup-populated grads BEFORE entering the
        # monitor so the peak isn't biased by warmup-cached blocks the
        # in-call allocations would invisibly slot into.
        try:
            _clear_grads()
            with _CUDAPeakMonitor() as mem:
                call()
            memory_peak_bytes = mem.peak_bytes
        except torch.OutOfMemoryError as e:
            timing_error = f"OOM in memory phase: {e}"
            torch.cuda.empty_cache()

        # Timing phase: cuda.Event per iter (no per-iter sync; one sync
        # after the loop). Mirrors the older benchmark.py methodology but
        # keeps per-iter samples so median aggregation works.
        if timing_error is None:
            events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            try:
                for _ in range(payload["iters"]):
                    _clear_grads()
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record()
                    call()
                    end_ev.record()
                    events.append((start_ev, end_ev))
            except torch.OutOfMemoryError as e:
                timing_error = f"OOM in timing phase: {e}"
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for s, e in events:
                times_ms.append(s.elapsed_time(e))
    else:
        try:
            gc.collect()
            with _RSSPeakMonitor() as mem:
                for _ in range(payload["iters"]):
                    _clear_grads()
                    t0 = time.perf_counter()
                    call()
                    times_ms.append(1000.0 * (time.perf_counter() - t0))
            memory_peak_bytes = mem.peak_bytes
        except torch.OutOfMemoryError as e:
            timing_error = f"OOM: {e}"

    if times_ms:
        times_ms.sort()
        median = times_ms[len(times_ms) // 2]
    else:
        median = float("nan")
    if memory_peak_bytes == 0 and timing_error is not None:
        memory_peak_bytes_out: float = float("nan")
    else:
        memory_peak_bytes_out = memory_peak_bytes

    result = {
        "label": label,
        "acc_policy": payload["acc_policy"],
        "chunking_method": payload["chunking_method"],
        "acc_dtype": payload.get("acc_dtype"),
        "time_ms": median,
        "memory_peak_bytes": memory_peak_bytes_out,
        "grad_input_error": grad_input_error,
        "grad_linear_weight_error": grad_linear_weight_error,
        "grad_linear_bias_error": grad_linear_bias_error,
        "num_tokens": N,
        "in_features": D,
        "num_classes": V,
        "dtype": payload["dtype"],
        "device_type": device_type,
        "reduction": reduction,
        "prob_target": prob_target,
        "bias": bias,
        "allow_retain_graph": payload["allow_retain_graph"],
    }
    if timing_error is not None:
        result["timing_error"] = timing_error
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", default=None, help="(internal) run a single config")
    p.add_argument("--num-tokens", type=int, default=4096)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--num-classes", type=int, default=32000)
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32", "both"), default="bfloat16")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--reduction", choices=("mean", "sum", "none"), default="mean")
    p.add_argument(
        "--prob-target",
        action="store_true",
        help="use a probability (soft-label) target of shape (N, V) at the "
        "input dtype instead of class indices. Requires reduction mean/sum "
        "(the chunked path falls back on none). The liger baseline is "
        "skipped (index targets only).",
    )
    p.add_argument(
        "--bias",
        action="store_true",
        help="add a linear_bias of shape (num_classes,); exercised by the "
        "chunked path, the reference, and liger (its fused op takes a "
        "bias). Adds a grad_linear_bias_error column to the grad-error "
        "check.",
    )
    p.add_argument("--allow-retain-graph", action="store_true")
    p.add_argument(
        "--include-acc-none",
        action="store_true",
        help="for fp16/bf16 input, also include configs with acc_dtype=None "
        "(the input-dtype-only path). By default only the _fp32 variants "
        "(acc_dtype=torch.float32, the realistic training path) are emitted.",
    )
    p.add_argument("--no-grad-error-check", action="store_true")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="CSV file to append rows to")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.worker is not None:
        _worker_main(json.loads(args.worker))
        return 0

    if args.device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = args.device

    if args.dtype == "both":
        dtypes = ["float16", "bfloat16"]
    else:
        dtypes = [args.dtype]

    if args.prob_target and args.reduction == "none":
        print(
            "error: --prob-target requires --reduction mean/sum (the chunked "
            "path falls back to the reference on none, so the benchmark "
            "would silently measure the reference)",
            file=sys.stderr,
        )
        return 1

    # Liger's fused linear cross entropy supports index targets only.
    has_liger = (
        device_type == "cuda"
        and not args.prob_target
        and _liger_callable() is not None
    )

    rows: list[dict] = []
    for dtype in dtypes:
        configs = _build_configs(dtype, include_acc_none=args.include_acc_none)
        if has_liger:
            configs.append(Config("liger", None, None, use_chunked=False))
        for cfg in configs:
            row = _measure_in_subprocess(
                num_tokens=args.num_tokens,
                in_features=args.in_features,
                num_classes=args.num_classes,
                dtype=dtype,
                device_type=device_type,
                reduction=args.reduction,
                prob_target=args.prob_target,
                bias=args.bias,
                config=cfg,
                allow_retain_graph=args.allow_retain_graph,
                warmup=args.warmup,
                iters=args.iters,
                grad_error_check=not args.no_grad_error_check,
                seed=args.seed,
            )
            rows.append(row)
            print(json.dumps(row))

    if args.out is not None:
        if not rows:
            return 0
        # Canonical column list so every CSV has consistent headers regardless
        # of which configs failed at this point. Unknown keys are dropped via
        # extrasaction='ignore'.
        keys = [
            "label",
            "acc_policy",
            "chunking_method",
            "acc_dtype",
            "num_tokens",
            "in_features",
            "num_classes",
            "dtype",
            "device_type",
            "reduction",
            "prob_target",
            "bias",
            "allow_retain_graph",
            "time_ms",
            "memory_peak_bytes",
            "grad_input_error",
            "grad_linear_weight_error",
            "grad_linear_bias_error",
            "error",
            "timing_error",
        ]
        write_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if write_header:
                w.writeheader()
            for row in rows:
                w.writerow(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())

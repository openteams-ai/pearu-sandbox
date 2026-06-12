# lce — linear_cross_entropy benchmark scripts

Benchmarks for the chunked `nn.functional.linear_cross_entropy` /
`nn.LinearCrossEntropyLoss` introduced in pytorch/pytorch#181574.
Scripts are CWD-independent (paths resolve via `__file__`). PyTorch must be
importable from the active venv, ideally as an editable install of a build
that includes the PR. Liger Kernel is optional; the `liger` baseline is
emitted automatically when `liger_kernel` is installed and the device is
CUDA.

## Scripts

- **`lce_benchmark.py`** — inner benchmark. Runs the full config matrix
  for a single `(num_tokens, in_features, num_classes, dtype, device,
  reduction)` point in subprocess-isolated workers and writes one CSV.
  Each row covers one config; columns include `label`, `acc_policy`,
  `chunking_method`, `acc_dtype`, `time_ms`, `memory_peak_bytes`,
  `grad_input_error`, `grad_linear_weight_error`.
- **`lce_benchmark_sweep.py`** — sweep driver. Iterates the three axes
  (`num_tokens`, `in_features`, `num_classes`) at device-specific defaults
  and dispatches `lce_benchmark.py` once per `(point, dtype)`. CSVs land
  under `sweep_out/data/<device-slug>/`. Produces a 4-row figure per
  `(dtype, device)` group when the run finishes.
- **`plot_auto_dispatch.py`** — focused plotter. Renders 3-row figures
  showing `reference`, `liger`, `auto_fp32`, and the aspect-ratio
  neighbors `auto_fp32` resolved to. Reads CSVs produced by the sweep
  driver.

## Config matrix

For each `(point, dtype)` the inner script emits:

- `reference` — full-materialization `linear` + `cross_entropy`.
- `liger` — Liger Kernel's fused op, on CUDA only, if installed.
- For each `policy in {accurate, balanced, compact}` and each
  `aspect_ratio` factor in `{1, 2, 4}`: a chunked op invocation.
- `auto` — `acc_policy="auto"`, `chunking_method="aspect_ratio:2"`.

For fp16/bf16 input, the **default emits only the `_fp32` variants**
(`acc_dtype=torch.float32`) — the realistic training path that exercises
cuBLAS `out_dtype=` / explicit-cast mixed precision. Pass
`--include-acc-none` on the benchmark scripts to additionally emit the
input-dtype-only variants, and on the plot scripts to render them.

For fp32 input, `acc_dtype=fp32` is redundant; only the non-fp32 variants
are emitted.

`reference` and `liger` carry no `acc_dtype` and are always emitted /
always plotted.

## Usage

A100 (or any SM80+ Ampere/Hopper):

```bash
python lce_benchmark_sweep.py --dtypes bfloat16 --device cuda
```

Both dtypes, plus the input-dtype-only variants:

```bash
python lce_benchmark_sweep.py --dtypes float16 bfloat16 --device cuda --include-acc-none
```

Restrict to one axis (cheaper smoke test):

```bash
python lce_benchmark_sweep.py --dtypes bfloat16 --axes num_classes
```

`reduction='none'` (per-sample loss). The chunked none path is a separate
op that recomputes the chunked grads in backward (one extra logits matmul
per chunk) since the per-sample loss makes ``grad_output`` a vector that
cannot be precomputed in forward -- so expect higher time than `mean`/`sum`
at a similar memory profile. CSVs are written to separate
`*_reduction-none.csv` files and plotted to separate `*_none.png` figures,
so a none sweep never clobbers or mixes with the `mean` results:

```bash
python lce_benchmark_sweep.py --dtypes bfloat16 --device cuda --reduction none
```

Probability (soft-label) targets (`--prob-target`). The target is a
`(num_tokens, num_classes)` softmax tensor at the input dtype; the
chunked path supports this for `mean`/`sum` (pytorch/pytorch#187053).
The `liger` baseline is skipped (its fused op takes index targets
only), so comparisons are vs `reference`. Note the target itself is an
`(N, V)` tensor and contributes to every config's memory floor — expect
smaller relative savings than the index-target sweeps. CSVs/figures get
`prob-target` / `_prob` suffixes:

```bash
python lce_benchmark_sweep.py --dtypes bfloat16 --device cuda --prob-target
```

`linear_bias` (`--bias`). Adds a `(num_classes,)` bias consumed by the
chunked path, the reference, and liger (its fused op takes a bias), plus
a `grad_linear_bias_error` column in the grad-error check (rendered as a
third metric in the auto-dispatch grad-error row). Composes with
`--prob-target` and `--reduction`; CSVs/figures get `bias` / `_bias`
suffixes:

```bash
python lce_benchmark_sweep.py --dtypes bfloat16 --device cuda --bias
```

Override the swept values for one or more axes (e.g. push num_classes
into LLaMA-3 / Qwen3 territory):

```bash
python lce_benchmark_sweep.py --dtypes bfloat16 \
    --num-classes 4096 8192 16384 32000 65536 128000 256000
```

Same idea for `--num-tokens` and `--in-features`. Omitted axes fall back
to the device-specific defaults in `CUDA_AXES` / `CPU_AXES`.

Re-run points whose CSVs already exist:

```bash
python lce_benchmark_sweep.py --force ...
```

Re-plot existing CSVs without re-running:

```bash
python lce_benchmark_sweep.py --plot-only ...
```

Focused auto-dispatch plots (fp32-acc variants + `reference` + `liger`):

```bash
python plot_auto_dispatch.py
```

Add the non-fp32 variants to either plotter:

```bash
python plot_auto_dispatch.py --include-acc-none
python lce_benchmark_sweep.py --plot-only --include-acc-none ...
```

## Notes

- **bf16 + `acc_dtype=fp32` requires SM80+** (cuBLAS out_dtype= path).
  On Turing (CC 7.5, e.g. RTX 2060) these configs error and produce NaN
  rows; the inner script captures and continues. fp16 + `acc_dtype=fp32`
  works since Volta (SM 7.0).
- **CUDA memory metric is `torch.cuda.max_memory_allocated()`** measured
  from a clean-slate state (warmup-populated grads cleared, allocator
  cache flushed before entering the monitor). This is the absolute peak
  GPU memory needed to execute the workload, including the persistent
  input + linear_weight + target tensors — directly comparable across
  configs.
- **CUDA timing uses `torch.cuda.Event(enable_timing=True)`** with one
  start/end event pair per iter. Iters are queued back-to-back with a
  single `synchronize()` after the loop; per-iter elapsed times come
  from `start.elapsed_time(end)` (GPU-side, no CPU jitter). Median of
  N iters is reported. Memory and timing run in separate phases so the
  event overhead doesn't bias the peak.
- Grad-error rows compare the chunked op against an fp64 reference
  jacobian. The reference is budget-checked against free VRAM and
  **skipped (NaN) when it would not fit** — common at the largest swept
  configs on shared GPUs. Timing/memory data still collected.
- All workers run in subprocesses so a single config's OOM does not
  affect later configs at the same point.
- CSV column schema is canonical (same columns regardless of which
  configs succeeded at a point) so plotting handles mixed-success CSVs.

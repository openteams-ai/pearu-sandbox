# Budget-regime chunking heuristic: a throughput-proven, SM-scaled chunk size

## Summary

In the budget regime (`in_features >= num_classes`, above the aspect_ratio crossing) the throughput-optimal chunk size is a per-architecture constant `k * SM_count`, rounded to a power of two, applied as a CAP on `aspect_ratio` (`B = min(aspect_ratio_B, round_pow2(k * SM_count))`).
`k` is a constant set by the GPU's tensor-core generation -- not by shape, not by memory bandwidth/capacity, not by SKU.
Measured across five GPUs: `k ~= 14` for Ampere/Hopper and `~= 17.4` for Blackwell at a 5%-off-peak throughput tolerance (`~= 23` and `~= 30` at 3%).
This replaces the `eps=1` budget heuristic (which sizes `B` proportional to `in_features`), whose functional form the data contradicts.

## Why throughput, and why a fresh heuristic

`aspect_ratio` is liger's **memory** heuristic: it sizes each chunk so the `(chunk, num_classes)` logits buffer matches the `(num_tokens, in_features)` input footprint, bounding the extra memory of the logits materialization to about one input.
Source: liger-kernel `src/liger_kernel/ops/fused_linear_cross_entropy.py`, chunk-size derivation comment -- "the increase in memory = BT x V ... reduction can be achieved by partitioning ... to achieve the same memory consumption as BT x H" -- with `inc_factor = ceil(V/H)`, `chunk_size = next_pow2(ceil(BT/inc_factor))`; present since liger's initial commit (`45055fe`, 2024-08-06).
The chunk-size value itself is derived purely from this memory match: neither the code nor the Liger-Kernel paper presents it as throughput-tuned or empirically validated for a specific regime (the paper motivates chunking by the logit-materialization memory bottleneck and credits its throughput gains to fusing each chunk under `torch.compile`, not to the chunk size).
It targets the vocab-head regime (`num_classes >> in_features`), where the `BT x V` logits dwarf the input and chunking saves the most.

In the budget regime (`in_features >= num_classes`) liger's formula degenerates: `inc_factor = ceil(V/H) = 1`, so `chunk_size = next_pow2(BT)` = a single chunk (no chunking at all).
So liger provides no heuristic here, and the materialization it guards against is no longer the dominant term -- chunking's memory purpose lapses.
Lacking an established chunk size, we optimize throughput directly (measured), treating memory as a checked constraint rather than the objective: at large `num_tokens` (the real LLM regime) the throughput-optimal chunk already sits at or below the reference peak, so throughput is the binding concern in this regime.

## Method

`chunk_size_sweep.py` drives the chunked op with an explicit `batch_chunk_size` (`chunking_method=None`, the landed #187219 API), so it is independent of the budget code and runs on plain `main`.
For each shape it sweeps `B` over powers of two, measures fwd+bwd median time and peak memory in isolated subprocesses, and extracts `B_knee` = the smallest `B` whose time is within `tol` of the best (the throughput plateau).
All GPUs were built from the same pinned `main` commit `e01c08cc020a946304debefe587c67c4e1694a27` so only hardware differs.

## Finding

`B_knee` scales linearly with SM count and is flat across shapes (it does not track `N`, `D`, or `V`); the large-`N` refinement confirmed it plateaus rather than growing with `N`.
Two controls isolate the cause: H100 vs H200 (same compute, faster HBM3e) give bit-identical constants, and B200 vs B300 (same Blackwell gen, different SKU/memory) also give bit-identical constants.
So `k` is determined by tensor-core generation alone.

Per-device `B/SM` (tol 0.05): A100 14.09, H100 14.05, H200 14.05, B200 17.43, B300 17.43.
Two classes: {Ampere, Hopper} `k ~= 14`, {Blackwell} `k ~= 17.4` (~1.24x), stable across tolerances (the ratio is ~1.25-1.30x at 3%).

## Mechanism (profiling)

The op is GEMM-dominated: three matmuls -- forward logits (`mm`), grad_input (`addmm`), grad_weight (`addmm_`) -- carry ~70-80% of CUDA time, backward ~2x forward.
GEMM share falls as tensor cores get faster (A100 ~80% -> B300 ~70%), so the memory-bound softmax/elementwise grows on newer hardware.
The forward and grad_input GEMMs have `B` as their M-dimension and saturate once M crosses the tensor-core tile/wave threshold -- which scales with SM count and grows with the arch's tile size, explaining both the SM-linearity and the higher Blackwell `k`.
The grad_weight accumulation (`addmm_`, contraction dim `B`) re-streams the `(num_classes, in_features)` accumulator once per chunk, so its cost grows as `B` shrinks; it is the long pole below the knee and the prime target if we ever want to raise the plateau.

## Why not the alternatives

`eps=1` sizes `B` proportional to `in_features` (`B = input_footprint / per_row`); the data shows `B` is independent of `in_features`.
A reference-memory cap would size `B` proportional to `N*V/D`; that model fit the data worst of all candidates.
Both optimize memory, which is the wrong objective here.

## Recommended heuristic

On CUDA, resolve `auto` to a concrete `batch_chunk_size`:

```
B = clamp( min(aspect_ratio_B, round_pow2(k * SM_count)), 1, num_batches )
```

where `aspect_ratio_B` is the size auto's `aspect_ratio` variant would produce and `round_pow2(k * SM_count)` is the throughput-saturation chunk.
`k * SM_count` is a CAP, not a target: where `aspect_ratio_B <= k*SM` (the vocab-head / LLM region) the cap is inert and `aspect_ratio` is used unchanged; where `aspect_ratio_B > k*SM` (near and above the crossing, or huge-batch / moderate-vocab) it caps the chunk at the saturation point.
This subsumes the crossing switch and is continuous in the shape.
Against uncapped `aspect_ratio` the cap trades at most the chosen tolerance in throughput for substantial memory: read off the existing per-GPU sweeps (`validate_cap.py`), the cap at tol 0.05 runs ~2-5% slower while using 12-44% less memory, and the memory savings grow with `num_tokens` (up to ~44% at N=65536). Tighten the tolerance for strictly-zero throughput loss at smaller memory savings.
We do not have a prediction for whether the cap engages in the LLM region; the `auto`-vs-liger memory plots will show it -- where our capped chunk diverges from liger's (uncapped `aspect_ratio`) chunk, the cap bit.

`SM_count` from `torch.cuda.get_device_properties(dev).multi_processor_count`; `k` from a small table keyed by compute-capability major (Ampere/Hopper one value, Blackwell ~1.25x).
Round to a power of two (consistency with `aspect_ratio`'s `next_pow2`; GEMM tiles favor it; the plateau is flat so rounding is free).
`k` encodes the throughput/memory tolerance (tighter tolerance -> larger `k` -> closer to peak, more memory); quote/measure it at `tol >= 0.05`, since the 3% band is noise-sensitive at the power-of-two boundary.
Reference values: `k ~= 14` (Ampere/Hopper) / `~= 17.4` (Blackwell) at tol 0.05; `~= 23` / `~= 30` at tol 0.03.

Non-CUDA: no cap -- `B = aspect_ratio_B`, which degenerates to a single chunk above the crossing (accepted; see Device scope).

### Memory dial: `auto:M`

`k * SM_count` is throughput-optimal only when the resulting chunk fits in available memory; on a high-SM / low-VRAM device with large `in_features` the per-chunk buffer (`~ k*SM * in_features * acc_bytes`) can OOM, and then a smaller chunk (more chunks, lower throughput) is strictly preferable to failing.
The op cannot reliably know available memory at resolution time (free memory is not peak headroom, and it races with other allocations), so the memory/throughput trade is exposed as an explicit user dial rather than auto-detected:

```
auto:M  ->  B = clamp( round_pow2( min(aspect_ratio_B, k*SM) / M ), 1, num_batches )
```

`auto` == `auto:1` is max throughput; `M > 1` gives an ~M-times smaller chunk (~M-times less per-chunk buffer, more chunks, lower throughput).
The user raises `M` until the op fits, getting the best throughput achievable at that memory level.
`M` mirrors the existing `aspect_ratio:N` divisor and applies to the final resolved `B` uniformly, so it also relieves OOM in the LLM region.
Ship `auto:M` only once a benchmark shows `k*SM` actually OOMs somewhere (the 80 GB cards measured so far have ample headroom); reserve the grammar now so adding it later is non-breaking.
`batch_chunk_size` remains the absolute-size escape hatch.

## Recommendation for #187271

Replace the `eps`/threshold logic with the `k * SM_count` rule above, or retire #187271 in favor of it.
The budget regime is outside the LLM-typical range, so there is no urgency; the throughput-proven heuristic is the better long-term answer and could supersede the budget method entirely.

## Device scope and terminology

The `k * SM_count` rule is CUDA-specific and was measured only on CUDA bf16; it slots into the existing CUDA branch of the `auto` resolver, and non-CUDA devices keep their current `aspect_ratio` default unchanged (no regression).
`B_knee` is shape-independent and tracks hardware parallelism, so it cannot be faithfully re-expressed as a function of `N`, `D`, or `V` -- the shape-form fits (`c*N`, `c*V`, `c*N*V/D`) were the worst candidates; the only honest generalization replaces `SM_count` with a device's parallel-unit/locality measure, which is hardware, not shape, and must be measured per backend.
Correctness does not depend on the choice: chunk size is invariant up to floating-point rounding (the fp64 invariance test), so a non-optimal chunk size on a non-CUDA device costs throughput only, never accuracy.
ROCm/AMD transfers the form -- HIP exposes `multi_processor_count` through the `torch.cuda` API -- but needs its own measured `k` (CDNA matrix cores differ; do not reuse the NVIDIA values); XPU and MPS expose different descriptors and each need their own measurement.
CPU is expected to follow a different mechanism (chunk-size optimum is cache-residency-bound, not wave-saturation-bound), so `k * cores` is not assumed; CPU and other backends stay on `aspect_ratio` until there is demand and a measured result.

Terminology: "budget" named the `eps=1` memory-budget approach; the throughput-saturation rule is not a budget, so the CUDA pick should fold into `auto` (or be renamed, e.g. `sm_saturation`) rather than be surfaced as a user-facing `budget` method.

## Caveats and future work

Measured for bf16 only; fp16 and fp32-accumulation paths may use different GEMM tiles and so a different `k` -- measure before assuming.
The shape grid is moderate; the SM-linearity and the two-class result are robust, but a new architecture should be measured (one sweep) rather than extrapolated.
A future `k` could be derived from queryable hardware (tile size * SM count) instead of an arch table, but the table is simpler and the set of architectures is small.

## Data and reproduction

Sweep + analysis: `chunk_size_sweep.py`; per-GPU data: `chunk_size_sweep_{a100,h100,h200,b200,b300}.csv`; summary figure: `chunk_size_summary.png` via `plot_chunk_summary.py`; run instructions: `RUN.md`.

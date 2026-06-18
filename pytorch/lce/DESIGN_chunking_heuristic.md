# Budget-regime chunking heuristic: a throughput-proven, SM-scaled chunk size

## Summary

In the budget regime (`in_features >= num_classes`, above the aspect_ratio crossing) the throughput-optimal chunk size is `B = k * SM_count`, rounded to a power of two.
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

```
B = clamp(round_pow2(k * SM_count), 1, num_batches)
```

`SM_count` from `torch.cuda.get_device_properties(dev).multi_processor_count` at runtime.
`k` from a small table keyed by compute-capability major: Ampere/Hopper one value, Blackwell ~1.25x.
Round to a power of two for consistency with `aspect_ratio`'s `next_pow2` and because GEMM tiles favor it; the plateau is flat so rounding is free.
`k` is the throughput/memory knob (tighter tolerance -> larger `k` -> closer to peak throughput, more memory): suggested defaults near the 3-5% band, e.g. `k ~= 16` (Ampere/Hopper) and `~= 20` (Blackwell) for a balanced point, or `~= 24` / `~= 30` to favor throughput.
Pick one tolerance as a one-time policy decision; quote/measure `k` at `tol >= 0.05`, since the 3% band is noise-sensitive at the power-of-two boundary.

## Recommendation for #187271

Replace the `eps`/threshold logic with the `k * SM_count` rule above, or retire #187271 in favor of it.
The budget regime is outside the LLM-typical range, so there is no urgency; the throughput-proven heuristic is the better long-term answer and could supersede the budget method entirely.

## Caveats and future work

Measured for bf16 only; fp16 and fp32-accumulation paths may use different GEMM tiles and so a different `k` -- measure before assuming.
The shape grid is moderate; the SM-linearity and the two-class result are robust, but a new architecture should be measured (one sweep) rather than extrapolated.
A future `k` could be derived from queryable hardware (tile size * SM count) instead of an arch table, but the table is simpler and the set of architectures is small.

## Data and reproduction

Sweep + analysis: `chunk_size_sweep.py`; per-GPU data: `chunk_size_sweep_{a100,h100,h200,b200,b300}.csv`; summary figure: `chunk_size_summary.png` via `plot_chunk_summary.py`; run instructions: `RUN.md`.

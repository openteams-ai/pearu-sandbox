# Budget-regime chunking heuristic: a throughput-proven, SM-scaled chunk size

## Summary

The objective is to maximize throughput subject to chunked peak memory not exceeding the unchunked reference (so chunking is never an OOM/memory regression). The shipped rule is a single device-independent memory cap on `aspect_ratio`:

```
B = clamp( min( aspect_ratio_B,  floor_tile( N*V / (4*D) ) ), 1, num_batches )
```

`N*V/(4*D)` is `B_ref`, the largest chunk whose peak memory stays at or below reference, fit empirically across five GPUs as `0.251 * N*V/D` (log2 RMSE 0.011, identical on every GPU -- memory is device-independent). `floor_tile` rounds down to a GEMM tile multiple (~128/256) rather than a power of two (see below).
The journey first established the *throughput* ceiling -- the throughput-saturating chunk is a per-arch constant `k * SM_count` (`k ~= 14` Ampere/Hopper, `~= 17.4` Blackwell at tol 0.05) -- but in the budget regime `B_ref < k*SM` (memory binds before saturation), so the memory cap is what ships; `k*SM` explains why bigger chunks stop helping and is an optional secondary cap (memory-minimization near the crossing).
This replaces the `eps=1` budget heuristic (which sizes `B` proportional to `in_features`), whose functional form the data contradicts.

## Why throughput, and why a fresh heuristic

`aspect_ratio` is liger's **memory** heuristic: it sizes each chunk so the `(chunk, num_classes)` logits buffer matches the `(num_tokens, in_features)` input footprint, bounding the extra memory of the logits materialization to about one input.
Source: liger-kernel `src/liger_kernel/ops/fused_linear_cross_entropy.py`, chunk-size derivation comment -- "the increase in memory = BT x V ... reduction can be achieved by partitioning ... to achieve the same memory consumption as BT x H" -- with `inc_factor = ceil(V/H)`, `chunk_size = next_pow2(ceil(BT/inc_factor))`; present since liger's initial commit (`45055fe`, 2024-08-06).
The chunk-size value itself is derived purely from this memory match: neither the code nor the Liger-Kernel paper presents it as throughput-tuned or empirically validated for a specific regime (the paper motivates chunking by the logit-materialization memory bottleneck and credits its throughput gains to fusing each chunk under `torch.compile`, not to the chunk size).
It targets the vocab-head regime (`num_classes >> in_features`), where the `BT x V` logits dwarf the input and chunking saves the most.

In the budget regime (`in_features >= num_classes`) liger's formula degenerates: `inc_factor = ceil(V/H) = 1`, so `chunk_size = next_pow2(BT)` = a single chunk (no chunking at all).
So liger provides no heuristic here, and the materialization it guards against is no longer the dominant term -- chunking's memory purpose lapses.
Lacking an established chunk size, we maximize throughput subject to a hard memory constraint -- chunked peak must not exceed the unchunked reference (chunking must never be a memory regression). The throughput-optimal chunk (`k*SM`) does exceed reference deep in the budget regime, so the binding term there is the memory cap `B_ref`, and the chunk runs below peak throughput to stay under reference (the price of OOM-safety). Where memory is not binding (near the crossing / large vocab) throughput dominates.

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

`eps=1` sizes `B` proportional to `in_features`; the throughput-saturating chunk is shape-independent (`k*SM`), so `eps=1` is the wrong form for throughput.
A reference-memory cap sizes `B` proportional to `N*V/D` -- which fit the *throughput* knee worst of all candidates, confirming it is not a throughput heuristic. But that is exactly the right form for the *memory* bound: `B_ref = N*V/(4*D)` is where chunked peak meets reference (below). The lesson is that throughput and memory want different forms; the shipped rule uses each for its own job (`k*SM` characterizes the throughput ceiling, `B_ref` is the binding memory cap).

## Recommended heuristic

Resolve `auto` to a concrete `batch_chunk_size`:

```
B = clamp( min( aspect_ratio_B,  floor_tile( N*V / (4*D) ) ), 1, num_batches )
```

`N*V/(4*D)` is `B_ref`, the largest chunk whose peak memory stays at or below reference. It is fit empirically (`validate_cap.py`) as `0.251 * N*V/D`, **log2 RMSE 0.011, identical on all five GPUs** -- memory is device-independent, so unlike the throughput `k` this needs no per-arch table. It reads intuitively: `B_ref = N/4` at the crossing (`D=V`) and halves each time `in_features` doubles. The `min` with `aspect_ratio_B` makes it inert in the LLM region (there `aspect_ratio_B << B_ref`), so `aspect_ratio`/liger behavior is unchanged where it is validated.

The `0.25` is empirical: the naive analytic guess is `~N*V/D` (coefficient 1), 4x too large, because the op carries buffers beyond `V*logits + D*acc` (input-acc copy, upcast scratch, precompute persistent). Round **down** so the cap is always `<= reference`. `0.25` is for bf16 + `compact`; the byte composition shifts with dtype / acc-policy, so re-fit `c` per (dtype, policy) before assuming it (still device-independent).

`floor_tile` rounds down to a GEMM tile multiple (~128/256), not a power of two: power-of-two flooring can nearly halve `B` (e.g. 1900 -> 1024) and waste throughput for no memory reason, whereas tile-flooring (1900 -> 1792) keeps `B` near `B_ref` while still `<= reference`. Power-of-two is a safe conservative default, but tile-alignment is the real GEMM requirement -- confirm with a tile-aligned non-power-of-two sub-sweep before relying on it (we only measured powers of two).

The throughput ceiling `k*SM` is not the shipped cap: in the budget regime `B_ref < k*SM` (to stay under reference you run below saturation -- the throughput you pay for OOM-safety). `k*SM` is retained only as an optional secondary cap that minimizes memory near the crossing for very large N (where `B_ref > k*SM`); the stated objective ("<= reference") does not require it.

Non-CUDA: `B = min(aspect_ratio_B, floor_tile(N*V/4D))` works unchanged -- `B_ref` is device-independent, no SM query needed.

### Memory dial: `auto:M`

`B_ref` keeps chunked at or below reference, but reference itself can be the OOM ceiling, or other allocations may leave less headroom; on a memory-constrained device the user may need to chunk finer still. The op cannot reliably know available memory at resolution time (free memory is not peak headroom, and it races with other allocations), so this is exposed as an explicit user dial rather than auto-detected:

```
auto:M  ->  B = clamp( floor_tile( min(aspect_ratio_B, N*V/4D) / M ), 1, num_batches )
```

`M` need not be a power of two: since throughput is monotonic in `B`, the largest tile multiple that fits (e.g. `M=4/3`, `B -> 3B/4`) beats coarse halving (`M=2`, `B -> B/2`) -- it avoids OOM while keeping more throughput. So `M` is a continuous-ish divisor with `floor_tile` applied, not restricted to powers of two.

`auto` == `auto:1` is max throughput; `M > 1` gives an ~M-times smaller chunk (~M-times less per-chunk buffer, more chunks, lower throughput).
The user raises `M` until the op fits, getting the best throughput achievable at that memory level.
`M` mirrors the existing `aspect_ratio:N` divisor and applies to the final resolved `B` uniformly, so it also relieves OOM in the LLM region.
Ship `auto:M` only once a memory-constrained run shows the `B_ref`-capped chunk can still OOM (i.e. reference itself does not fit, or co-allocations eat the headroom); reserve the grammar now so adding it later is non-breaking.
`batch_chunk_size` remains the absolute-size escape hatch.

## Recommendation for #187271

Replace the `eps`/threshold logic with the `min(aspect_ratio_B, floor_tile(N*V/4D))` rule above, or retire #187271 in favor of it.
The budget regime is outside the LLM-typical range, so there is no urgency; the memory-bounded rule (with `k*SM` as the throughput rationale) is the better long-term answer and could supersede the budget method entirely.

## Device scope and terminology

The shipped rule -- `min(aspect_ratio_B, floor_tile(N*V/4D))` -- is **device-independent**: it needs no SM query and applies on every backend (the `B_ref` coefficient is the same across all five GPUs measured, since memory footprint is the same arithmetic everywhere). So non-CUDA devices get the same memory cap; this is not a regression (it only ever shrinks `aspect_ratio` to stay under reference).
The `k*SM` throughput ceiling, in contrast, *is* per-architecture and CUDA-specific (measured only on CUDA bf16). It is not the shipped cap; it is the rationale for why bigger chunks stop helping, and an optional memory-minimizer near the crossing. Were it used as a cap, it would need a per-arch `k` table -- ROCm transfers the form via `multi_processor_count` but with its own `k` (CDNA matrix cores differ), and XPU/MPS/CPU would each need their own (CPU likely cache-residency-bound, a different mechanism). The `B_ref` rule avoids all of this.
Correctness does not depend on the chunk size: it is invariant up to floating-point rounding (the fp64 invariance test), so a non-optimal chunk on any backend costs throughput only, never accuracy.
Re-fit the `B_ref` coefficient per (dtype, acc-policy) -- it is `0.25` for bf16 + `compact`; the `per_row` byte composition shifts with dtype/acc-dtype (still device-independent).

Terminology: "budget" named the `eps=1` memory-budget approach; the shipped rule is a different (and correct) memory bound, so the CUDA pick should fold into `auto` (the resolver sets a concrete `batch_chunk_size`) rather than be surfaced as a user-facing `budget` method.

## Caveats and future work

Measured for bf16 + `compact` only; re-fit the `B_ref` coefficient (`0.25`) per (dtype, acc-policy) before assuming it -- the `per_row` byte composition shifts with dtype/acc-dtype (the result stays device-independent).
Tile-alignment is unverified: we only swept powers of two, so `floor_tile` (vs `floor_pow2`) rests on GEMM reasoning, not data -- confirm with a tile-aligned non-power-of-two sub-sweep (e.g. `B` over multiples of 256) that the throughput plateau has no power-of-two-specific cliffs before shipping tile-flooring.
The `k*SM` throughput finding (per-arch, two classes) is robust across five GPUs, but it is the rationale, not the shipped cap; the shipped `B_ref` rule is device-independent and needs no per-arch measurement for a new GPU.
The shape grid is moderate; `B_ref = 0.25*N*V/D` is clean (log2 RMSE 0.011) but should be spot-checked at LLM-region and extreme-batch shapes the grid did not cover.

## Data and reproduction

Sweep + analysis: `chunk_size_sweep.py`; per-GPU data: `chunk_size_sweep_{a100,h100,h200,b200,b300}.csv`; summary figure: `chunk_size_summary.png` via `plot_chunk_summary.py`; run instructions: `RUN.md`.

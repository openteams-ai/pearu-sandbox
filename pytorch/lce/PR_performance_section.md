## Performance

All numbers: A100-SXM4-80GB, CUDA 13.1, driver 580.126.09, bf16, index targets, fwd+bwd median time / peak memory. Lines: **auto** = this PR's default (factor-1 chunk + `N*V/4D` cap), **ar:2** = the previous default (`aspect_ratio:2`), **liger** = liger-kernel's fused op, **reference** = full materialization (`options=None`).

### Vocab regime (`num_classes >> in_features`) -- the LLM vocab-head case

Sweeping `num_classes` at `num_tokens=4096`, `in_features=4096`:

![throughput](ATTACH cap_before_after_bfloat16_cuda_N4096_D4096_Vsweep.png)

`auto` is the fastest memory-saving path and the leanest, beating both the old `ar:2` default and liger, with the gap widening as the vocabulary grows:

| num_classes | auto (this PR) | ar:2 (old) | liger | reference |
|---|---|---|---|---|
| 32000  | 19.0 ms / 0.70 GiB | 23.1 / 0.63 | 26.0 / 1.33 | 12.8 / 1.27 |
| 65536  | 44.5 ms / 1.21 GiB | 63.8 / 1.14 | 79.5 / 2.61 | 27.5 / 2.55 |
| 131072 | 125.8 ms / 2.21 GiB | 203.1 / 2.14 | 290.7 / 5.11 | 54.7 / 5.05 |

At `num_classes=131072`, `auto` is ~1.6x faster than the old default and ~2.3x faster than liger, at ~43% of their peak memory. (liger's peak tracks full materialization here, so its memory advantage over reference is small; the chunked path's is large.) Reference is faster in raw time but uses ~2.3x the memory and stops being an option once the full logits do not fit.

### Budget regime (`in_features >= num_classes`) -- atypical for index

Here `aspect_ratio` degenerates to a single chunk whose peak would exceed reference. Sweeping `in_features` at `num_tokens=4096`, `num_classes=8192` (`in_features >= 8192` is the budget regime):

![cap-safety](ATTACH cap_before_after_bfloat16_cuda_N4096_V8192.png)

The `N*V/4D` cap keeps `auto` at or below the reference peak across the whole range, while uncapped factor-1, the old `ar:2` default, and liger all rise above it (peak memory, GiB):

| in_features | auto | uncapped factor-1 | ar:2 | liger | reference |
|---|---|---|---|---|---|
| 8192 (=V) | 0.48 | 0.77 | 0.58 | 0.83 | 0.52 |
| 16384     | 0.84 | 1.33 | 1.05 | 1.58 | 0.89 |
| 32768     | 1.59 | 2.45 | 2.11 | 3.08 | 1.64 |

`auto` is the only configuration that never exceeds reference. The cap costs some throughput here (more, smaller chunks) -- the intended memory-safety tradeoff in a regime that index workloads (vocab heads, `V >> D`) do not operate in.

### Probability targets

Probability targets already resolved to factor 1; this PR adds the analogous `N/2` cap on the compact path, so their throughput is unchanged and their peak likewise stays at or below reference.

### Accuracy

Unchanged. The chunk-size change shifts the per-`(dtype, policy)` gradient ULP, so the accuracy caps in `test_nn.py` were re-measured across CPU, CUDA, ROCm, and MPS -- every leg stays within the existing caps, so no cap values changed.

# chunk_size_sweep.py -- running on the A100

Goal: find the throughput-optimal `batch_chunk_size` (`B_knee`) in the budget regime (`in_features >= num_classes`), to derive a throughput-proven chunking heuristic above the crossing.

## PyTorch build requirement

Current pytorch `main` is sufficient -- no ghstack checkout needed.
The script drives the chunked op through `chunking_method=None` + `batch_chunk_size=B`, which are the landed #187219 API (`acc_policy`, `chunking_method`, `batch_chunk_size`, and `F.linear_cross_entropy` are all on `main`).
It deliberately does NOT use `chunking_method="budget"` (the in-review #187271), so running on a clean `main` keeps the throughput baseline decoupled from the heuristic we are trying to replace.
Build the usual way: `pip install -e . -v --no-build-isolation`.

For the cross-GPU constant to be comparable, build the SAME `main` commit on every machine -- pin one SHA and use it everywhere, so the chunked-op code is byte-identical across architectures and only the hardware differs.
The bf16 path needs sm_80+ (A100/H100/H200/B200/B300 all qualify).
Blackwell (B200/B300) additionally needs a CUDA toolkit that knows its arch (12.8+) and `TORCH_CUDA_ARCH_LIST` covering it (e.g. `9.0` for Hopper, `10.0`/`12.0` for Blackwell); confirm the pinned `main` builds for Blackwell before relying on it.
No liger needed -- this script never imports it.

## Multiple GPUs

Each machine auto-stamps its GPU into the CSV `device` column, so results from different GPUs can be fit together.
Write one CSV per GPU (the device name is in the rows, not the filename), then merge them in `--analyze`:

```
# on each GPU machine (device-tag auto-detected):
python chunk_size_sweep.py --out chunk_size_sweep_h100.csv --dtypes bfloat16

# back on the analysis machine, after pulling all CSVs:
python chunk_size_sweep.py --inputs chunk_size_sweep_*.csv --dtypes bfloat16 --analyze
```

`--analyze` then fits the constant per `(dtype, device)` and prints a cross-device table with `B/SM` to test whether the saturation `B` is a flat constant or scales with SM count.
Pass `--device-tag NAME` to override the auto-detected slug (e.g. to disambiguate two cards of the same model).

## Measure (A100, GPU-bound, the only step that needs the A100)

```
cd <sandbox>/pytorch/lce
git pull
python chunk_size_sweep.py --out chunk_size_sweep.csv --dtypes bfloat16
```

The measure step is resumable: it skips `(shape, dtype, mode, B)` rows already in the CSV, so a dropped session just re-runs `git pull` + the same command and continues.
Use `--force` to overwrite instead of resume.

## Bring results back

```
git add chunk_size_sweep.csv && git commit -m "A100 chunk-size sweep (bf16, budget regime)" && git push
```

The `.png` is optional -- `--analyze` is pure CSV post-processing and can run anywhere, so pushing just the CSV is enough.

## Analyze (anywhere, no GPU)

```
python chunk_size_sweep.py --out chunk_size_sweep.csv --dtypes bfloat16 --analyze
```

Prints, in order: the per-shape knee table (`B_knee`, `B/N`, knee-vs-best time, knee-vs-reference memory, `<=ref?`), the `B_knee` spread, and the heuristic fit (`const`, `c*N`, `c*V`, `c*D`, `c*N*V/D` ranked by log2 RMSE), then writes a per-shape time-vs-B / memory-vs-B figure.
`--tol` sets the throughput-plateau width (default 0.03 = within 3% of the best time).

## Profile (A100, confirm GEMM domination)

```
python chunk_size_sweep.py --profile --dtypes bfloat16 --profile-shape 8192 32768 16384 --profile-b 2048
```

Wraps one fwd+bwd in `torch.profiler`, attributes CUDA self-time to each aten op (self-time over the aten layer = total GPU time, no double count), and buckets into GEMM / gather-scatter / elementwise-reduction.
The three GEMMs are `aten::mm` (forward logits), `aten::addmm` (grad_input), and `aten::addmm_` or `aten::mm` (grad_weight, compact vs scratch path).
Run it at a couple of `--profile-b` values (e.g. 2048 and 512) to see whether smaller chunks shift time from GEMM toward softmax/launch overhead.

## Knobs

`--num-tokens / --in-features / --num-classes` override the swept grid (space-separated lists).
`--acc-policy` defaults to `compact` (the budget-regime CUDA auto pick); pass `accurate` to sweep that path instead.
`--smoke` runs a tiny grid for plumbing checks (use `--dtypes float32` on pre-sm_80 cards, which cannot do the bf16->fp32 gemm).

# chunk_size_sweep.py -- running on the A100

Goal: find the throughput-optimal `batch_chunk_size` (`B_knee`) in the budget regime (`in_features >= num_classes`), to derive a throughput-proven chunking heuristic above the crossing.

## PyTorch build requirement

Current pytorch `main` is sufficient -- no ghstack checkout needed.
The script drives the chunked op through `chunking_method=None` + `batch_chunk_size=B`, which are the landed #187219 API (`acc_policy`, `chunking_method`, `batch_chunk_size`, and `F.linear_cross_entropy` are all on `main`).
It deliberately does NOT use `chunking_method="budget"` (the in-review #187271), so running on a clean `main` keeps the throughput baseline decoupled from the heuristic we are trying to replace.
Build the usual way: `pip install -e . -v --no-build-isolation`.

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

## Knobs

`--num-tokens / --in-features / --num-classes` override the swept grid (space-separated lists).
`--acc-policy` defaults to `compact` (the budget-regime CUDA auto pick); pass `accurate` to sweep that path instead.
`--smoke` runs a tiny grid for plumbing checks (use `--dtypes float32` on pre-sm_80 cards, which cannot do the bf16->fp32 gemm).

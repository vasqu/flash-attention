# Indexed FA4 SM90 optimization roadmap

This document separates measured policy changes from kernel experiments that
need dedicated H100 evidence before they enter automatic dispatch.

## Measured changes in incremental patch 0022

- Promote ratio-8 GQA (`H64/H8`, `D128/DV128`) at the same directly measured
  true-prefill lengths and batch limits as the other qualified common profiles.
  The 252-case BF16 run measured a 2.185x profile geomean opportunity and a
  1.494x worst best-path improvement, with no correctness failures.
- Keep S512/S1024 high-batch and S2048/B16 expansion explicit. The report shows
  large sparse wins, but the prior matrix did not pair both locality patterns
  at every boundary.
- Use `common-indexed-5h` for the next promotion decision. Its 420 cases add
  ratio-2 GQA, ratio-7 GQA, D96 MHA, and paired window/mixed controls.
- Require the generated prefill certificate to report complete timings,
  correctness within the BF16 thresholds, geomean >=1.03x, and minimum
  >=0.98x before extending automatic dispatch.

## Next certificate-driven decisions

1. Decide S512 and S1024 bitmask-to-block-sparse batch crossovers independently.
2. Decide S2048/B16 promotion only from the paired window and mixed rows.
3. Qualify ratio-2 GQA, ratio-7 GQA, and D96 MHA as complete profile families;
   do not infer them from ratio-4/ratio-8 GQA or D64/D128 MHA.
4. Keep decode, chunked-query, and true-prefill promotion certificates
   independent even when they share head dimensions.

## Measured changes in incremental patch 0007

- Route `B1/Q128/K>=65536/T2048/H32/32/D128` MHA to row-sparse. The full
  BF16+FP16 run measured 1.36-1.40x lower end-to-end latency than dense indexed.
- Use one head per warp for the narrow high-ratio MQA `Q~128/T~129/D<=128`
  band. The BF16+FP16 top-k sweep measured a repeatable 2.6-3.0% improvement.
- Keep adjacent controls unchanged: MHA `K32768/T2048`, true `Q=K` prefill,
  and MQA `T73`/`T257`.

## Prioritized kernel experiments

### 1. Reusable scheduler metadata and split workspace

FlashMLA computes tile-scheduler metadata and split counts once before the
layer loop. Add an optional prepared execution object containing row mapping,
chosen splits, and reusable partial-output/LSE workspace. This should reduce
Python/CuTe launch setup and allocator traffic without changing the public
indices format. Benchmark decode across 1, 8, 32, and 128 batches and report
prepared-kernel and end-to-end costs separately.

### 2. Producer/consumer sparse MQA prefill kernel

Build a specialized `hkv=1` path rather than expanding query unions. A producer
warpgroup gathers blocks of selected K/V into double-buffered shared memory;
two consumer warpgroups pipeline WGMMA QK, online softmax, and PV. Start with
BF16, D/DV 128, top-k multiples of 128, and `Q=K` serving prefill. This follows
the useful shape restrictions of FlashMLA's sparse prefill implementation while
preserving FA4's arbitrary per-query selected indices.

### 3. Programmatic dependent launch for split rows

The row kernel and split-combine kernel are ordered but currently pay separate
launch latency. Test CUDA programmatic dependent launch so the combine kernel
can be submitted early and wait on row completion. This is most relevant to
small-batch decode where the selected-token kernel itself is near 0.14 ms.

### 4. Cluster/DSM K/V sharing for high-ratio MQA decode

FlashMLA's sparse decode shares dequantized K/V between two CTA head partitions
through distributed shared memory. Prototype the same idea for BF16 K/V at
large head ratios and wide D/DV, where multiple query-head partitions currently
reload identical selected tokens. Gate the experiment on cluster occupancy and
measure both Q=1 and speculative Q=8.

### 5. Fine-grained gather pipeline and cache policy

Add a selected-token producer stage that issues vectorized global loads for the
next index block while consumers process the current block. Compare ordinary,
recent-tail, and random patterns; consider L2-persistent or evict-first hints
only when the index locality metric predicts reuse. Do not globally force a
cache policy because random selected rows can evict useful Q/output state.

### 6. Offline-generated dispatch table

Keep hand-written safety guards, but generate crossover tables from benchmark
CSV for stable dimensions `(dtype, D, DV, head ratio, Q regime, density band,
row count)`. Emit conservative decisions only when the best backend wins by a
confidence margin in both BF16 and FP16. Fall back to the existing rules for
unseen shapes.

## Non-goals for the next policy patch

- Do not route general GQA/MHA to a FlashMLA-style MQA kernel.
- Do not inspect indices or synchronize inside the attention call.
- Do not select union without an overlap/inflation hint or a measured MQA rule.
- Do not treat dense FA4 as a same-semantics baseline.

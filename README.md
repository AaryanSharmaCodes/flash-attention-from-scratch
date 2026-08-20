# FlashAttention from scratch

Attention built up from raw CUDA kernels, one optimisation at a time, with every
step measured against cuBLAS and against a CPU reference.

The repo is a ladder rather than a single kernel. It starts with a matmul written
deliberately the wrong way round and ends with a fused attention kernel that never
materialises the N x N score matrix. Each rung fixes one identified bottleneck in
the rung below it, and the benchmark reports what that fix was worth.

Everything below was measured on a Tesla T4 (sm_75, 320 GB/s, 4 MB L2). Numbers
come from the JSON files in `results/`, each written by the benchmark named next
to it.

**What it does.** Reaches 99.7% of cuBLAS on a 1024 matmul and 78.5% at 4096,
sustains 72% of peak memory bandwidth on softmax, and computes attention at
N=4096 in 2.94x less time than the three-kernel version while using no scratch
memory at all instead of 64 MB.

| | |
| --- | --- |
| matmul, 1024 | 3558 GFLOP/s, 99.7% of cuBLAS |
| matmul, 4096 | 3306 GFLOP/s, 78.5% of cuBLAS |
| range across the matmul ladder | 60.6 to 3558 GFLOP/s, a factor of 59 |
| difference in the result across that range | 0 bits |
| softmax, 4096 x 1024 | 229 GB/s, 71.6% of peak |
| attention scratch at N=4096 | 0 MB, against 64 MB unfused |
| attention speedup at N=4096 | 2.94x |
| attention speedup at N=1024 | 0.78x, and the L2 size says why |
| fused kernel register use | 165 registers, 0 bytes spilled |

**What it does not do.** Forward pass only, no backward. Single head, no batching
and no causal mask. FP32 throughout, so the T4's tensor cores are never touched
and neither this code nor the cuBLAS baseline uses them. The fused kernel is
compiled for a head dimension of 64. The two fastest matmul kernels assume the
dimensions are multiples of their tile sizes. This is a study of the memory
hierarchy, not a library.

## The matmul ladder

Five kernels computing the same product, differing only in how data reaches the
ALU. At 4096:

| kernel | GFLOP/s | % of cuBLAS | what changed |
| --- | --- | --- | --- |
| naive | 61.8 | 1.5 | one thread per output, threads walk down a column |
| coalesced | 536.3 | 12.7 | the same code with the thread mapping transposed |
| tiled | 814.7 | 19.4 | tiles staged in shared memory |
| register_tiled | 1693.9 | 40.2 | eight outputs per thread, held in registers |
| vectorized | 3306.2 | 78.5 | an 8 x 8 block per thread, float4 loads, A transposed |
| cuBLAS | 4210.1 | 100 | |

The step worth noticing is the first one. `naive` and `coalesced` are the same
arithmetic, the same occupancy and the same flop count; the only difference is
which of `threadIdx.x` and `threadIdx.y` selects the column, and it is worth 8.7x.

The step worth noticing second is that the entire ladder returns bit-identical
results. Every kernel accumulates each output into one scalar with k increasing,
and tiling changes where a value is staged rather than the order things are added
in, so nothing reorders and nothing rounds differently. Throughput moves by a
factor of 59 and the answer does not move at all.

## The softmax ladder

At 4096 rows of 1024:

| kernel | GB/s | % of peak |
| --- | --- | --- |
| naive, one thread per row | 26.5 | 8.3 |
| shared memory tree reduction | 229.1 | 71.6 |
| warp shuffle reduction | 229.1 | 71.6 |
| online, one pass fewer | 229.1 | 71.6 |

The last two optimisations are real, correct, and worth nothing here. At 72% of
peak bandwidth the kernel is limited by moving the data, and neither replacing the
tree reduction with shuffles nor removing a pass over the row moves less of it --
a row is 4 KB and the re-reads were being served by L1 rather than DRAM.

They stay in the repo because a measurement showing an optimisation does not help
is worth more than a ladder where every rung happens to look like an improvement.
The warp reduction also has to exist for its own sake, since the fused attention
kernel reduces where there is no bandwidth wall to hide behind.

## Fused attention

`softmax(scale * Q K^T) V`, computed two ways. The unfused path is three launches
with the scores written to global memory in between, and it needs no new kernels:
the score matrix is a matmul, the softmax is a row-wise softmax over an N x N
matrix, and the value product is another matmul. The fused path is one launch that
keeps a running maximum, a running sum and a running output per query row, and
never writes a score anywhere.

| N | scores | unfused | fused | speedup | scratch |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4 MB | 0.386 ms | 0.495 ms | 0.78x | 4 MB to 0 |
| 2048 | 16 MB | 1.510 ms | 0.976 ms | 1.55x | 16 MB to 0 |
| 4096 | 64 MB | 6.516 ms | 2.219 ms | 2.94x | 64 MB to 0 |

Fusion loses at N=1024. That is the most informative row in the table. A T4 has
4 MB of L2 and the score matrix at N=1024 is 4 MB, so the round trip through
global memory that fusion exists to avoid is being served by cache and costs
almost nothing, while the fused kernel still pays 25% occupancy and a dot product
per key. By N=4096 the scores are sixteen times the size of L2 and the unfused
path is moving a quarter of a gigabyte the fused path never touches.

So fusion buys memory unconditionally and time only once the intermediate stops
fitting in cache. The memory claim is the durable one, and it is what makes long
context possible at all.

## Correctness

Every kernel is checked against a reference more precise than itself: matmul
against cuBLAS by Frobenius relative error, softmax and attention against CPU
implementations that accumulate in double.

| check | result |
| --- | --- |
| matmul vs cuBLAS | 6.0e-07 at 1024, exactly 0 at 4096 |
| softmax vs CPU reference | 3.0e-07 |
| fused attention vs CPU reference | 6.6e-07 |
| fused vs unfused attention | 4.5e-07 to 6.0e-07 |
| streaming vs textbook attention on CPU | 2.4e-07 across 26 cases |

Twice during this work a correctness check failed kernels that were right, and
both times the fix was to the reference rather than the kernel. `DECISIONS.md`
records what happened, because those were the two most useful things that went
wrong.

## Running it

The reference checks are plain C++ and need no GPU:

```
make test
```

Everything else needs an NVIDIA card:

```
make bench       # the three ladders
make diagnose    # matmul against a double-precision CPU reference
```

## A note on the timings

The same binary on the same card gave 0.942 ms and 0.386 ms for unfused attention
at N=1024 in two runs, a factor of 2.4, because a T4 boosts from 585 MHz to
1590 MHz and the second run followed five minutes of benchmarking. The ratios
moved by less than ten percent across the same pair of runs.

So every comparison here is made within a single run with the baseline timed
alongside the thing being compared, and speedups rather than milliseconds are what
gets quoted. An absolute time in this repo describes the clock state of one
afternoon.

## Layout

```
ref/         CPU implementations, written to be obviously correct
src/kernels/ the CUDA kernels
src/host/    benchmark and diagnostic drivers
tests/       reference checks that run without a GPU
results/     JSON written by the benchmarks; the source of every number above
```

`DECISIONS.md` records the choices that were not obvious.

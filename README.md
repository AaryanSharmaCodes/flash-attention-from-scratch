# FlashAttention from scratch

Attention built up from raw CUDA kernels, one optimisation at a time, with every
step measured against cuBLAS and PyTorch.

The repo is a ladder rather than a single kernel. It starts with a matmul that is
deliberately written the wrong way round, and ends with a fused attention kernel
that never materialises the N x N score matrix. Each rung fixes one specific
bottleneck in the rung below it, and the benchmark reports what that fix was
worth.

## Status

Stage 1 of 3 is written; numbers below are filled in from the JSON in `results/`
once each stage has been run on a GPU.

| stage | what it adds | state |
| --- | --- | --- |
| CPU reference | textbook and streaming attention, agreeing to 3e-07 | done, checked |
| 1. matmul ladder | naive, coalesced, shared-memory tiled, register tiled | written, not yet measured |
| 2. softmax | tree reduction, then warp shuffle | not started |
| 3. fused attention | unfused three-kernel, then tiled and fused | not started |

## Running it

The reference checks are plain C++ and need no GPU:

```
make test
```

Everything else needs an NVIDIA card:

```
make bench
```

## Layout

```
ref/         CPU implementations, written to be obviously correct
src/kernels/ the CUDA kernels
src/host/    benchmark and correctness drivers
tests/       reference checks that run without a GPU
results/     JSON written by the benchmarks; the source of every number quoted here
```

`DECISIONS.md` records the choices that were not obvious.

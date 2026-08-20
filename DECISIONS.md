# Decisions

Notes on choices that were not obvious, kept so I can remember why the code
looks the way it does.

## Check the streaming softmax on the CPU before writing any CUDA

The interesting part of FlashAttention is not the parallelism, it is the
rescaling: the running sum and the partial output are both held relative to the
running maximum, so both have to be corrected by `exp(m_old - m_new)` whenever a
block raises the maximum. Getting that wrong produces output that is plausible
but slightly off, which is miserable to debug through a kernel.

So `ref/attention_online` is a single-threaded version of exactly that
arithmetic, checked against the textbook implementation across several block
widths including ones that do not divide the sequence length. It agrees to
3e-07. Anything the GPU disagrees about later is a CUDA bug, and I do not have
to wonder whether the maths was ever right.

## The ladder keeps the slow kernels instead of deleting them

`naive` and `coalesced` are the same arithmetic and differ only in which of
`threadIdx.x` and `threadIdx.y` selects the column. Keeping both is the cheapest
possible demonstration that memory access pattern, on its own and with the flop
count held fixed, is worth a large multiple. Deleting the slow one would leave
the repo faster and much less informative.

## Tiles are zero-filled at the edges rather than handled separately

The tiled kernel pads out-of-range loads with zeros, so a ragged final tile
contributes nothing to the accumulator and needs no cleanup pass. The register
tiled kernel does not do this: it assumes the dimensions are multiples of the
tile sizes, because the bounds checks in the inner loop cost more than they are
worth and the benchmark sizes are powers of two anyway. That restriction is
stated in the header rather than silently assumed.

## cuBLAS is called with the operands swapped

cuBLAS is column-major and this code is row-major. Passing `B` then `A` with the
dimensions exchanged makes it compute `C^T = B^T A^T`, which is the row-major
`C = A B` already in the right layout. The alternative, transposing on the way
in and out, would have made the reference slower than the kernels it is meant to
be a ceiling for.

## Error is measured relative, not absolute

Entries of `C` grow with `K`, so a fixed absolute tolerance would pass at K=256
and fail at K=4096 for arithmetic that is equally correct. The check divides by
the reference magnitude, with a floor so that near-zero entries do not blow up
the ratio.

## Timing discards the first launches

The first launch of a kernel pays for module loading and possibly JIT. On a cold
context that is milliseconds, which would swamp everything else at these sizes.
Five warmup iterations, then twenty timed ones, measured with CUDA events rather
than host clocks so the number does not include launch queueing.

## Softmax is reported in bandwidth, not flops

Softmax does one exponential and a couple of comparisons per element it loads,
so it is bound by memory rather than arithmetic. Quoting GFLOP/s for it would be
meaningless. The benchmark reports achieved bandwidth as a share of the card's
theoretical peak instead.

Effective bandwidth is computed from the traffic the problem requires -- one read
of the input, one write of the output -- and not from the traffic a particular
kernel chooses to generate. The three-pass kernels read each row three times, and
counting those redundant reads would let a worse kernel report a better number.

## The online softmax is in the repo mainly as a bridge

It saves one of the three passes over the row, which is worth something, but the
reason it exists here is that its combine step

    m = max(m1, m2)
    l = l1 * exp(m1 - m) + l2 * exp(m2 - m)

is exactly the rescale that FlashAttention applies when it moves from one block
of keys to the next. Doing it first across the lanes of a warp, where it is easy
to check against the CPU reference, means the fused attention kernel is applying
a rule that has already been shown to work.

## Test inputs are shifted away from zero

The softmax benchmark generates values around 20 rather than around 0. With
inputs near zero a kernel that forgets to subtract the row max still produces the
right answer, and the test passes for the wrong reason. Around 20, and more so at
the tails, the unguarded version overflows and gets caught.

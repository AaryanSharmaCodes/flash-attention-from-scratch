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

## The first correctness check was measuring the wrong thing

The benchmark originally compared each element of C against its cuBLAS value and
divided by that value. Every kernel in the ladder came back at 1.64e-01 and was
flagged as wrong.

Four kernels with different memory strategies producing the same wrong answer to
three significant figures is not plausible, so the measurement was the suspect
rather than the kernels. Checking everything, cuBLAS included, against a
double-precision CPU matmul settled it:

    true C at 512:  mean |C| 18.04, max |C| 109.47, min |C| 3.743e-06
    every implementation:  max abs 1.019e-04, frobenius 4.048e-07
    pairwise difference between all five:  0.000e+00

The worst element by the old measure was C[258995], true value -9.275e-06, where
512 products cancel almost to nothing. An absolute difference of 3.4e-06 there is
ordinary float32 noise, but dividing it by an entry that small manufactures a
large number that says nothing about the kernel.

The check is now the Frobenius relative error, which is what is normally quoted
for GEMM, and the tolerance is 1e-05 against an observed 4e-07. Absolute error is
reported alongside it, since a single number was what caused the confusion.

## All five implementations agree bit for bit, which is expected

The pairwise table is exactly zero everywhere, cuBLAS included. That looked
alarming and is not:

Tiling changes where a value is staged on its way to the ALU. It does not change
the order the products are added in. All four kernels accumulate each output into
a single scalar with k strictly increasing -- naive walks k = 0..K-1, tiled walks
tile 0 then tile 1, register tiled walks blocks of eight -- so the sequence of
additions is identical in every case. Float addition is not associative, so a
reordering would move the low bits, but nothing here reorders. nvcc also
contracts acc += a*b into a single FMA by default, and cuBLAS evidently picks a
matching order at this size.

Worth stating plainly: throughput moves by a factor of 32 across the ladder and
the result does not change by one bit. These optimisations are about moving data,
not about arithmetic.

## Quoting "percentage of cuBLAS" needs a caveat

The same benchmark run twice on the same T4 gave cuBLAS 2721 GFLOP/s and then
3311 GFLOP/s at 1024, a 22% swing. The ladder's own kernels moved by less than
half a percent across the same two runs (1908 then 1914), so this is the card
changing clock and power state rather than anything in the code.

That means "percentage of cuBLAS" is only meaningful when both numbers come from
the same run, which they do here since the benchmark times cuBLAS itself each
time. It also means a single percentage is not worth quoting to better than a few
points, and the README says so rather than implying a precision the measurement
does not have.

## A fifth rung, because the fourth one plateaued

register_tiled measured 1914, 1863 GFLOP/s at 1024 and 2048 -- essentially flat,
while cuBLAS climbed from 3311 to 4666 over the same range. Flat throughput
against a growing problem is what it looks like when there is not enough
arithmetic in flight to hide memory latency, and no amount of tuning the existing
kernel fixes that.

So each thread now owns an 8 x 8 block of C rather than an 8 x 1 column. Sixteen
shared reads produce sixty-four multiply-adds, against nine reads for eight
before. Global loads become float4, and the A tile is transposed on the way into
shared memory so the inner loop can read it four at a time as well -- in the
original layout the TM rows a thread wants at a fixed k are BK apart, and
transposing makes them adjacent.

## The same reference mistake, made twice

The softmax benchmark flagged three of four kernels as wrong at 4096 columns.
The pattern gave it away: the one-thread-per-row kernel scored 1.19e-07 while the
three parallel ones scored around 1.56e-05, a hundred times worse.

The per-row kernel is not more accurate. It sums the row sequentially from j=0,
which is exactly what the CPU reference did, so the two made identical rounding
errors and cancelled. The parallel kernels sum in a different order and therefore
disagree with the reference by whatever the reference's own error happens to be.

That error is predictable. Summing n floats sequentially in float32 carries a
relative error near sqrt(n) * eps:

    1024 columns:  sqrt(1024) * 1.19e-07 = 3.8e-06,  observed 5.36e-06
    4096 columns:  sqrt(4096) * 1.19e-07 = 7.6e-06,  observed 1.56e-05

The reference now accumulates in double, which puts its own error near 1e-15 and
makes it an actual referee. This is the second time in this repo that a
correctness failure turned out to be the reference rather than the kernel, which
is worth stating as a rule: a reference is only a reference if it is more precise
than the thing it is judging.

The tolerance scales as sqrt(cols) for the same reason. The kernels sum in
float32 and always will, so their error grows with the row length no matter how
they are written. A fixed tolerance would pass everything at 1024 or fail
everything at 4096.

## Two softmax optimisations that changed nothing, kept anyway

    shared_tree     0.147 ms   228.2 GB/s   71.3% of peak
    warp_shuffle    0.146 ms   230.2 GB/s   71.9% of peak
    online          0.146 ms   230.1 GB/s   71.9% of peak

The warp shuffle version replaces a log2(BLOCK)-step shared memory tree with a
register reduction and two barriers instead of eight. The online version removes
an entire pass over the row. Neither is worth anything here, and the reason is
the same for both: at 72% of theoretical bandwidth the kernel is limited by
moving the data, and neither change moves less of it.

The removed pass in particular looks like it should help until you notice a row
is 4 KB at 1024 columns and 16 KB at 4096, so the re-reads were being served by
L1 and never reached DRAM. The pass was free before it was removed.

Both kernels stay in the repo. They are correct, they are the textbook technique,
and the measurement showing they do not help is more useful than a ladder where
every rung happens to look like an improvement. The warp reduction also has to
exist for its own sake, since the fused attention kernel reduces across a warp
where there is no bandwidth wall to hide behind.

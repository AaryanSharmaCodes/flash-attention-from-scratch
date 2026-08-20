// Row-wise softmax over a row-major (rows x cols) matrix.
//
// Softmax does almost no arithmetic per byte it touches, so all of these are
// memory bound. The interesting number is achieved bandwidth against what the
// card can actually sustain, not GFLOP/s.
#pragma once

namespace softmax {

// One thread per row. Rows are handled independently and each thread walks its
// row three times. With few rows this leaves most of the GPU idle.
void launch_naive(const float* in, float* out, int rows, int cols);

// One block per row, reducing through shared memory in a tree. Three passes
// over the row: max, then exponentiate and sum, then normalise.
void launch_shared(const float* in, float* out, int rows, int cols);

// Same structure, but the reduction happens in registers via warp shuffles,
// with shared memory used only to combine across warps.
void launch_warp(const float* in, float* out, int rows, int cols);

// Tracks a running max and a running sum together, so the row is read twice
// instead of three times. This is the softmax half of FlashAttention.
void launch_online(const float* in, float* out, int rows, int cols);

}  // namespace softmax

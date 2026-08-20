// CPU reference implementations. These are deliberately simple and slow: they
// exist to be obviously correct, so the GPU kernels have something to be checked
// against. Nothing in here is meant to be fast.
#pragma once
#include <cstddef>

namespace ref {

// C = A * B, with A being (M x K), B (K x N), C (M x N), all row-major.
void matmul(const float* A, const float* B, float* C, int M, int N, int K);

// Row-wise softmax over a (rows x cols) row-major matrix, done in the
// max-subtracted form so that exp() never sees a large positive argument.
void softmax_rows(const float* in, float* out, int rows, int cols);

// Attention the textbook way: build the full (N x N) score matrix, softmax it,
// then multiply by V. Memory grows with N^2, which is the whole problem.
//   Q, K, V are (N x d) row-major; out is (N x d).
void attention_naive(const float* Q, const float* K, const float* V,
                     float* out, int N, int d, float scale);

// Attention computed in tiles, carrying a running max and a running sum so the
// (N x N) score matrix is never held anywhere. This is the FlashAttention
// algorithm with the GPU stripped out of it -- same arithmetic, one thread.
// If this agrees with attention_naive, the streaming rescale logic is right,
// and any later disagreement on the GPU is a CUDA bug rather than a maths bug.
void attention_online(const float* Q, const float* K, const float* V,
                      float* out, int N, int d, float scale, int block_c);

}  // namespace ref

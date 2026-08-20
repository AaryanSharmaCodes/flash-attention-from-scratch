#include "../../include/attention.h"
#include "../../include/sgemm.h"
#include "../../include/softmax.h"
#include "../../include/util.cuh"

#include <cfloat>

namespace attention {
namespace {

// ---------------------------------------------------------------------------
// Scores for the unfused path: S = scale * Q K^T.
//
// K is stored (N x d) like Q, so the transpose is done by indexing rather than
// by moving anything: both operands are walked along d.
//
// The shared tiles are padded to 33 columns. Without the padding, the read
// Ks[tx][k] has tx varying across the warp with k fixed, which is a stride of 32
// floats -- every lane lands on the same bank and the access serialises 32 ways.
// One extra column shifts each row by one bank and the conflict disappears.
// ---------------------------------------------------------------------------
constexpr int TS = 32;

__global__ void qk_scores(const float* __restrict__ Q, const float* __restrict__ K,
                          float* __restrict__ S, int N, int d, float scale) {
  __shared__ float Qs[TS][TS + 1];
  __shared__ float Ks[TS][TS + 1];

  const int tx = threadIdx.x % TS;
  const int ty = threadIdx.x / TS;
  const int row = blockIdx.y * TS + ty;  // query index
  const int col = blockIdx.x * TS + tx;  // key index
  const int krow = blockIdx.x * TS + ty; // key index for the cooperative load

  float acc = 0.0f;
  for (int t = 0; t < d; t += TS) {
    Qs[ty][tx] = (row < N && t + tx < d) ? Q[(size_t)row * d + t + tx] : 0.0f;
    Ks[ty][tx] = (krow < N && t + tx < d) ? K[(size_t)krow * d + t + tx] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TS; ++k) acc += Qs[ty][k] * Ks[tx][k];
    __syncthreads();
  }

  if (row < N && col < N) S[(size_t)row * N + col] = acc * scale;
}

// ---------------------------------------------------------------------------
// Fused attention.
//
// One thread per query row. The thread holds q and the running output o in
// registers and never writes a score anywhere; K and V arrive in shared memory
// a tile at a time, shared by every thread in the block.
//
// The state carried per row is (m, l, o): the largest score seen so far, the sum
// of exponentials so far, and the weighted sum of value vectors so far. The last
// two are held relative to m, so whenever a new key raises the maximum both have
// to be scaled by exp(m_old - m_new). That single correction is what makes one
// streaming pass equivalent to the two-pass softmax.
//
// The rescale happens per key rather than per tile. Doing it per tile would need
// the tile's scores kept somewhere first, and at BC=32 that is another 32 values
// per thread on top of the 128 already held for q and o. Per key costs one extra
// exponential against d fused multiply-adds, and exp(0) = 1 in the common case
// where the maximum does not move.
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int BR, int BC>
__global__ void fused(const float* __restrict__ Q, const float* __restrict__ K,
                      const float* __restrict__ V, float* __restrict__ O, int N,
                      float scale) {
  __shared__ float Ks[BC][HEAD_DIM];
  __shared__ float Vs[BC][HEAD_DIM];

  const int row = blockIdx.x * BR + threadIdx.x;
  const bool active = row < N;

  // Indexed only by unrolled compile-time constants, so both stay in registers.
  float q[HEAD_DIM];
  float o[HEAD_DIM];
#pragma unroll
  for (int k = 0; k < HEAD_DIM; ++k) {
    q[k] = active ? Q[(size_t)row * HEAD_DIM + k] : 0.0f;
    o[k] = 0.0f;
  }

  // -FLT_MAX rather than -infinity: the first correction is then exp of a very
  // negative finite number, which is zero, instead of inf minus inf, which is
  // not a number.
  float m = -FLT_MAX;
  float l = 0.0f;

  for (int j0 = 0; j0 < N; j0 += BC) {
    // Cooperative load of the tile. BR threads fill BC * HEAD_DIM floats, so
    // each thread copies a strided share of them.
    for (int idx = threadIdx.x; idx < BC * HEAD_DIM; idx += BR) {
      const int t = idx / HEAD_DIM;
      const int k = idx % HEAD_DIM;
      const int src = j0 + t;
      Ks[t][k] = (src < N) ? K[(size_t)src * HEAD_DIM + k] : 0.0f;
      Vs[t][k] = (src < N) ? V[(size_t)src * HEAD_DIM + k] : 0.0f;
    }
    __syncthreads();

    const int limit = min(BC, N - j0);
    for (int t = 0; t < limit; ++t) {
      float s = 0.0f;
#pragma unroll
      for (int k = 0; k < HEAD_DIM; ++k) s += q[k] * Ks[t][k];
      s *= scale;

      const float m_new = fmaxf(m, s);
      const float corr = __expf(m - m_new);
      const float p = __expf(s - m_new);

      l = l * corr + p;
#pragma unroll
      for (int k = 0; k < HEAD_DIM; ++k) o[k] = o[k] * corr + p * Vs[t][k];
      m = m_new;
    }
    // Keeps a thread that has finished the tile from overwriting it while
    // slower threads are still reading.
    __syncthreads();
  }

  // The division by l is deferred to here, so it costs d operations per row
  // rather than d per key.
  if (active) {
#pragma unroll
    for (int k = 0; k < HEAD_DIM; ++k) O[(size_t)row * HEAD_DIM + k] = o[k] / l;
  }
}

}  // namespace

void launch_unfused(const float* Q, const float* K, const float* V, float* O,
                    float* scratch, int N, int d, float scale) {
  const dim3 block(TS * TS);
  const dim3 grid(ceil_div(N, TS), ceil_div(N, TS));
  qk_scores<<<grid, block>>>(Q, K, scratch, N, d, scale);
  CUDA_CHECK_LAUNCH();

  // The scores are an N x N row-major matrix, which is exactly the shape the
  // softmax ladder already handles.
  softmax::launch_warp(scratch, scratch, N, N);

  // And O = S V is an ordinary (N x N) by (N x d) product, so the matmul ladder
  // covers it without a new kernel.
  sgemm::launch_tiled(scratch, V, O, N, d, N);
}

void launch_fused(const float* Q, const float* K, const float* V, float* O, int N,
                  int d, float scale) {
  constexpr int HEAD_DIM = 64, BR = 64, BC = 32;
  if (d != HEAD_DIM) {
    std::fprintf(stderr, "fused attention is compiled for d=%d, got %d\n", HEAD_DIM, d);
    std::exit(1);
  }
  fused<HEAD_DIM, BR, BC><<<ceil_div(N, BR), BR>>>(Q, K, V, O, N, scale);
  CUDA_CHECK_LAUNCH();
}

}  // namespace attention

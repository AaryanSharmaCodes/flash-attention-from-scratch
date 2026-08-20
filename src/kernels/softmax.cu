#include "../../include/softmax.h"
#include "../../include/util.cuh"

#include <cfloat>

namespace softmax {
namespace {

constexpr int BLOCK = 256;
constexpr unsigned FULL_MASK = 0xffffffffu;

// ---------------------------------------------------------------------------
// Reduction helpers.
//
// __shfl_down_sync moves a register straight between lanes of a warp without
// going through shared memory at all. Five rounds collapse 32 lanes into lane
// zero. The mask names the lanes taking part; every thread named in it has to
// reach the instruction, which is why these are called unconditionally rather
// than inside a divergent branch.
// ---------------------------------------------------------------------------
__inline__ __device__ float warp_reduce_max(float v) {
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, off));
  return v;
}

__inline__ __device__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1)
    v += __shfl_down_sync(FULL_MASK, v, off);
  return v;
}

// Reduce across the block, then hand the answer back to every thread. Only one
// float per warp crosses shared memory, against one per thread for a tree
// reduction.
__inline__ __device__ float block_reduce_max(float v, float* scratch) {
  const int lane = threadIdx.x % warpSize;
  const int wid = threadIdx.x / warpSize;
  const int nwarps = blockDim.x / warpSize;

  v = warp_reduce_max(v);
  if (lane == 0) scratch[wid] = v;
  __syncthreads();

  v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : -FLT_MAX;
  if (wid == 0) v = warp_reduce_max(v);
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  return scratch[0];
}

__inline__ __device__ float block_reduce_sum(float v, float* scratch) {
  const int lane = threadIdx.x % warpSize;
  const int wid = threadIdx.x / warpSize;
  const int nwarps = blockDim.x / warpSize;

  v = warp_reduce_sum(v);
  if (lane == 0) scratch[wid] = v;
  __syncthreads();

  v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : 0.0f;
  if (wid == 0) v = warp_reduce_sum(v);
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  return scratch[0];
}

// ---------------------------------------------------------------------------
// 1. One thread per row.
//
// Correct, and a poor fit for the hardware: consecutive threads are a whole row
// apart, so nothing coalesces, and a batch of 1024 rows occupies four blocks.
// ---------------------------------------------------------------------------
__global__ void naive(const float* __restrict__ in, float* __restrict__ out,
                      int rows, int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  const float* r = in + (size_t)row * cols;
  float* o = out + (size_t)row * cols;

  float m = -FLT_MAX;
  for (int j = 0; j < cols; ++j) m = fmaxf(m, r[j]);

  float sum = 0.0f;
  for (int j = 0; j < cols; ++j) {
    o[j] = __expf(r[j] - m);
    sum += o[j];
  }
  for (int j = 0; j < cols; ++j) o[j] /= sum;
}

// ---------------------------------------------------------------------------
// 2. One block per row, tree reduction through shared memory.
//
// The strided loop means the threads of a warp read adjacent floats, so the row
// is streamed in coalesced. The reduction halves the active threads each round,
// so the last rounds run at a fraction of a warp -- which is what the shuffle
// version below improves on.
// ---------------------------------------------------------------------------
__global__ void shared_tree(const float* __restrict__ in, float* __restrict__ out,
                            int rows, int cols) {
  __shared__ float sdata[BLOCK];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float* r = in + (size_t)row * cols;
  float* o = out + (size_t)row * cols;

  float local = -FLT_MAX;
  for (int j = tid; j < cols; j += BLOCK) local = fmaxf(local, r[j]);
  sdata[tid] = local;
  __syncthreads();
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  const float m = sdata[0];
  // Every thread has read sdata[0] before the next phase overwrites sdata[tid].
  __syncthreads();

  float lsum = 0.0f;
  for (int j = tid; j < cols; j += BLOCK) {
    const float e = __expf(r[j] - m);
    o[j] = e;
    lsum += e;
  }
  sdata[tid] = lsum;
  __syncthreads();
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  const float total = sdata[0];
  __syncthreads();

  for (int j = tid; j < cols; j += BLOCK) o[j] /= total;
}

// ---------------------------------------------------------------------------
// 3. Warp shuffle reduction.
//
// Identical passes, but the reduction stays in registers. Shared memory carries
// one float per warp rather than one per thread, and there are two
// __syncthreads() per reduction instead of log2(BLOCK) of them.
// ---------------------------------------------------------------------------
__global__ void warp_shuffle(const float* __restrict__ in, float* __restrict__ out,
                             int rows, int cols) {
  __shared__ float scratch[BLOCK / 32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float* r = in + (size_t)row * cols;
  float* o = out + (size_t)row * cols;

  float local = -FLT_MAX;
  for (int j = tid; j < cols; j += BLOCK) local = fmaxf(local, r[j]);
  const float m = block_reduce_max(local, scratch);

  float lsum = 0.0f;
  for (int j = tid; j < cols; j += BLOCK) {
    const float e = __expf(r[j] - m);
    o[j] = e;
    lsum += e;
  }
  const float total = block_reduce_sum(lsum, scratch);

  for (int j = tid; j < cols; j += BLOCK) o[j] /= total;
}

// ---------------------------------------------------------------------------
// 4. Online softmax.
//
// The three kernels above read the row three times: once for the max, once to
// exponentiate, once to normalise. Carrying (max, sum) together removes the
// first pass. Each thread folds its own slice into a running pair, and the pairs
// combine associatively:
//
//     m = max(m1, m2)
//     l = l1 * exp(m1 - m) + l2 * exp(m2 - m)
//
// which is the same rescaling FlashAttention applies, here across lanes of a
// warp rather than across blocks of keys.
// ---------------------------------------------------------------------------
__global__ void online(const float* __restrict__ in, float* __restrict__ out,
                       int rows, int cols) {
  __shared__ float smax[BLOCK / 32];
  __shared__ float ssum[BLOCK / 32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % warpSize;
  const int wid = tid / warpSize;
  const int nwarps = BLOCK / warpSize;
  const float* r = in + (size_t)row * cols;
  float* o = out + (size_t)row * cols;

  float m = -FLT_MAX, l = 0.0f;
  for (int j = tid; j < cols; j += BLOCK) {
    const float x = r[j];
    const float m_new = fmaxf(m, x);
    // exp(m - m_new) is 1 when the max did not move, so the common case costs
    // one exponential and nothing else.
    l = l * __expf(m - m_new) + __expf(x - m_new);
    m = m_new;
  }

  // Combine the per-thread pairs down the warp, applying the same rescale.
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    const float m_other = __shfl_down_sync(FULL_MASK, m, off);
    const float l_other = __shfl_down_sync(FULL_MASK, l, off);
    const float m_new = fmaxf(m, m_other);
    l = l * __expf(m - m_new) + l_other * __expf(m_other - m_new);
    m = m_new;
  }

  if (lane == 0) {
    smax[wid] = m;
    ssum[wid] = l;
  }
  __syncthreads();

  if (wid == 0) {
    m = (tid < nwarps) ? smax[tid] : -FLT_MAX;
    l = (tid < nwarps) ? ssum[tid] : 0.0f;
#pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const float m_other = __shfl_down_sync(FULL_MASK, m, off);
      const float l_other = __shfl_down_sync(FULL_MASK, l, off);
      const float m_new = fmaxf(m, m_other);
      l = l * __expf(m - m_new) + l_other * __expf(m_other - m_new);
      m = m_new;
    }
    if (tid == 0) {
      smax[0] = m;
      ssum[0] = l;
    }
  }
  __syncthreads();

  const float row_max = smax[0], row_sum = ssum[0];
  for (int j = tid; j < cols; j += BLOCK) o[j] = __expf(r[j] - row_max) / row_sum;
}

}  // namespace

void launch_naive(const float* in, float* out, int rows, int cols) {
  naive<<<ceil_div(rows, BLOCK), BLOCK>>>(in, out, rows, cols);
  CUDA_CHECK_LAUNCH();
}

void launch_shared(const float* in, float* out, int rows, int cols) {
  shared_tree<<<rows, BLOCK>>>(in, out, rows, cols);
  CUDA_CHECK_LAUNCH();
}

void launch_warp(const float* in, float* out, int rows, int cols) {
  warp_shuffle<<<rows, BLOCK>>>(in, out, rows, cols);
  CUDA_CHECK_LAUNCH();
}

void launch_online(const float* in, float* out, int rows, int cols) {
  online<<<rows, BLOCK>>>(in, out, rows, cols);
  CUDA_CHECK_LAUNCH();
}

}  // namespace softmax

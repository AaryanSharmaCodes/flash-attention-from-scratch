#include "../../include/sgemm.h"
#include "../../include/util.cuh"

namespace sgemm {
namespace {

// ---------------------------------------------------------------------------
// 1. Naive.
//
// Deliberately mapped the wrong way round: threadIdx.x selects the row, so the
// 32 threads of a warp write C[0*N+c], C[1*N+c], ... which are N floats apart.
// Every store touches its own cache line and the memory system does 32 separate
// transactions for what could have been one.
// ---------------------------------------------------------------------------
__global__ void naive(const float* __restrict__ A, const float* __restrict__ B,
                      float* __restrict__ C, int M, int N, int K) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 2. Coalesced.
//
// The only change is that threadIdx.x now selects the column. A warp covers 32
// adjacent floats of one row of C, which the hardware services as a handful of
// wide transactions. B is read the same way. A is now identical across the warp,
// so it broadcasts. Same flops, same occupancy, different address pattern.
// ---------------------------------------------------------------------------
__global__ void coalesced(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 3. Shared-memory tiled.
//
// Both kernels above read each element of A exactly N times and each element of
// B exactly M times, all from global memory. Here a BS x BS tile is staged in
// shared memory once and then read BS times from there, cutting global traffic
// by a factor of BS.
//
// The two __syncthreads() are both load-bearing. The first stops a thread
// reading a tile before its neighbours have finished writing it; the second
// stops a fast thread overwriting the tile on the next iteration while a slow
// one is still reading it. Dropping the second is a classic race that passes
// on small inputs and fails on large ones.
// ---------------------------------------------------------------------------
template <int BS>
__global__ void tiled(const float* __restrict__ A, const float* __restrict__ B,
                      float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BS][BS];
  __shared__ float Bs[BS][BS];

  const int tx = threadIdx.x % BS;
  const int ty = threadIdx.x / BS;
  const int row = blockIdx.y * BS + ty;
  const int col = blockIdx.x * BS + tx;

  float acc = 0.0f;
  for (int t = 0; t < K; t += BS) {
    // Zero-fill past the edge so the ragged last tile contributes nothing,
    // rather than needing a separate cleanup kernel.
    As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
    Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BS; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }

  // As[ty][k] is one address for the whole warp, which shared memory broadcasts.
  // Bs[k][tx] runs across tx, hitting 32 distinct banks. Neither access
  // conflicts, which is why this layout is worth keeping.
  if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 4. Register tiled.
//
// The tiled kernel is now limited by shared memory bandwidth: every fused
// multiply-add needs two shared loads. Giving each thread TM outputs stacked in
// a column lets one load of B feed TM multiply-adds, since those TM outputs all
// share the same column of B. The ratio of arithmetic to shared traffic goes
// from 1:2 to TM:(TM+1).
//
// The accumulators live in registers, so TM also sets register pressure, which
// sets occupancy. TM=8 is the usual place where the extra arithmetic per load
// still outweighs the blocks lost.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM>
__global__ void register_tiled(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move each pointer to this block's slab so the indexing below stays local.
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // Loading and computing use different thread shapes. With BM*BN/TM threads,
  // both tiles hold exactly that many elements, so each thread loads one of each.
  const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
  const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN, threadCol = threadIdx.x % BN;

  float acc[TM] = {0.0f};

  for (int t = 0; t < K; t += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
      // Hoisting this out of the inner loop is the entire trick: one shared
      // read, TM multiply-adds against it.
      const float b = Bs[dot * BN + threadCol];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        acc[i] += As[(threadRow * TM + i) * BK + dot] * b;
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) C[(threadRow * TM + i) * N + threadCol] = acc[i];
}

// ---------------------------------------------------------------------------
// 5. Two-dimensional register tiling with vectorised loads.
//
// The kernel above gives each thread a column of eight outputs, so one shared
// load of B feeds eight multiply-adds. That runs out of headroom: the measured
// throughput sits near 1900 GFLOP/s at every size while cuBLAS keeps climbing,
// which is the signature of not having enough arithmetic in flight to cover
// memory latency.
//
// Giving each thread an 8 x 8 block instead changes the ratio sharply. Loading
// TM values of A and TN values of B, 16 shared reads, produces TM * TN = 64
// multiply-adds. The arithmetic per shared load goes from 8 to 4, and the
// arithmetic per thread goes up eightfold.
//
// Two further changes come with it:
//
// Global loads use float4, so each thread moves 16 bytes per instruction rather
// than 4. Fewer, wider transactions is what the memory system prefers, and it
// cuts the instruction count spent on addressing.
//
// The A tile is transposed as it is written into shared memory, from (BM x BK)
// to (BK x BM). In the inner loop each thread wants TM consecutive rows of A at
// one value of k. In the original layout those are BK apart; transposed they
// are adjacent, so that read becomes a float4 as well.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void vectorized(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BK * BM];  // transposed
  __shared__ float Bs[BK * BN];

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // With BM*BN/(TM*TN) threads, each tile is exactly one float4 per thread.
  const int innerRowA = threadIdx.x / (BK / 4);
  const int innerColA = threadIdx.x % (BK / 4);
  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);

  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  float acc[TM][TN] = {0.0f};
  float regM[TM], regN[TN];

  for (int t = 0; t < K; t += BK) {
    // One 16-byte load, scattered into four rows of the transposed tile. The
    // scatter costs four shared stores, paid once per tile rather than once per
    // use, and it is what makes the inner loop's reads contiguous.
    const float4 a = *reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4]);
    As[(innerColA * 4 + 0) * BM + innerRowA] = a.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a.w;

    // B is already in the orientation the inner loop wants, so it goes straight
    // across as one float4 store.
    *reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4]) =
        *reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4]);
    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
      // Pull the operands into registers once, then reuse each of them TN and
      // TM times respectively in the rank-one update below.
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[dot * BM + threadRow * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) regN[j] = Bs[dot * BN + threadCol * TN + j];

#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; j += 4)
      *reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + j]) =
          *reinterpret_cast<float4*>(&acc[i][j]);
}

}  // namespace

void launch_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  const dim3 block(32, 32);
  const dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
  naive<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAUNCH();
}

void launch_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
  const dim3 block(32, 32);
  const dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
  coalesced<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAUNCH();
}

void launch_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int BS = 32;
  const dim3 block(BS * BS);
  const dim3 grid(ceil_div(N, BS), ceil_div(M, BS));
  tiled<BS><<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAUNCH();
}

void launch_register_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
  const dim3 block((BM * BN) / TM);
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  register_tiled<BM, BN, BK, TM><<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAUNCH();
}

void launch_vectorized(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  const dim3 block((BM * BN) / (TM * TN));  // 256 threads
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_LAUNCH();
}

}  // namespace sgemm

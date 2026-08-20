// Launchers for the matmul ladder. Each one computes C = A B for row-major
// A (M x K), B (K x N), C (M x N), and they differ only in how they move data
// through the memory hierarchy.
#pragma once

namespace sgemm {

// One thread per output element. Threads within a warp walk down a column of
// C, so each warp touches 32 different cache lines per store.
void launch_naive(const float* A, const float* B, float* C, int M, int N, int K);

// Identical arithmetic, but the thread-to-output mapping is transposed so a
// warp covers 32 adjacent elements of a row instead. Only the indexing changes.
void launch_coalesced(const float* A, const float* B, float* C, int M, int N, int K);

// Stages square tiles of A and B in shared memory, so each loaded value is
// reused by a whole tile of threads rather than read from global memory again.
void launch_tiled(const float* A, const float* B, float* C, int M, int N, int K);

// Gives each thread a column of eight outputs held in registers, so one load
// from shared memory feeds eight fused multiply-adds instead of one.
// Requires M, N and K to be multiples of 64, 64 and 8 respectively.
void launch_register_tiled(const float* A, const float* B, float* C, int M, int N, int K);

}  // namespace sgemm

// Compares fused attention against the three-kernel version that writes the
// score matrix out to global memory.
//
// Two things are being measured. Time, which is the usual reason to fuse, and
// scratch memory, which is the reason FlashAttention exists: the unfused path
// needs an N x N buffer and the fused path needs none, so the gap widens with
// the square of the sequence length.
#include "../../include/attention.h"
#include "../../include/util.cuh"
#include "../../ref/reference.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

double time_it(void (*run)(), int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) run();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s));
  for (int i = 0; i < iters; ++i) run();
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms / iters;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    worst = std::fmax(worst, std::fabs((double)a[i] - b[i]));
  return worst;
}

// Globals so the timing helper can take a plain function pointer.
const float *gQ, *gK, *gV;
float *gO, *gScratch;
int gN, gD;
float gScale;

void run_unfused() {
  attention::launch_unfused(gQ, gK, gV, gO, gScratch, gN, gD, gScale);
}
void run_fused() { attention::launch_fused(gQ, gK, gV, gO, gN, gD, gScale); }

int gBR = 64, gBC = 32;
void run_fused_config() {
  attention::launch_fused_config(gQ, gK, gV, gO, gN, gD, gScale, gBR, gBC);
}

}  // namespace

int main(int argc, char** argv) {
  const int N = (argc > 1) ? std::atoi(argv[1]) : 2048;
  const int d = 64;
  const int iters = 20, warmup = 5;
  const float scale = 1.0f / std::sqrt((float)d);

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device            %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("problem           N=%d, d=%d\n", N, d);

  const double scratch_mb = (double)N * N * sizeof(float) / (1024.0 * 1024.0);
  const double io_mb = 3.0 * N * d * sizeof(float) / (1024.0 * 1024.0);
  std::printf("scratch, unfused  %.1f MB   (N x N scores)\n", scratch_mb);
  std::printf("scratch, fused    0.0 MB\n");
  std::printf("Q, K, V together  %.1f MB\n\n", io_mb);

  std::mt19937 rng(0);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> hQ((size_t)N * d), hK((size_t)N * d), hV((size_t)N * d);
  for (auto* buf : {&hQ, &hK, &hV})
    for (auto& x : *buf) x = normal(rng);

  float *dQ, *dK, *dV, *dO, *dScratch;
  CUDA_CHECK(cudaMalloc(&dQ, hQ.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK, hK.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV, hV.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dO, hQ.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dScratch, (size_t)N * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(float), cudaMemcpyHostToDevice));

  gQ = dQ; gK = dK; gV = dV; gO = dO; gScratch = dScratch;
  gN = N; gD = d; gScale = scale;

  std::vector<float> out_unfused((size_t)N * d), out_fused((size_t)N * d);

  CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * sizeof(float)));
  run_unfused();
  CUDA_CHECK(cudaMemcpy(out_unfused.data(), dO, out_unfused.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * sizeof(float)));
  run_fused();
  CUDA_CHECK(cudaMemcpy(out_fused.data(), dO, out_fused.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Both kernels sum N terms in float32, so the tolerance scales the same way
  // the accumulation error does.
  const double tol = 8.0 * std::sqrt((double)N) * 1.192e-7;

  // The CPU reference is O(N^2 d) on one core, so it only runs at the smaller
  // sizes. Above that the two GPU paths are still checked against each other,
  // which is the comparison that would catch a fusion bug anyway.
  double err_unfused = -1.0, err_fused = -1.0;
  if (N <= 1024) {
    std::vector<float> want((size_t)N * d);
    ref::attention_naive(hQ.data(), hK.data(), hV.data(), want.data(), N, d, scale);
    err_unfused = max_abs(out_unfused, want);
    err_fused = max_abs(out_fused, want);
    std::printf("vs CPU reference  unfused %.3e   fused %.3e   (tolerance %.3e)\n",
                err_unfused, err_fused, tol);
  } else {
    std::printf("vs CPU reference  skipped, N > 1024\n");
  }
  const double err_pair = max_abs(out_fused, out_unfused);
  std::printf("fused vs unfused  %.3e   (tolerance %.3e)\n\n", err_pair, tol);

  const double ms_unfused = time_it(run_unfused, warmup, iters);
  const double ms_fused = time_it(run_fused, warmup, iters);

  // 2 * N^2 * d for the scores, the same again for the value product.
  const double flops = 4.0 * (double)N * N * d;
  std::printf("%-16s %9.3f ms  %8.1f GFLOP/s\n", "unfused", ms_unfused,
              flops / (ms_unfused * 1e6));
  std::printf("%-16s %9.3f ms  %8.1f GFLOP/s\n", "fused", ms_fused,
              flops / (ms_fused * 1e6));
  std::printf("%-16s %9.2fx\n", "speedup", ms_unfused / ms_fused);

  // Tile-shape sweep. The default shape is one point on this grid, and the
  // fused kernel losing at small N is a question about occupancy, so the useful
  // thing is to measure the alternatives rather than reason about them.
  std::printf("\ntile shape sweep (BR = rows per block, BC = keys staged)\n");
  std::printf("%6s %6s %12s %12s %10s\n", "BR", "BC", "smem KB", "ms", "vs default");
  const int shapes[][2] = {{64, 16}, {64, 32}, {64, 64},
                           {128, 16}, {128, 32}, {128, 64},
                           {256, 16}, {256, 32}};
  double best_ms = ms_fused;
  int best_br = 64, best_bc = 32;
  for (const auto& sh : shapes) {
    gBR = sh[0];
    gBC = sh[1];
    CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * sizeof(float)));
    if (!attention::launch_fused_config(dQ, dK, dV, dO, N, d, scale, gBR, gBC)) continue;

    std::vector<float> probe((size_t)N * d);
    CUDA_CHECK(cudaMemcpy(probe.data(), dO, probe.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const double e = max_abs(probe, out_unfused);

    const double t = time_it(run_fused_config, warmup, iters);
    const double smem_kb = sh[1] * 64.0 * 2.0 * sizeof(float) / 1024.0;
    std::printf("%6d %6d %12.1f %12.3f %9.2fx%s\n", sh[0], sh[1], smem_kb, t,
                ms_fused / t, e < tol ? "" : "   <-- WRONG");
    if (t < best_ms && e < tol) {
      best_ms = t;
      best_br = sh[0];
      best_bc = sh[1];
    }
  }
  std::printf("best: BR=%d BC=%d at %.3f ms, %.2fx the default shape\n", best_br,
              best_bc, best_ms, ms_fused / best_ms);
  std::printf("against unfused: %.2fx\n", ms_unfused / best_ms);

  const bool ok = err_pair < tol && (err_fused < 0 || err_fused < tol);

  char path[128];
  std::snprintf(path, sizeof(path), "results/attention_%d.json", N);
  FILE* f = std::fopen(path, "w");
  if (f) {
    std::fprintf(f, "{\n  \"device\": \"%s\",\n  \"N\": %d, \"d\": %d, \"iters\": %d,\n",
                 prop.name, N, d, iters);
    std::fprintf(f, "  \"scratch_mb\": {\"unfused\": %.1f, \"fused\": 0.0},\n", scratch_mb);
    std::fprintf(f, "  \"tolerance\": %.3e,\n", tol);
    std::fprintf(f, "  \"err_vs_reference\": {\"unfused\": %.3e, \"fused\": %.3e},\n",
                 err_unfused, err_fused);
    std::fprintf(f, "  \"err_fused_vs_unfused\": %.3e,\n", err_pair);
    std::fprintf(f, "  \"unfused\": {\"ms\": %.4f, \"gflops\": %.1f},\n", ms_unfused,
                 flops / (ms_unfused * 1e6));
    std::fprintf(f, "  \"fused\": {\"ms\": %.4f, \"gflops\": %.1f},\n", ms_fused,
                 flops / (ms_fused * 1e6));
    std::fprintf(f, "  \"speedup\": %.3f,\n", ms_unfused / ms_fused);
    std::fprintf(f, "  \"best_shape\": {\"BR\": %d, \"BC\": %d, \"ms\": %.4f, "
                    "\"speedup_vs_unfused\": %.3f},\n", best_br, best_bc, best_ms,
                 ms_unfused / best_ms);
    std::fprintf(f, "  \"correct\": %s\n}\n", ok ? "true" : "false");
    std::fclose(f);
    std::printf("\nwrote %s\n", path);
  }

  if (!ok) std::printf("\nCORRECTNESS FAILED\n");

  CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK)); CUDA_CHECK(cudaFree(dV));
  CUDA_CHECK(cudaFree(dO)); CUDA_CHECK(cudaFree(dScratch));
  return ok ? 0 : 1;
}

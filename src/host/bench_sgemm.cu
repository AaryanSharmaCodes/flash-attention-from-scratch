// Runs the matmul ladder, checks each kernel against cuBLAS, and writes the
// timings to results/sgemm.json. Every number quoted in the README comes from
// this file rather than being typed in by hand.
#include "../../include/sgemm.h"
#include "../../include/util.cuh"

#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Result {
  std::string name;
  double ms;
  double gflops;
  double pct_of_cublas;
  double max_abs_err;
  double frobenius_rel;
  bool correct;
};

// cuBLAS is column-major. Handing it B and A in that order, with the dimensions
// swapped, makes it compute C^T = B^T A^T -- which is exactly our row-major
// C = A B, with no transposes or extra copies anywhere.
void cublas_sgemm(cublasHandle_t h, const float* A, const float* B, float* C,
                  int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

// Timing needs the warmup: the first launch pays for module load and JIT, which
// on a cold context can be milliseconds and would swamp the measurement.
double time_kernel(void (*fn)(const float*, const float*, float*, int, int, int),
                   const float* A, const float* B, float* C, int M, int N, int K,
                   int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn(A, B, C, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) fn(A, B, C, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

// Error against cuBLAS, measured over the whole matrix rather than element by
// element.
//
// The first version of this compared each element to its own reference value.
// That reports nonsense here: entries of C are sums of K products that can
// cancel to nearly zero, and at K=512 the smallest entry of C is 3.7e-06
// against a mean of 18.0. Dividing an ordinary rounding error by an entry that
// small produced 1.6e-01 and flagged every kernel as wrong, including ones that
// agreed with cuBLAS bit for bit.
//
// The Frobenius relative error, ||got - want|| / ||want||, is the usual measure
// for GEMM for exactly this reason: it judges the result against the scale of
// the whole matrix, so a near-zero entry contributes in proportion to how much
// it actually matters.
double frobenius_rel_error(const std::vector<float>& got, const std::vector<float>& want) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = (double)got[i] - want[i];
    num += d * d;
    den += (double)want[i] * want[i];
  }
  return std::sqrt(num) / std::sqrt(den);
}

double max_abs_error(const std::vector<float>& got, const std::vector<float>& want) {
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i)
    worst = std::fmax(worst, std::fabs((double)got[i] - want[i]));
  return worst;
}

}  // namespace

int main(int argc, char** argv) {
  const int size = (argc > 1) ? std::atoi(argv[1]) : 2048;
  const int M = size, N = size, K = size;
  const int iters = 20, warmup = 5;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device            %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("problem           %d x %d x %d\n\n", M, N, K);

  std::mt19937 rng(0);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
  for (auto& x : hA) x = normal(rng);
  for (auto& x : hB) x = normal(rng);

  float *dA, *dB, *dC, *dRef;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dRef, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  cublasCreate(&handle);

  // Reference result and reference timing, both from cuBLAS.
  cublas_sgemm(handle, dA, dB, dRef, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> want((size_t)M * N);
  CUDA_CHECK(cudaMemcpy(want.data(), dRef, want.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < warmup; ++i) cublas_sgemm(handle, dA, dB, dRef, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s));
  for (int i = 0; i < iters; ++i) cublas_sgemm(handle, dA, dB, dRef, M, N, K);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float cublas_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cublas_total, s, e));
  const double cublas_ms = cublas_total / iters;

  // 2MNK: one multiply and one add per inner-loop step.
  const double flops = 2.0 * M * N * K;
  const double cublas_gflops = flops / (cublas_ms * 1e6);

  struct Entry {
    const char* name;
    void (*fn)(const float*, const float*, float*, int, int, int);
  };
  const Entry ladder[] = {
      {"naive", sgemm::launch_naive},
      {"coalesced", sgemm::launch_coalesced},
      {"tiled", sgemm::launch_tiled},
      {"register_tiled", sgemm::launch_register_tiled},
      {"vectorized", sgemm::launch_vectorized},
  };

  std::vector<Result> results;
  std::vector<float> got((size_t)M * N);

  for (const Entry& entry : ladder) {
    CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
    entry.fn(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const double abs_err = max_abs_error(got, want);
    const double frob = frobenius_rel_error(got, want);

    // float32 accumulation over K terms lands around 4e-07 here, so 1e-05
    // leaves two orders of magnitude of headroom while still catching a kernel
    // that is genuinely computing the wrong thing.
    const bool correct = frob < 1e-5;

    const double ms = time_kernel(entry.fn, dA, dB, dC, M, N, K, warmup, iters);
    const double gflops = flops / (ms * 1e6);
    results.push_back({entry.name, ms, gflops, 100.0 * gflops / cublas_gflops, abs_err, frob, correct});

    std::printf("%-16s %8.3f ms  %8.1f GFLOP/s  %5.1f%% of cuBLAS  frob %.2e  abs %.2e %s\n",
                entry.name, ms, gflops, results.back().pct_of_cublas, frob, abs_err,
                correct ? "" : "  <-- WRONG");
  }
  std::printf("%-16s %8.3f ms  %8.1f GFLOP/s\n", "cuBLAS", cublas_ms, cublas_gflops);

  char path[128];
  std::snprintf(path, sizeof(path), "results/sgemm_%d.json", size);
  FILE* f = std::fopen(path, "w");
  if (f) {
    std::fprintf(f, "{\n  \"device\": \"%s\",\n  \"sm\": \"%d.%d\",\n", prop.name, prop.major, prop.minor);
    std::fprintf(f, "  \"M\": %d, \"N\": %d, \"K\": %d,\n  \"iters\": %d,\n", M, N, K, iters);
    std::fprintf(f, "  \"cublas\": {\"ms\": %.4f, \"gflops\": %.1f},\n  \"kernels\": [\n", cublas_ms, cublas_gflops);
    for (size_t i = 0; i < results.size(); ++i) {
      const Result& r = results[i];
      std::fprintf(f,
                   "    {\"name\": \"%s\", \"ms\": %.4f, \"gflops\": %.1f, "
                   "\"pct_of_cublas\": %.1f, \"frobenius_rel\": %.3e, "
                   "\"max_abs_err\": %.3e, \"correct\": %s}%s\n",
                   r.name.c_str(), r.ms, r.gflops, r.pct_of_cublas, r.frobenius_rel,
                   r.max_abs_err, r.correct ? "true" : "false", i + 1 < results.size() ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    std::printf("\nwrote %s\n", path);
  }

  bool all_correct = true;
  for (const Result& r : results) all_correct = all_correct && r.correct;

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dRef));
  return all_correct ? 0 : 1;
}

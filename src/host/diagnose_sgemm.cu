// Locates a matmul disagreement rather than just reporting one.
//
// The benchmark compares against cuBLAS and reports a per-element relative
// error. When every kernel in the ladder reports the same error, that number is
// not telling us about the kernels, so this tool checks the same results three
// different ways:
//
//   1. against a double-precision CPU matmul, which is as close to the true
//      answer as anything here gets, so cuBLAS is on trial too;
//   2. with several error measures at once, since a per-element relative error
//      is meaningless where the true value is near zero;
//   3. against each other, because kernels that agree exactly with one another
//      but not with the reference indicate the reference is at fault.
#include "../../include/sgemm.h"
#include "../../include/util.cuh"

#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

// Accumulating in double gives a baseline whose own error is far below the
// float32 differences being investigated.
void matmul_f64(const std::vector<float>& A, const std::vector<float>& B,
                std::vector<double>& C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += (double)A[i * K + k] * (double)B[k * N + j];
      C[(size_t)i * N + j] = acc;
    }
}

struct Errors {
  double max_abs;         // largest absolute difference anywhere
  double max_rel_elem;    // per-element relative error, the benchmark's measure
  double frobenius_rel;   // ||got - want|| / ||want||, the usual measure for GEMM
  double worst_got, worst_want;
  int worst_index;
};

Errors compare(const std::vector<float>& got, const std::vector<double>& want) {
  Errors e{0, 0, 0, 0, 0, 0};
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::fabs(got[i] - want[i]);
    e.max_abs = std::fmax(e.max_abs, d);
    num += d * d;
    den += want[i] * want[i];

    const double denom = std::fabs(want[i]) > 1e-4 ? std::fabs(want[i]) : 1e-4;
    const double rel = d / denom;
    if (rel > e.max_rel_elem) {
      e.max_rel_elem = rel;
      e.worst_got = got[i];
      e.worst_want = want[i];
      e.worst_index = (int)i;
    }
  }
  e.frobenius_rel = std::sqrt(num) / std::sqrt(den);
  return e;
}

void report(const char* name, const Errors& e) {
  std::printf("%-16s abs %.3e | per-elem rel %.3e | frobenius %.3e\n", name,
              e.max_abs, e.max_rel_elem, e.frobenius_rel);
  std::printf("%-16s worst element C[%d]: got %+.6e  true %+.6e\n", "", e.worst_index,
              e.worst_got, e.worst_want);
}

}  // namespace

int main(int argc, char** argv) {
  // Small, because the double-precision reference is O(N^3) on one CPU core.
  const int size = (argc > 1) ? std::atoi(argv[1]) : 512;
  const int M = size, N = size, K = size;

  std::mt19937 rng(0);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
  for (auto& x : hA) x = normal(rng);
  for (auto& x : hB) x = normal(rng);

  std::printf("problem %d x %d x %d\n\n", M, N, K);
  std::vector<double> truth((size_t)M * N);
  matmul_f64(hA, hB, truth, M, N, K);

  // Scale of the true result, for judging whether an absolute error is large.
  double mean_abs = 0.0, max_abs_true = 0.0, min_abs_true = 1e300;
  for (double v : truth) {
    mean_abs += std::fabs(v);
    max_abs_true = std::fmax(max_abs_true, std::fabs(v));
    min_abs_true = std::fmin(min_abs_true, std::fabs(v));
  }
  mean_abs /= truth.size();
  std::printf("true C: mean |C| %.4f, max |C| %.4f, min |C| %.3e\n", mean_abs,
              max_abs_true, min_abs_true);
  std::printf("float32 eps at the mean magnitude is about %.3e\n\n",
              mean_abs * 1.19e-7 * std::sqrt((double)K));

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.0f, beta = 0.0f;

  std::vector<std::vector<float>> outputs;
  std::vector<const char*> names;

  // cuBLAS first, so it is measured against the same truth as everything else.
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    std::vector<float> got((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    report("cuBLAS", compare(got, truth));
    outputs.push_back(got);
    names.push_back("cuBLAS");
  }

  struct Entry {
    const char* name;
    void (*fn)(const float*, const float*, float*, int, int, int);
  };
  const Entry ladder[] = {
      {"naive", sgemm::launch_naive},
      {"coalesced", sgemm::launch_coalesced},
      {"tiled", sgemm::launch_tiled},
      {"register_tiled", sgemm::launch_register_tiled},
  };

  for (const Entry& entry : ladder) {
    CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
    entry.fn(dA, dB, dC, M, N, K);
    std::vector<float> got((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    report(entry.name, compare(got, truth));
    outputs.push_back(got);
    names.push_back(entry.name);
  }

  // Kernels that agree with each other bit for bit, while disagreeing with the
  // reference, would mean the reference is the odd one out.
  std::printf("\npairwise max |difference| between implementations\n");
  std::printf("%-16s", "");
  for (const char* n : names) std::printf("%14s", n);
  std::printf("\n");
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::printf("%-16s", names[i]);
    for (size_t j = 0; j < outputs.size(); ++j) {
      double d = 0.0;
      for (size_t t = 0; t < outputs[i].size(); ++t)
        d = std::fmax(d, std::fabs((double)outputs[i][t] - outputs[j][t]));
      std::printf("%14.3e", d);
    }
    std::printf("\n");
  }

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}

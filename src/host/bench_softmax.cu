// Runs the softmax ladder, checks each kernel against the CPU reference, and
// writes results/softmax_<rows>x<cols>.json.
//
// These kernels are memory bound, so the number that matters is how close the
// achieved bandwidth gets to what the card can sustain. Effective bandwidth is
// computed from the traffic the problem actually requires -- one read of the
// input and one write of the output -- so a kernel that reads the row three
// times shows up as a fraction of peak rather than being flattered by counting
// its own redundant loads.
#include "../../include/softmax.h"
#include "../../include/util.cuh"
#include "../../ref/reference.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

struct Result {
  std::string name;
  double ms;
  double gbs;
  double pct_of_peak;
  double max_abs_err;
  bool correct;
};

double time_kernel(void (*fn)(const float*, float*, int, int), const float* in,
                   float* out, int rows, int cols, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn(in, out, rows, cols);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) fn(in, out, rows, cols);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const int cols = (argc > 2) ? std::atoi(argv[2]) : 4096;
  const int iters = 50, warmup = 10;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  // Memory clock is in kHz and the bus is DDR, hence the factor of two.
  const double peak_gbs =
      2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
  std::printf("device            %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("peak bandwidth    %.0f GB/s\n", peak_gbs);
  std::printf("problem           %d rows x %d cols\n", rows, cols);

  std::mt19937 rng(0);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> hIn((size_t)rows * cols);
  // Shift the values well away from zero so that a kernel which forgets to
  // subtract the row max overflows and is caught, rather than quietly passing.
  for (auto& x : hIn) x = normal(rng) * 4.0f + 20.0f;

  std::vector<float> want((size_t)rows * cols);
  ref::softmax_rows(hIn.data(), want.data(), rows, cols);

  float *dIn, *dOut;
  CUDA_CHECK(cudaMalloc(&dIn, hIn.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dOut, hIn.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), hIn.size() * sizeof(float), cudaMemcpyHostToDevice));

  struct Entry {
    const char* name;
    void (*fn)(const float*, float*, int, int);
  };
  const Entry ladder[] = {
      {"naive", softmax::launch_naive},
      {"shared_tree", softmax::launch_shared},
      {"warp_shuffle", softmax::launch_warp},
      {"online", softmax::launch_online},
  };

  const double bytes = 2.0 * (double)rows * cols * sizeof(float);

  // The kernels sum the row in float32, so they carry an error of roughly
  // sqrt(cols) * eps no matter how carefully they are written -- 7.6e-06 at
  // 4096 columns. A fixed tolerance would either pass everything at 1024 or
  // fail everything at 4096, so the bound scales the same way the error does.
  // The factor of eight is headroom for the order the additions happen in and
  // for __expf being the approximate exponential.
  const double tol = 8.0 * std::sqrt((double)cols) * 1.192e-7;
  std::printf("tolerance         %.3e  (8 * sqrt(cols) * eps)\n\n", tol);
  std::vector<Result> results;
  std::vector<float> got((size_t)rows * cols);

  for (const Entry& entry : ladder) {
    CUDA_CHECK(cudaMemset(dOut, 0, hIn.size() * sizeof(float)));
    entry.fn(dIn, dOut, rows, cols);
    CUDA_CHECK(cudaMemcpy(got.data(), dOut, got.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double err = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
      err = std::fmax(err, std::fabs((double)got[i] - want[i]));

    const double ms = time_kernel(entry.fn, dIn, dOut, rows, cols, warmup, iters);
    const double gbs = bytes / (ms * 1e6);
    results.push_back({entry.name, ms, gbs, 100.0 * gbs / peak_gbs, err, err < tol});

    std::printf("%-16s %8.3f ms  %8.1f GB/s  %5.1f%% of peak  max err %.2e %s\n",
                entry.name, ms, gbs, results.back().pct_of_peak, err,
                results.back().correct ? "" : "  <-- WRONG");
  }

  char path[160];
  std::snprintf(path, sizeof(path), "results/softmax_%dx%d.json", rows, cols);
  FILE* f = std::fopen(path, "w");
  if (f) {
    std::fprintf(f, "{\n  \"device\": \"%s\",\n  \"peak_gbs\": %.0f,\n", prop.name, peak_gbs);
    std::fprintf(f, "  \"rows\": %d, \"cols\": %d, \"iters\": %d, \"tolerance\": %.3e,\n  \"kernels\": [\n", rows, cols, iters, tol);
    for (size_t i = 0; i < results.size(); ++i) {
      const Result& r = results[i];
      std::fprintf(f,
                   "    {\"name\": \"%s\", \"ms\": %.4f, \"gbs\": %.1f, "
                   "\"pct_of_peak\": %.1f, \"max_abs_err\": %.3e, \"correct\": %s}%s\n",
                   r.name.c_str(), r.ms, r.gbs, r.pct_of_peak, r.max_abs_err,
                   r.correct ? "true" : "false", i + 1 < results.size() ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    std::printf("\nwrote %s\n", path);
  }

  bool all_correct = true;
  for (const Result& r : results) all_correct = all_correct && r.correct;

  CUDA_CHECK(cudaFree(dIn));
  CUDA_CHECK(cudaFree(dOut));
  return all_correct ? 0 : 1;
}

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel launches fail asynchronously, so an error raised by one kernel is
// usually reported at whatever call happens next. Checking every call keeps
// the blame attached to the right line.
#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err__ = (expr);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,     \
                   #expr, cudaGetErrorString(err__));                        \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// Call after a launch to surface configuration errors (too many threads, too
// much shared memory) that cudaGetLastError reports immediately.
#define CUDA_CHECK_LAUNCH()                                                  \
  do {                                                                       \
    CUDA_CHECK(cudaGetLastError());                                          \
    CUDA_CHECK(cudaDeviceSynchronize());                                      \
  } while (0)

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

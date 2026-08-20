// Checks the streaming attention against the textbook version on the CPU.
// The point is to pin down the maths on hardware that is easy to debug, so
// that a later mismatch on the GPU can only be a CUDA problem.
#include "../ref/reference.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float worst = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) worst = std::fmax(worst, std::fabs(a[i] - b[i]));
  return worst;
}

static int check(const char* name, float diff, float tol) {
  const bool ok = diff <= tol;
  std::printf("%-46s max|diff| = %.3e  %s\n", name, diff, ok ? "ok" : "FAILED");
  return ok ? 0 : 1;
}

int main() {
  std::mt19937 rng(0);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  int failures = 0;

  for (int N : {8, 64, 129}) {
    for (int d : {16, 64}) {
      std::vector<float> Q((size_t)N * d), K((size_t)N * d), V((size_t)N * d);
      for (auto* buf : {&Q, &K, &V})
        for (auto& x : *buf) x = normal(rng);

      const float scale = 1.0f / std::sqrt((float)d);
      std::vector<float> want((size_t)N * d), got((size_t)N * d);
      ref::attention_naive(Q.data(), K.data(), V.data(), want.data(), N, d, scale);

      // Block width should not change the answer. Widths that do not divide N
      // are included on purpose, since the ragged last block is where a
      // streaming implementation usually goes wrong.
      for (int bc : {1, 7, 32, 256}) {
        ref::attention_online(Q.data(), K.data(), V.data(), got.data(), N, d, scale, bc);
        char label[96];
        std::snprintf(label, sizeof(label), "online vs naive  N=%d d=%d block=%d", N, d, bc);
        failures += check(label, max_abs_diff(want, got), 1e-5f);
      }
    }
  }

  // Scores far from zero are the case the max subtraction exists to handle.
  // Run it without the guard and this row comes back as NaN.
  {
    const int N = 32, d = 32;
    std::vector<float> Q((size_t)N * d), K((size_t)N * d), V((size_t)N * d);
    for (auto& x : Q) x = normal(rng) + 30.0f;
    for (auto& x : K) x = normal(rng) + 30.0f;
    for (auto& x : V) x = normal(rng);

    std::vector<float> want((size_t)N * d), got((size_t)N * d);
    const float scale = 1.0f / std::sqrt((float)d);
    ref::attention_naive(Q.data(), K.data(), V.data(), want.data(), N, d, scale);
    ref::attention_online(Q.data(), K.data(), V.data(), got.data(), N, d, scale, 16);

    bool finite = true;
    for (float x : got) finite = finite && std::isfinite(x);
    std::printf("%-46s %s\n", "large scores stay finite", finite ? "ok" : "FAILED");
    failures += finite ? 0 : 1;
    failures += check("online vs naive  large scores", max_abs_diff(want, got), 1e-4f);
  }

  std::printf("\n%s\n", failures ? "FAILURES" : "all reference checks passed");
  return failures ? 1 : 0;
}

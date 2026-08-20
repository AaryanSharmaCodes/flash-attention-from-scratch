#include "reference.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ref {

void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
      C[i * N + j] = acc;
    }
  }
}

void softmax_rows(const float* in, float* out, int rows, int cols) {
  // The exponentials and the sum are accumulated in double even though the
  // result is stored as float.
  //
  // This started as a plain float32 sum and it made the reference useless as a
  // referee. Summing n floats sequentially carries a relative error around
  // sqrt(n) * eps, which at 4096 columns is 7.6e-06 -- the same size as the
  // disagreement being measured. The GPU kernels that summed in a different
  // order looked wrong, and the one that happened to sum in the same order as
  // this loop looked perfect, when in truth neither was more accurate than the
  // other. Accumulating in double drops the reference's own error to around
  // 1e-15 and lets the comparison mean something.
  std::vector<double> exps(cols);
  for (int i = 0; i < rows; ++i) {
    const float* r = in + (size_t)i * cols;
    float* o = out + (size_t)i * cols;

    // Subtracting the row max keeps every exp() argument at or below zero.
    // Without it a score of 90 or so overflows float and the row becomes NaN.
    double m = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < cols; ++j) m = std::max(m, (double)r[j]);

    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      exps[j] = std::exp((double)r[j] - m);
      sum += exps[j];
    }
    for (int j = 0; j < cols; ++j) o[j] = (float)(exps[j] / sum);
  }
}

void attention_naive(const float* Q, const float* K, const float* V, float* out,
                     int N, int d, float scale) {
  std::vector<float> scores((size_t)N * N);

  // S = scale * Q K^T. Note K is (N x d) and we want its transpose, so the
  // inner loop walks both operands along d.
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < d; ++k) acc += Q[i * d + k] * K[j * d + k];
      scores[(size_t)i * N + j] = acc * scale;
    }
  }

  softmax_rows(scores.data(), scores.data(), N, N);

  // out = P V
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < d; ++k) {
      float acc = 0.0f;
      for (int j = 0; j < N; ++j) acc += scores[(size_t)i * N + j] * V[j * d + k];
      out[i * d + k] = acc;
    }
  }
}

void attention_online(const float* Q, const float* K, const float* V, float* out,
                      int N, int d, float scale, int block_c) {
  const float NEG_INF = -std::numeric_limits<float>::infinity();
  std::vector<float> acc(d);
  std::vector<float> s(block_c);

  for (int i = 0; i < N; ++i) {
    // Running state for this query row: the largest score seen so far, the
    // sum of exponentials so far, and the partially accumulated output. All
    // three are held relative to the current max, which is why they need
    // rescaling every time the max moves.
    float m = NEG_INF;
    float l = 0.0f;
    std::fill(acc.begin(), acc.end(), 0.0f);

    for (int j0 = 0; j0 < N; j0 += block_c) {
      const int jn = std::min(block_c, N - j0);

      float m_block = NEG_INF;
      for (int t = 0; t < jn; ++t) {
        float dot = 0.0f;
        for (int k = 0; k < d; ++k) dot += Q[i * d + k] * K[(j0 + t) * d + k];
        s[t] = dot * scale;
        m_block = std::max(m_block, s[t]);
      }

      const float m_new = std::max(m, m_block);

      // Everything accumulated so far was scaled by exp(-m). Moving the max to
      // m_new means multiplying it all by exp(m - m_new), a factor <= 1. This
      // single correction is what lets the pass be streaming instead of
      // two-pass, and it is the only interesting line in the algorithm.
      const float correction = (m == NEG_INF) ? 0.0f : std::exp(m - m_new);
      l *= correction;
      for (int k = 0; k < d; ++k) acc[k] *= correction;

      for (int t = 0; t < jn; ++t) {
        const float p = std::exp(s[t] - m_new);
        l += p;
        for (int k = 0; k < d; ++k) acc[k] += p * V[(j0 + t) * d + k];
      }

      m = m_new;
    }

    // The division by l is deferred to the very end, so it happens once per
    // row rather than once per block.
    for (int k = 0; k < d; ++k) out[i * d + k] = acc[k] / l;
  }
}

}  // namespace ref

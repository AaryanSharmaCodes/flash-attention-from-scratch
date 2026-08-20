// Scaled dot-product attention, forward pass, single head, no mask.
//
//     O = softmax(scale * Q K^T) V,  with Q, K, V and O all (N x d) row-major.
//
// Two implementations, differing in whether the N x N score matrix is ever
// written down. That is the whole point of the comparison: the arithmetic is
// identical and the memory behaviour is not.
#pragma once

namespace attention {

// Three kernel launches with the scores materialised in global memory between
// them: S = scale * Q K^T, then a row-wise softmax over S, then O = S V.
// Needs an N x N scratch buffer, which is 64 MB at N = 4096 and grows with the
// square of the sequence length.
void launch_unfused(const float* Q, const float* K, const float* V, float* O,
                    float* scratch, int N, int d, float scale);

// One kernel. Each thread owns a query row and walks the keys in tiles, keeping
// a running maximum, a running sum and a running output, so no part of S is
// ever stored. Scratch requirement is zero and shared memory use does not
// depend on N.
//
// Supports d = 64. The head dimension is a template parameter so the per-thread
// arrays stay in registers, and only that instantiation is compiled.
void launch_fused(const float* Q, const float* K, const float* V, float* O,
                  int N, int d, float scale);

// The same fused kernel with the tile shape chosen at runtime, for sweeping.
// BR is threads per block and therefore query rows per block; BC is how many
// keys are staged in shared memory at once. Returns false for a combination
// that was not compiled.
bool launch_fused_config(const float* Q, const float* K, const float* V, float* O,
                         int N, int d, float scale, int BR, int BC);

}  // namespace attention

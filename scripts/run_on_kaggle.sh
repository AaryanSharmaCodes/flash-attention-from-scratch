#!/usr/bin/env bash
# Builds and runs everything on a machine that has a GPU. Intended to be run
# from a Kaggle or Colab cell, where the repo has just been cloned.
set -euo pipefail

echo "=== toolkit ==="
nvcc --version | tail -2
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

echo
echo "=== reference checks (CPU) ==="
make test

echo
echo "=== matmul ladder ==="
make bench

echo
echo "=== results written ==="
for f in results/*.json; do echo "--- $f"; cat "$f"; done

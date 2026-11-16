#!/usr/bin/env bash
# Build and run the C++ stopping-rule benchmark (macOS + Homebrew: brew install faiss).
# Run from the repo root after: uv run python cpp/export_inputs.py (and cpp/export_fitted.py)
# ./cpp/run.sh          stopping rule vs plain Faiss  -> results/cpp_stoprule_bench.csv
# ./cpp/run.sh fitted   static fitted rule vs plain   -> results/cpp_fitted_bench.csv
set -euo pipefail
BREW=$(brew --prefix)
mkdir -p build
clang++ -O3 -std=c++17 -march=native cpp/stoprule_bench.cpp -o build/stoprule_bench \
  -I"$BREW/include" -I"$BREW/opt/libomp/include" \
  -L"$BREW/lib" -L"$BREW/opt/libomp/lib" -lfaiss -lomp -Wl,-rpath,"$BREW/lib" -Wl,-rpath,"$BREW/opt/libomp/lib"
# single thread everywhere (Faiss uses OpenMP, the quantizer uses OpenBLAS)
MODE=${1:-}
OUT=results/cpp_${MODE:-stoprule}_bench.csv
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./build/stoprule_bench data/cpp "$OUT" $MODE

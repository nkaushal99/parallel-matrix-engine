# Parallel Matrix Engine

A staged C++17 matrix-multiplication project. It includes a correct naive baseline and a loop-reordered sequential kernel for comparing memory-access patterns.

## Layout

```text
include/     Shared matrix, kernel, and timing interfaces
src/         Naive and loop-reordered sequential kernels and drivers
```

## Build

Install a C++17 compiler and CMake.

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

These commands create `matrix_naive` and `matrix_reordered` in the `build/` directory.

## Run

```bash
./matrix_naive
./matrix_reordered
```

Both executables print machine-readable `key=value` output. The `result_sum` values must match; compare `time_ms` to measure the effect of the loop order.

## Loop-order optimisation

`matrix_naive` uses `i-j-k`, which reads down a column of row-major matrix `B` in its innermost loop. `matrix_reordered` uses `i-k-j`, making the innermost loop traverse contiguous elements of `B` and the output row of `C`. This improves spatial locality and cache-line use without changing the mathematical result.

## Planned progression

Subsequent commits will add, in order: cache blocking, OpenMP, and MPI. Each should retain correctness verification against the naive baseline.

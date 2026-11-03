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

`matrix_naive` uses `i-j-k`. With row-major storage, changing `k` in the inner loop reads `B[k][j]`: one element from each successive row of `B`. Those accesses are separated by an entire row, so most of each fetched cache line is not reused immediately.

`matrix_reordered` uses `i-k-j`. Its inner loop reads `B[k][j]` and writes `C[i][j]` while `j` increases, so both accesses move through contiguous elements in a row. A cache line fetched for `B[k][j]` or `C[i][j]` therefore supplies several upcoming iterations, improving spatial locality and cache-line utilisation without changing the mathematical result.

## Planned progression

Subsequent commits will add, in order: cache blocking, OpenMP, and MPI. Each should retain correctness verification against the naive baseline.

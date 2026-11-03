# Parallel Matrix Engine

A staged C++17 matrix-multiplication project. It includes naive, loop-reordered, and cache-blocked sequential kernels for comparing memory-access patterns.

## Layout

```text
include/     Shared matrix, kernel, and timing interfaces
src/         Naive, reordered, and cache-blocked kernels and drivers
```

## Build

Install a C++17 compiler and CMake.

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

These commands create `matrix_naive`, `matrix_reordered`, and `matrix_blocked` in the `build/` directory.

## Run

```bash
./matrix_naive
./matrix_reordered
./matrix_blocked
```

All three executables print machine-readable `key=value` output. The `result_sum` values must match; compare `time_ms` to measure the effect of each optimization.

## Loop-order optimisation

`matrix_naive` uses `i-j-k`. With row-major storage, changing `k` in the inner loop reads `B[k][j]`: one element from each successive row of `B`. Those accesses are separated by an entire row, so most of each fetched cache line is not reused immediately.

`matrix_reordered` uses `i-k-j`. Its inner loop reads `B[k][j]` and writes `C[i][j]` while `j` increases, so both accesses move through contiguous elements in a row. A cache line fetched for `B[k][j]` or `C[i][j]` therefore supplies several upcoming iterations, improving spatial locality and cache-line utilisation without changing the mathematical result.

## Cache blocking and tiling

`matrix_blocked` keeps the cache-friendly `i-k-j` inner-loop order, but performs it within 32×32 tiles. The outer `ii`, `jj`, and `kk` loops restrict one phase of the calculation to a small tile of `A`, `B`, and `C`.

Compared with processing whole matrices at once, those smaller working sets are more likely to remain in cache while their values are reused. In particular, a tile of `C` is updated repeatedly across the matching `k` tiles before moving to the next output tile, reducing cache evictions and memory traffic. This can improve performance beyond loop reordering alone, although the best tile size depends on the CPU cache hierarchy. Boundary checks retain correctness when a matrix dimension is not divisible by 32.

## Planned progression

Subsequent commits will add, in order: OpenMP and MPI. Each should retain correctness verification against the naive baseline.

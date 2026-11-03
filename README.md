# Parallel Matrix Engine

The initial baseline for a staged C++17 matrix-multiplication project. This commit intentionally contains only a correct, naive sequential kernel with `i-j-k` loop order. It is the reference implementation against which later cache and parallel variants will be validated.

## Layout

```text
include/     Shared matrix, kernel, and timing interfaces
src/         Matrix storage, naive kernel, and command-line driver
```

## Build

Install a C++17 compiler and CMake.

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

These commands create the `matrix_naive` executable in the `build/` directory.

## Run

```bash
./matrix_naive
```

The executable prints machine-readable `key=value` output, including elapsed time and a deterministic result sum. Preserve that sum when adding later variants to establish correctness.

## Planned progression

Subsequent commits will add, in order: loop reordering, cache blocking, OpenMP, and MPI. Each should retain correctness verification against this baseline.

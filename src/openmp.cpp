#include "kernels.hpp"

#include <stdexcept>

#include <omp.h>

Matrix multiply_openmp_blocked(const Matrix& a, const Matrix& b, std::size_t block_size) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matrix dimensions are incompatible");
    }
    if (block_size == 0) {
        throw std::invalid_argument("block size must be positive");
    }

    Matrix c(a.rows(), b.cols());
    const auto row_count = static_cast<std::ptrdiff_t>(a.rows());
    const auto column_count = static_cast<std::ptrdiff_t>(b.cols());
    const auto tile_size = static_cast<std::ptrdiff_t>(block_size);

    // Each iteration owns one C tile, so no two threads write the same elements.
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t ii = 0; ii < row_count; ii += tile_size) {
        for (std::ptrdiff_t jj = 0; jj < column_count; jj += tile_size) {
            const auto i_begin = static_cast<std::size_t>(ii);
            const auto i_end = std::min(i_begin + block_size, a.rows());
            const auto j_begin = static_cast<std::size_t>(jj);
            const auto j_end = std::min(j_begin + block_size, b.cols());

            for (std::size_t kk = 0; kk < a.cols(); kk += block_size) {
                const std::size_t k_end = std::min(kk + block_size, a.cols());
                for (std::size_t i = i_begin; i < i_end; ++i) {
                    for (std::size_t k = kk; k < k_end; ++k) {
                        const double a_value = a(i, k);
                        for (std::size_t j = j_begin; j < j_end; ++j) {
                            c(i, j) += a_value * b(k, j);
                        }
                    }
                }
            }
        }
    }
    return c;
}

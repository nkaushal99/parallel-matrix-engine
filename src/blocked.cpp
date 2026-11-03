#include "kernels.hpp"

#include <stdexcept>

Matrix multiply_blocked(const Matrix& a, const Matrix& b, std::size_t block_size) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matrix dimensions are incompatible");
    }
    if (block_size == 0) {
        throw std::invalid_argument("block size must be positive");
    }

    Matrix c(a.rows(), b.cols());
    for (std::size_t ii = 0; ii < a.rows(); ii += block_size) {
        const std::size_t i_end = std::min(ii + block_size, a.rows());
        for (std::size_t jj = 0; jj < b.cols(); jj += block_size) {
            const std::size_t j_end = std::min(jj + block_size, b.cols());
            for (std::size_t kk = 0; kk < a.cols(); kk += block_size) {
                const std::size_t k_end = std::min(kk + block_size, a.cols());
                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t k = kk; k < k_end; ++k) {
                        const double a_value = a(i, k);
                        for (std::size_t j = jj; j < j_end; ++j) {
                            c(i, j) += a_value * b(k, j);
                        }
                    }
                }
            }
        }
    }
    return c;
}

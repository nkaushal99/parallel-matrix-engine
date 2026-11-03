#include "kernels.hpp"

#include <stdexcept>

Matrix multiply_naive(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matrix dimensions are incompatible");
    }
    Matrix c(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            for (std::size_t k = 0; k < a.cols(); ++k) {
                c(i, j) += a(i, k) * b(k, j);
            }
        }
    }
    return c;
}

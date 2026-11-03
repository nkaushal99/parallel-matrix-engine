#include "kernels.hpp"

#include <stdexcept>

Matrix multiply_reordered(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matrix dimensions are incompatible");
    }

    Matrix c(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t k = 0; k < a.cols(); ++k) {
            const double a_value = a(i, k);
            for (std::size_t j = 0; j < b.cols(); ++j) {
                c(i, j) += a_value * b(k, j);
            }
        }
    }
    return c;
}

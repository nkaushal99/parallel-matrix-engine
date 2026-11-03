#include "matrix.hpp"

#include <cmath>

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

Matrix make_deterministic_matrix(std::size_t rows, std::size_t cols, unsigned seed) {
    Matrix matrix(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const auto value = static_cast<int>((i * 31 + j * 17 + seed) % 101) - 50;
            matrix(i, j) = value / 25.0;
        }
    }
    return matrix;
}

bool approximately_equal(const Matrix& left, const Matrix& right, double tolerance) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        return false;
    }
    for (std::size_t i = 0; i < left.rows() * left.cols(); ++i) {
        if (std::abs(left.data()[i] - right.data()[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

double sum_elements(const Matrix& matrix) {
    double sum = 0.0;
    for (std::size_t i = 0; i < matrix.rows() * matrix.cols(); ++i) {
        sum += matrix.data()[i];
    }
    return sum;
}

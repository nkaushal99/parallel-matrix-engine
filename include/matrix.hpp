#pragma once

#include <vector>

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0);

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

    double& operator()(std::size_t row, std::size_t col) { return data_[row * cols_ + col]; }
    const double& operator()(std::size_t row, std::size_t col) const {
        return data_[row * cols_ + col];
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

Matrix make_deterministic_matrix(std::size_t rows, std::size_t cols, unsigned seed);
bool approximately_equal(const Matrix& left, const Matrix& right, double tolerance = 1e-9);
double sum_elements(const Matrix& matrix);

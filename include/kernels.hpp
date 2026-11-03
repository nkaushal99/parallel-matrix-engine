#pragma once

#include "matrix.hpp"

Matrix multiply_naive(const Matrix& a, const Matrix& b);
Matrix multiply_reordered(const Matrix& a, const Matrix& b);
Matrix multiply_blocked(const Matrix& a, const Matrix& b, std::size_t block_size);

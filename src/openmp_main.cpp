#include "kernels.hpp"
#include "timer.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <omp.h>

int main() {
    try {
        constexpr std::size_t size = 512;
        constexpr std::size_t block_size = 32;
        const Matrix a = make_deterministic_matrix(size, size, 7);
        const Matrix b = make_deterministic_matrix(size, size, 19);

        Timer timer;
        const Matrix c = multiply_openmp_blocked(a, b, block_size);
        const double time_elapsed_ms = timer.elapsed_ms();

        std::cout << std::fixed << std::setprecision(3)
                  << "method=openmp_blocked\n"
                  << "size=" << size << '\n'
                  << "block_size=" << block_size << '\n'
                  << "threads=" << omp_get_max_threads() << '\n'
                  << "time_ms=" << time_elapsed_ms << '\n'
                  << "result_sum=" << sum_elements(c) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

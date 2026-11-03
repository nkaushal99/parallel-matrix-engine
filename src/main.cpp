#include "kernels.hpp"
#include "timer.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        std::size_t size = 512;

        const Matrix a = make_deterministic_matrix(size, size, 7);
        const Matrix b = make_deterministic_matrix(size, size, 19);

        Timer timer;
        const Matrix c = multiply_naive(a, b);
        double time_elapsed_ms = timer.elapsed_ms();

        std::cout << std::fixed << std::setprecision(3)
                  << "method=naive\n"
                  << "size=" << size << '\n'
                  << "time_ms=" << time_elapsed_ms << '\n'
                  << "result_sum=" << sum_elements(c) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

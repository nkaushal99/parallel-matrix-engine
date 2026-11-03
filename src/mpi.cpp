#include "kernels.hpp"

#include <mpi.h>

#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int rank_count = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    int exit_code = 0;
    try {
        constexpr std::size_t size = 512;
        constexpr std::size_t block_size = 32;
        const int size_as_int = static_cast<int>(size);

        std::vector<int> row_counts(rank_count);
        std::vector<int> row_offsets(rank_count);
        std::vector<int> element_counts(rank_count);
        std::vector<int> element_offsets(rank_count);

        for (int process = 0; process < rank_count; ++process) {
            row_counts[process] = size_as_int / rank_count + (process < size_as_int % rank_count ? 1 : 0);
            row_offsets[process] = process == 0 ? 0 : row_offsets[process - 1] + row_counts[process - 1];
            element_counts[process] = row_counts[process] * size_as_int;
            element_offsets[process] = row_offsets[process] * size_as_int;
        }

        Matrix a;
        Matrix b(size, size);
        Matrix global_c;
        if (rank == 0) {
            a = make_deterministic_matrix(size, size, 7);
            b = make_deterministic_matrix(size, size, 19);
            global_c = Matrix(size, size);
        }

        Matrix local_a(static_cast<std::size_t>(row_counts[rank]), size);
        MPI_Barrier(MPI_COMM_WORLD);
        const double start_time = MPI_Wtime();

        MPI_Scatterv(rank == 0 ? a.data() : nullptr, element_counts.data(), element_offsets.data(), MPI_DOUBLE,
                     local_a.data(), element_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(b.data(), size_as_int * size_as_int, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        const Matrix local_c = multiply_blocked(local_a, b, block_size);
        const double local_sum = sum_elements(local_c);

        MPI_Gatherv(local_c.data(), element_counts[rank], MPI_DOUBLE,
                    rank == 0 ? global_c.data() : nullptr, element_counts.data(), element_offsets.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        const double local_elapsed_seconds = MPI_Wtime() - start_time;
        double max_elapsed_seconds = 0.0;
        MPI_Reduce(&local_elapsed_seconds, &max_elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(3)
                      << "method=mpi_blocked\n"
                      << "size=" << size << '\n'
                      << "block_size=" << block_size << '\n'
                      << "ranks=" << rank_count << '\n'
                      << "time_ms=" << max_elapsed_seconds * 1000.0 << '\n'
                      << "result_sum=" << global_sum << '\n';
        }
    } catch (const std::exception& error) {
        if (rank == 0) {
            std::cerr << "error: " << error.what() << '\n';
        }
        exit_code = 1;
    }

    if (exit_code != 0) {
        MPI_Abort(MPI_COMM_WORLD, exit_code);
    }
    MPI_Finalize();
    return exit_code;
}

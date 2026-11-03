cmake_minimum_required(VERSION 3.20)

project(parallel_matrix_engine LANGUAGES CXX)

# Directly force Release mode for all builds
set(CMAKE_BUILD_TYPE "Release" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(matrix_core src/matrix.cpp src/naive.cpp src/reordered.cpp src/blocked.cpp)
target_include_directories(matrix_core PUBLIC include)

add_executable(matrix_naive src/main.cpp)
target_link_libraries(matrix_naive PRIVATE matrix_core)

add_executable(matrix_reordered src/reordered_main.cpp)
target_link_libraries(matrix_reordered PRIVATE matrix_core)

add_executable(matrix_blocked src/blocked_main.cpp)
target_link_libraries(matrix_blocked PRIVATE matrix_core)

find_package(OpenMP REQUIRED)

add_executable(matrix_openmp src/openmp_main.cpp src/openmp.cpp)
target_link_libraries(matrix_openmp PRIVATE matrix_core OpenMP::OpenMP_CXX)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matrix_mpi src/mpi.cpp)
target_link_libraries(matrix_mpi PRIVATE matrix_core MPI::MPI_CXX)

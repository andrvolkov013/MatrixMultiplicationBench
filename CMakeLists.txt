cmake_minimum_required(VERSION 3.31)
project(MatrixMultiplicationBench LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(MatrixMultiplicationBench main.cu gpu_functions.cu
        matrix_multiplication_bench.cu matrix_lib.cpp)

set(CMAKE_CUDA_ARCHITECTURES 86)

set_target_properties(MatrixMultiplicationBench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    target_link_libraries(MatrixMultiplicationBench PUBLIC OpenMP::OpenMP_CXX)
endif()

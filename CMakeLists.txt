cmake_minimum_required(VERSION 3.17)
project(TSP CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(TSP main.cu)

set_target_properties(
        TSP
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
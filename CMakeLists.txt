cmake_minimum_required(VERSION 3.20)
project(matMultGPU CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(matMultGPU main.cu)



set_target_properties(matMultGPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "-Wall")
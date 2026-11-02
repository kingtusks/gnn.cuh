cmake_minimum_required(VERSION 3.24)
project(gnn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(gnn
    src/gmat.cu
    src/gmat.cuh
    src/gnn.cu
    src/gnn.cuh
    src/main.cu
)

set_target_properties(gnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(gnn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler -Wall>
)

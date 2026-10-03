cmake_minimum_required(VERSION 3.24)
project(gnn LANGUAGES CXX CUDA)

option(GNN_HIP "AMD GPU instead of an NVIDIA one", OFF)

if(GNN_HIP)
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
else()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gnn gnn.cu)

if(GNN_HIP)
  set_source_files_properties(gnn.cu PROPERTIES LANGUAGE HIP)
  target_compile_definitions(gnn PRIVATE GMAT_HIP)
endif()

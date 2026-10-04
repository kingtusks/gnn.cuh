cmake_minimum_required(VERSION 3.24)
project(gnn LANGUAGES CXX)

option(GNN_HIP "Build with HIP/ROCm for AMD instead of CUDA (Linux only)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(GNN_HIP)
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
else()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
  set(CMAKE_HIP_FLAGS_RELEASE "-O3")
endif()

add_executable(gnn gnn.cu)

if(GNN_HIP)
  set_source_files_properties(gnn.cu PROPERTIES LANGUAGE HIP)
  target_compile_definitions(gnn PRIVATE GMAT_HIP)
  target_compile_options(gnn PRIVATE -Wall -Wextra)
elseif(NOT MSVC)
  target_compile_options(gnn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()

if(UNIX)
  target_link_libraries(gnn PRIVATE m)
endif()

#pragma once

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#ifndef GNN_ASSERT
#include <cassert>
#define GNN_ASSERT assert
#endif //GNN_ASSERT

namespace gnn {
    void cuda_check(cudaError_t err, const char* file, int line);
    #define CUDA_CHECK(call) ::gnn::cuda_check(call, __FILE__, __LINE__)
    #define TENSOR_PRINT(t) (t).tensor_print(#t)

    inline constexpr unsigned int tile {32};
    inline constexpr uint64_t tensor_rand_seed {0x16122008ULL};

    struct Tensor {
        unsigned int rows, cols, stride;
        unsigned int depth, dstride;
        float *es;

        Tensor() = default;
        Tensor(size_t rows, size_t cols, size_t depth);
        Tensor(size_t rows, size_t cols, size_t depth, size_t stride, size_t dstride, const float *h_es);

        // lil suggestion to maybe make namespaces for 2d and 3d+ tensors (later tho)
        void tensor_fill(float n);
        void tensor_print(const char *name) const;

        __host__ __device__ float& at(size_t i, size_t j, size_t k) const
        {
            return es[k * dstride + i * stride + j];
        }

        //private:
    };
}

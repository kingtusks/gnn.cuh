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
    #define TENSOR_PRINT(t) (t).print(#t)

    inline constexpr unsigned int tile {32};
    inline constexpr uint64_t tensor_rand_seed {0x16122008ULL};

    struct Tensor {
        unsigned int rows, cols, stride;
        unsigned int depth, dstride;
        float *es;

        Tensor() = default;
        Tensor(size_t rows, size_t cols, size_t depth);
        Tensor(size_t rows, size_t cols, size_t depth, size_t stride, size_t dstride, const float *h_es);

        void pad(Tensor dst, float p_num, size_t p_size);
        void fill(float n);
        void print(const char *name) const;

        __host__ __device__ float& at(size_t i, size_t j, size_t k) const
        {
            return es[k * dstride + i * stride + j];
        }

        //private:
    };
}

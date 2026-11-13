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
    static std::uint64_t tensor_rand_seed {0x16122008ULL};

    void cuda_check(cudaError_t err, const char* file, int line);
    void set_seed(std::uint64_t seed);

    __device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t x)
    {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return (x ^ (x >> 31));
    }

    __device__ __forceinline__ float rand_float(std::uint64_t seed, std::uint64_t i)
    {
        uint64_t r {splitmix64(splitmix64(seed) ^ i)};
        return (float) ((r >> 40) * (1.f / 16777216.f));
    }

    #define CUDA_CHECK(call) ::gnn::cuda_check(call, __FILE__, __LINE__)
    #define TENSOR_PRINT(t) (t).print(#t)

    struct Tensor {
        unsigned int rows_, cols_, stride_;
        unsigned int depth_, dstride_;
        float *es_;

        Tensor() = default;
        Tensor(size_t rows, size_t cols, size_t depth);
        Tensor(size_t rows, size_t cols, size_t depth, size_t stride, size_t dstride, const float *h_es);

        void pad(Tensor dst, float p_num, size_t p_size);
        void fill(float n);
        void rand(float low, float high);
        Tensor mat(size_t depth);
        Tensor mats(size_t depth, size_t n);
        Tensor row(size_t depth, size_t row);
        Tensor rows(size_t depth, size_t row, size_t n);
        void print(const char *name) const;

        __host__ __device__ float& at(size_t i, size_t j, size_t k) const
        {
            return es_[k * dstride_ + i * stride_ + j];
        }

        //private:
    };
}

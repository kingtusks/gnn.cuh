#pragma once

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <initializer_list>
#ifndef GNN_ASSERT
#include <cassert>
#define GNN_ASSERT assert
#endif //GNN_ASSERT

namespace gnn {
    static std::uint64_t tensor_rand_seed {0x16122008ULL};
    const static size_t max_dims {5}; // 5 for now cus uh yea

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
        unsigned int dims_, count_;
        unsigned int shape_[max_dims];
        unsigned int strides_[max_dims];
        float *es_;

        Tensor() = default;
        // cols, rows, depth, etc (col is first for the stride shit)
        Tensor(std::initializer_list<unsigned int> dims);

        void print(const char* name) const;

        __host__ __device__ float& at(const size_t* c) const
        {
            size_t len {sizeof(c) / sizeof(c[0])};
            GNN_ASSERT(len == dims_);

            size_t s {c[0]};
            for (size_t i {1}; i++ < dims_)
                s += c[i] * stride_[i - 1];

            return es_[s];
        }

        //private:
}

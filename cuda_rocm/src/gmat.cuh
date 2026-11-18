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
    inline std::uint64_t tensor_rand_seed {0x16122008ULL};
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
        std::uint64_t r {splitmix64(splitmix64(seed) ^ i)};
        return (float) ((r >> 40) * (1.f / 16777216.f));
    }

    #define CUDA_CHECK(call) ::gnn::cuda_check(call, __FILE__, __LINE__)
    #define TENSOR_PRINT(t) (t).print(#t)

    struct Tensor {
        bool owns_ {true};
        unsigned int dims_ {}, count_ {};
        unsigned int shape_[max_dims] {};
        unsigned int strides_[max_dims] {};
        float *es_ {};

        Tensor(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(const Tensor& other);
        Tensor& operator=(Tensor&& other) noexcept;
        ~Tensor();

        // cols, rows, depth, etc (col is first for the stride shit)
        Tensor() = default;
        Tensor(std::initializer_list<unsigned int> dims);
        Tensor(std::initializer_list<unsigned int> dims, std::initializer_list<float> data);
        Tensor(std::initializer_list<unsigned int> dims, const float* data);

        // Tensor& fill(float n);
        // Tensor& rand(float low = {0.f}, float high = {1.f});
        void print(const char* name) const;

        template <size_t N>
        __device__ float& at(const size_t (&c)[N]) const
        {
            GNN_ASSERT(N == dims_);

            size_t s {};
            for (size_t i {}; i < N; ++i)
                s += c[i] * strides_[i];

            return es_[s];
        }

        //private:
    };
} //namespace gnn

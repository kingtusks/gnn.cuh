#pragma once

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <initializer_list>
#include <vector>
#ifndef GNN_ASSERT
#include <cassert>
#define GNN_ASSERT assert
#endif //GNN_ASSERT

namespace gnn
{
    inline std::uint64_t tensor_rand_seed {0x040507ULL};
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
    #define TENSOR_INFO(t) (t).info(#t)

    // used bc we cant transfer shape and strides directly from the tensor object to the kernel
    struct KernelTensorInfo {
        unsigned int dims_;
        unsigned int shape_[max_dims];
        unsigned int strides_[max_dims];
    };

    struct Tensor
    {
        bool owns_ {true};
        unsigned int dims_ {};
        size_t count_ {};
        unsigned int shape_[max_dims] {};
        unsigned int strides_[max_dims] {};
        float *es_ {};

        //cc, mc, ca, ma, d
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

        Tensor& copy_from(const Tensor& src);
        Tensor slice(unsigned int dim, unsigned int start, unsigned int len) const;
        bool is_contiguous() const;
        void same_structure(const Tensor& other) const;
        Tensor& fill(float n);
        Tensor& rand(float low = {0.0f}, float high = {1.0f});
        Tensor& pad(const Tensor& src, float p_num = {0.0f}, unsigned int p_size = {1});
        Tensor& sgd(const Tensor& g, float rate);
        Tensor& momentum(const Tensor& g, Tensor& v, float rate);
        Tensor& adam(const Tensor& g, Tensor& m1, Tensor& m2, size_t t, float b1 = {0.9f}, float b2 = {0.999f}, float rate = {1e-3f}, float eps = {1e-8f});
        Tensor& max_pool(const Tensor& src, size_t fx, size_t fy, size_t sx, size_t sy);
        Tensor& avg_pool(const Tensor& src, size_t fx, size_t fy, size_t sx, size_t sy);
        Tensor& sum_bias(const Tensor& other);
        Tensor& sum_collapse(const Tensor& other);
        Tensor& sum(const Tensor& other);
        Tensor& operator+=(const Tensor& other) { return sum(other); };
        Tensor operator+(const Tensor& other) const;
        Tensor& dot(const Tensor& a, const Tensor& b, bool aT = {false}, bool bT = {false}, bool naive = {false});
        Tensor& sig();
        Tensor& tanh();
        Tensor& relu();
        Tensor& leaky_relu(float slope = {1e-2f});
        Tensor& elu(float alpha = {1.0f});
        Tensor& softmax();
        KernelTensorInfo kernel_info() const;
        void info(const char* name) const;
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
    }; //struct Tensor
} //namespace gnn

#pragma once

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

//temporary
#define MAT_HOST_DEVICE

#ifndef GNN_ASSERT
#include <cassert>
#define GNN_ASSERT assert
#endif //GNN_ASSERT

namespace Matrix {
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m)
#define CUDA_CHECK(call) cuda_check(call, __FILE__, __LINE__)
    inline constexpr unsigned int TILE{32};
    inline constexpr uint64_t MAT_RAND_SEED{16122008};
    struct Mat {
        unsigned int rows, cols, stride;
        float *es;
    };

    MAT_HOST_DEVICE void cuda_check(cudaError_t err, const char* file, int line);

    Mat mat_alloc(size_t rows, size_t cols);
    Mat mat_alloc_from(size_t rows, size_t cols, size_t stride, const float* h_es);
    Mat mat_alloc_feature_map(size_t m_sz, size_t k_sz, size_t p, size_t c_stride);
    Mat mat_alloc_pad(Mat m, size_t p_size);

    __global__ void mat_fill_contiguous_kernel(float* p, float n, size_t area);
    void mat_fill_contiguous(Mat m, float n);
    __global__ void mat_fill_noncontiguous_kernel(Mat m, float n);
    void mat_fill_noncontiguous(Mat m, float n);
    void mat_fill(Mat m, float n);

    __global__ void mat_pad_fill(Mat dst, float p_num, size_t p_size);
    __global__ void mat_pad_inject(Mat dst, Mat m, size_t p_size);
    void mat_pad(Mat dst, Mat m, float p_num, size_t p_size);

    void mat_set_seed(uint64_t seed);
    __global__ void mat_rand_contiguous_kernel(float* p, float low, float high, size_t area, uint64_t seed);
    void mat_rand_contiguous(Mat m, float low, float high, uint64_t seed);
    __global__ void mat_rand_noncontiguous_kernel(Mat m, float low, float high, uint64_t seed);
    void mat_rand_noncontiguous(Mat m, float low, float high, uint64_t seed);
    void mat_rand(Mat m, float low, float high);

    MAT_HOST_DEVICE Mat mat_row(Mat m, size_t row);
    MAT_HOST_DEVICE Mat mat_rows(Mat m, size_t row, size_t n);

    __global__ void mat_copy_contiguous_kernel(float* dst, const float* src, size_t area);
    MAT_HOST_DEVICE void mat_copy_contiguous(Mat dst, Mat m);
    __global__ void mat_copy_noncontiguous_kernel(Mat dst, Mat m);
    MAT_HOST_DEVICE void mat_copy_noncontiguous(Mat dst, Mat m);
    MAT_HOST_DEVICE void mat_copy(Mat dst, Mat m);

    __global__ void mat_sgd_helper_kernel(Mat dst, Mat m, float s);
    MAT_HOST_DEVICE void mat_sgd_helper(Mat dst, Mat m, float s);
    __global__ void mat_momentum_helper_kernel(Mat dst, Mat m, Mat v, float rate);
    MAT_HOST_DEVICE void mat_momentum_helper(Mat dst, Mat m, Mat v, float rate);
    __global__ void mat_adam_helper_kernel(Mat dst, Mat m, Mat m1, Mat m2, float c1, float c2, size_t t, float rate);
    MAT_HOST_DEVICE void mat_adam_helper(Mat dst, Mat m, Mat m1, Mat m2, size_t t, float rate);

    __global__ void mat_pool_max_kernel(Mat dst, Mat m, size_t fx, size_t fy, size_t sx, size_t sy);
    MAT_HOST_DEVICE void mat_pool_max(Mat dst, Mat m, const size_t* sf);
    __global__ void mat_pool_average_kernel(Mat dst, Mat m, size_t fx, size_t fy, size_t sx, size_t sy);
    MAT_HOST_DEVICE void mat_pool_average(Mat dst, Mat m, const size_t* sf);

    __global__ void mat_sum_bias_kernel(Mat dst, Mat m);
    MAT_HOST_DEVICE void mat_sum_bias(Mat dst, Mat m);
    __global__ void mat_sum_collapse_kernel(Mat dst, Mat m);
    MAT_HOST_DEVICE void mat_sum_collapse(Mat dst, Mat m);

    __global__ void mat_sum_contiguous_kernel(float* dst, const float* src, size_t area);
    MAT_HOST_DEVICE void mat_sum_contiguous(Mat dst, Mat m);
    __global__ void mat_sum_noncontiguous_kernel(Mat dst, Mat m);
    MAT_HOST_DEVICE void mat_sum_noncontiguous(Mat dst, Mat m);
    MAT_HOST_DEVICE void mat_sum(Mat dst, Mat m);

    __global__ void mat_dot_ta_kernel(Mat dst, Mat a, Mat b);
    MAT_HOST_DEVICE void mat_dot_ta(Mat dst, Mat a, Mat b);
    __global__ void mat_dot_tb_kernel(Mat dst, Mat a, Mat b);
    MAT_HOST_DEVICE void mat_dot_tb(Mat dst, Mat a, Mat b);

    __global__ void mat_dot_kernel(Mat dst, Mat a, Mat b);
    MAT_HOST_DEVICE void mat_dot(Mat dst, Mat a, Mat b);

    __global__ void mat_sig_contiguous_kernel(float* p, size_t area);
    MAT_HOST_DEVICE void mat_sig_contiguous(Mat m);
    __global__ void mat_sig_noncontiguous_kernel(Mat m);
    MAT_HOST_DEVICE void mat_sig_noncontiguous(Mat m);
    MAT_HOST_DEVICE void mat_sig(Mat m);

    __global__ void mat_tanh_contiguous_kernel(float* p, size_t area);
    void mat_tanh_contiguous(Mat m);
    __global__ void mat_tanh_noncontiguous_kernel(Mat m);
    void mat_tanh_noncontiguous(Mat m);
    void mat_tanh(Mat m);

    __global__ void mat_relu_contiguous_kernel(float* p, size_t area);
    void mat_relu_contiguous(Mat m);
    __global__ void mat_relu_noncontiguous_kernel(Mat m);
    void mat_relu_noncontiguous(Mat m);
    void mat_relu(Mat m);

    __global__ void mat_leaky_relu_contiguous_kernel(float* p, size_t area);
    void mat_leaky_relu_contiguous(Mat m);
    __global__ void mat_leaky_relu_noncontiguous_kernel(Mat m);
    void mat_leaky_relu_noncontiguous(Mat m);
    void mat_leaky_relu(Mat m);

    __global__ void mat_elu_contiguous_kernel(float* p, size_t area);
    void mat_elu_contiguous(Mat m);
    __global__ void mat_elu_noncontiguous_kernel(Mat m);
    void mat_elu_noncontiguous(Mat m);
    void mat_elu(Mat m);

    __global__ void mat_softmax_kernel(Mat m);
    MAT_HOST_DEVICE void mat_softmax(Mat m);

    void mat_print(Mat m, const char *name);

    static __device__ __forceinline__ float sigmoidf_d(float x) {
        return 1.f / (1.f + expf(-x));
    }

    static __device__ __forceinline__ float tanhf_d(float x) {
        return tanhf(x);
    }

    static __device__ __forceinline__ float reluf_d(float x) {
        return fmaxf(0.f, x);
    }

    static __device__ __forceinline__ float leaky_reluf_d(float x) {
        return fmaxf(.01f * x, x);
    }

    static __device__ __forceinline__ float eluf_d(float x) {
        return fmaxf((expf(x) - 1.0f), x);
    }

    static __device__ __forceinline__ void softmaxf_d(Mat m, size_t i) {
        float x{MAT_AT(m, i, 0)};
        for (size_t j{1}; j < m.cols; ++j)
            x = fmaxf(x, MAT_AT(m, i, j));

        float s{}, e{};
        for (size_t j{}; j < m.cols; ++j) {
            e = expf(MAT_AT(m, i, j) - x);
            MAT_AT(m, i, j) = e;
            s += e;
        }

        float inv_s{1.f / s};
        for (size_t j{}; j < m.cols; ++j) MAT_AT(m, i, j) *= inv_s;
    }

    __device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return (x ^ (x >> 31));
    }

    __device__ __forceinline__ float rand_float(uint64_t seed, uint64_t i) {
        uint64_t r{splitmix64(splitmix64(seed) ^ i)};
        return (float) ((r >> 40) * (1.f / 16777216.f));
    }
}

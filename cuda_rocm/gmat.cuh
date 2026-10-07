#pragma once

//maybe idk
#ifdef GMAT_HIP
#include <hip/hip_runtime.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemset hipMemset
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define __trap() __builtin_trap()
#define MAT_HOST_DEVICE
#else
#include <cuda_runtime.h>
#define MAT_HOST_DEVICE
#endif

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdint.h>

#ifndef GNN_ASSERT
#include <assert.h>
#define GNN_ASSERT assert
#endif //GNN_ASSERT

#ifndef MAT_RAND_SEED
#define MAT_RAND_SEED 16122008
#endif //MAT_RAND_SEED

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

//TILE * TILE <= 1024 (32 max)
#define TILE 32
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]
#define VEC_AT(v, i, j) (v)[(i)*(m).stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m)

MAT_HOST_DEVICE void cuda_check(cudaError_t err, const char* file, int line);
#define CUDA_CHECK(call) cuda_check(call, __FILE__, __LINE__)

Mat mat_alloc(size_t rows, size_t cols);
Mat mat_alloc_from(size_t rows, size_t cols, size_t stride, const float* h_es);

__global__ void mat_fill_contiguous_kernel(float* p, float n, size_t area);
void mat_fill_contiguous(Mat m, float n);
__global__ void mat_fill_noncontiguous_kernel(Mat m, float n);
void mat_fill_noncontiguous(Mat m, float n);
void mat_fill(Mat m, float n);

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

__global__ void mat_sub_scaled_kernel(Mat dst, Mat m, float s);
MAT_HOST_DEVICE void mat_sub_scaled(Mat dst, Mat m, float s);

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

static __device__ __forceinline__ void softmaxf_d(Mat m, size_t i) {
    float x = MAT_AT(m, i, 0);
    for (size_t j = 1; j < m.cols; ++j)
        x = fmaxf(x, MAT_AT(m, i, j));

    float s = 0;
    for (size_t j = 0; j < m.cols; ++j) {
        float e = expf(MAT_AT(m, i, j) - x);
        MAT_AT(m, i, j) = e;
        s += e;
    }

    float inv_s = 1.f / s;
    for (size_t j = 0; j < m.cols; ++j) MAT_AT(m, i, j) *= inv_s;
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return (x ^ (x >> 31));
}

__device__ __forceinline__ float rand_float(uint64_t seed, uint64_t i) {
    uint64_t r = splitmix64(splitmix64(seed) ^ i);
    return (float) ((r >> 40) * (1.f / 16777216.f));
}

#ifdef GMAT_IMPLEMENTATION

//CUDA CHECK

MAT_HOST_DEVICE void cuda_check(cudaError_t err, const char* file, int line) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (err != cudaSuccess) {
        printf("%s:%d CUDA Error: %s\n", file, line, cudaGetErrorString(err));
        __trap();
    }
#else
    if (err != cudaSuccess) {
        printf("%s:%d CUDA Error: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif
}

//GENERAL PURPOSE

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    CUDA_CHECK(cudaMalloc((void**)&m.es, rows * cols * sizeof(float)));
    return m;
}

Mat mat_alloc_from(size_t rows, size_t cols, size_t stride, const float* h_es) {
    Mat m = mat_alloc(rows, cols);
    CUDA_CHECK(cudaMemcpy2D(
                m.es, m.stride * sizeof(float),
                h_es, stride * sizeof(float),
                cols * sizeof(float),
                rows, cudaMemcpyHostToDevice));
    return m;
}

__global__ void mat_fill_contiguous_kernel(float* p, float n, size_t area) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride)
        p[i] = n;
}

void mat_fill_contiguous(Mat m, float n) {
    unsigned int threads = 256;
    size_t area = (size_t) m.rows * m.cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    mat_fill_contiguous_kernel<<<blocks, threads>>>(m.es, n, area);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_fill_noncontiguous_kernel(Mat m, float n) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = n;
}

void mat_fill_noncontiguous(Mat m, float n) {
    dim3 threads(32, 8);
    dim3 blocks((m.cols + threads.x - 1) / threads.x, (m.rows + threads.y - 1) / threads.y);

    mat_fill_noncontiguous_kernel<<<blocks, threads>>>(m, n);
    CUDA_CHECK(cudaGetLastError());
}

void mat_fill(Mat m, float n) {
    (m.stride == m.cols) ? mat_fill_contiguous(m, n) : mat_fill_noncontiguous(m, n);
}

static uint64_t mat_rand_seed = MAT_RAND_SEED;

void mat_set_seed(uint64_t seed) {
    mat_rand_seed = seed;
}

__global__ void mat_rand_contiguous_kernel(float* p, float low, float high, size_t area, uint64_t seed) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride)
        p[i] = rand_float(seed, i) * (high - low) + low;
}

void mat_rand_contiguous(Mat m, float low, float high, uint64_t seed) {
    unsigned int threads = 256;
    size_t area = (size_t) m.rows * m.cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    mat_rand_contiguous_kernel<<<blocks, threads>>>(m.es, low, high, area, seed);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_rand_noncontiguous_kernel(Mat m, float low, float high, uint64_t seed) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m.rows && j < m.cols) {
        size_t true_i = i * m.cols + j;
        MAT_AT(m, i, j) = rand_float(seed, true_i) * (high - low) + low;
    }
}

void mat_rand_noncontiguous(Mat m, float low, float high, uint64_t seed) {
    dim3 threads(32, 8);
    dim3 blocks((m.cols + threads.x - 1) / threads.x, (m.rows + threads.y - 1) / threads.y);

    mat_rand_noncontiguous_kernel<<<blocks, threads>>>(m, low, high, seed);
    CUDA_CHECK(cudaGetLastError());
}

void mat_rand(Mat m, float low, float high) {
    uint64_t seed = mat_rand_seed++;
    //uint64_t seed = mat_rand_seed;
    (m.stride == m.cols) ? mat_rand_contiguous(m, low, high, seed) : mat_rand_noncontiguous(m, low, high, seed);
}

MAT_HOST_DEVICE Mat mat_row(Mat m, size_t row) {
    Mat r;
    r.rows = 1;
    r.cols = m.cols;
    r.stride = m.stride;
    r.es = &MAT_AT(m, row, 0);
    return r;
}

MAT_HOST_DEVICE Mat mat_rows(Mat m, size_t row, size_t n) {
    Mat r;
    r.rows = n;
    r.cols = m.cols;
    r.stride = m.stride;
    r.es = &MAT_AT(m, row, 0);
    return r;
}

__global__ void mat_copy_contiguous_kernel(float* dst, const float* src, size_t area) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride) dst[i] = src[i];
}

MAT_HOST_DEVICE void mat_copy_contiguous(Mat dst, Mat m) {
    unsigned int threads = 256;
    size_t area = (size_t) m.rows * m.cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    mat_copy_contiguous_kernel<<<blocks, threads>>>(dst.es, m.es, area);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_copy_noncontiguous_kernel(Mat dst, Mat m) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m.rows && j < m.cols) MAT_AT(dst, i, j) = MAT_AT(m, i, j);
}

MAT_HOST_DEVICE void mat_copy_noncontiguous(Mat dst, Mat m) {
    dim3 threads(32, 8);
    dim3 blocks((m.cols + threads.x - 1) / threads.x, (m.rows + threads.y - 1) / threads.y);
    mat_copy_noncontiguous_kernel<<<blocks, threads>>>(dst, m);
    CUDA_CHECK(cudaGetLastError());
}

MAT_HOST_DEVICE void mat_copy(Mat dst, Mat m) {
    (m.stride == m.cols && dst.stride == dst.cols) ? mat_copy_contiguous(dst, m) : mat_copy_noncontiguous(dst, m);
}

//MATRIX OPS

__global__ void mat_sub_scaled_kernel(Mat dst, Mat m, float s) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m.rows && j < m.cols)
        MAT_AT(dst, i, j) -= s * MAT_AT(m, i, j);
}

MAT_HOST_DEVICE void mat_sub_scaled(Mat dst, Mat m, float s) {
    GNN_ASSERT(dst.rows == m.rows);
    GNN_ASSERT(dst.cols == m.cols);

    dim3 threads(32, 8);
    dim3 blocks((m.cols + threads.x - 1) / threads.x, (m.rows + threads.y - 1) / threads.y);
    mat_sub_scaled_kernel<<<blocks, threads>>>(dst, m, s);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_sum_bias_kernel(Mat dst, Mat m) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dst.rows && j < dst.cols)
        MAT_AT(dst, i, j) += MAT_AT(m, 0, j);
}

MAT_HOST_DEVICE void mat_sum_bias(Mat dst, Mat m) {
    GNN_ASSERT(m.rows == 1);
    GNN_ASSERT(m.cols == dst.cols);

    dim3 threads(32, 8);
    dim3 blocks(
        (unsigned int) ((dst.cols + threads.x - 1) / threads.x),
        (unsigned int) ((dst.rows + threads.y - 1) / threads.y)
    );
    mat_sum_bias_kernel<<<blocks, threads>>>(dst, m);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_sum_collapse_kernel(Mat dst, Mat m) {
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m.cols) return;

    float s = 0;
    for (size_t i = 0; i < m.rows; ++i)
        s += MAT_AT(m, i, j);
    MAT_AT(dst, 0, j) = s;
}

MAT_HOST_DEVICE void mat_sum_collapse(Mat dst, Mat m) {
    GNN_ASSERT(dst.rows == 1);
    GNN_ASSERT(dst.cols == m.cols);

    unsigned int threads = 256;
    unsigned int blocks = (unsigned int) ((m.cols + threads - 1) / threads);
    mat_sum_collapse_kernel<<<blocks, threads>>>(dst, m);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_sum_contiguous_kernel(float* dst, const float* src, size_t area) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride) dst[i] += src[i];
}

void mat_sum_contiguous(Mat dst, Mat m) {
    unsigned int threads = 256;
    size_t area = (size_t) m.rows * m.cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    mat_sum_contiguous_kernel<<<blocks, threads>>>(dst.es, m.es, area);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_sum_noncontiguous_kernel(Mat dst, Mat m) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m.rows && j < m.cols) MAT_AT(dst, i, j) += MAT_AT(m, i, j);
}

void mat_sum_noncontiguous(Mat dst, Mat m) {
    dim3 threads(32, 8);
    dim3 blocks((m.cols + threads.x - 1) / threads.x, (m.rows + threads.y - 1) / threads.y);
    mat_sum_noncontiguous_kernel<<<blocks, threads>>>(dst, m);
    CUDA_CHECK(cudaGetLastError());
}

void mat_sum(Mat dst, Mat m) {
    GNN_ASSERT(dst.rows == m.rows);
    GNN_ASSERT(dst.cols == m.cols);

    (m.stride == m.cols && dst.stride == dst.cols) ? mat_sum_contiguous(dst, m) : mat_sum_noncontiguous(dst, m);
}

__global__ void mat_dot_ta_kernel(Mat dst, Mat a, Mat b) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    size_t i = (size_t) blockIdx.y * TILE + threadIdx.y;
    size_t j = (size_t) blockIdx.x * TILE + threadIdx.x;
    float res = 0;

    for (size_t t = 0; t < a.rows; t += TILE) {
        size_t ka = t + threadIdx.x;
        size_t kb = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (i < a.cols && ka < a.rows) ? MAT_AT(a, ka, i) : 0;
        Bs[threadIdx.y][threadIdx.x] = (kb < b.rows && j < b.cols) ? MAT_AT(b, kb, j) : 0;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            res += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (i < dst.rows && j < dst.cols)
        MAT_AT(dst, i, j) = res;
}

MAT_HOST_DEVICE void mat_dot_ta(Mat dst, Mat a, Mat b) {
    GNN_ASSERT(a.rows == b.rows);
    GNN_ASSERT(dst.rows == a.cols);
    GNN_ASSERT(dst.cols == b.cols);

    dim3 threads(TILE, TILE);
    dim3 blocks((unsigned int) ((b.cols + TILE - 1) / TILE),
                (unsigned int) ((a.cols + TILE - 1) / TILE));
    mat_dot_ta_kernel<<<blocks, threads>>>(dst, a, b);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_dot_tb_kernel(Mat dst, Mat a, Mat b) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    size_t i = (size_t) blockIdx.y * TILE + threadIdx.y;
    size_t j = (size_t) blockIdx.x * TILE + threadIdx.x;
    float res = 0;

    for (size_t t = 0; t < a.cols; t += TILE) {
        size_t ka = t + threadIdx.x;
        size_t kb = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (i < a.rows && ka < a.cols) ? MAT_AT(a, i, ka) : 0;
        Bs[threadIdx.y][threadIdx.x] = (j < b.rows && kb < b.cols) ? MAT_AT(b, j, kb) : 0;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            res += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (i < dst.rows && j < dst.cols)
        MAT_AT(dst, i, j) = res;
}

MAT_HOST_DEVICE void mat_dot_tb(Mat dst, Mat a, Mat b) {
    GNN_ASSERT(a.cols == b.cols);
    GNN_ASSERT(dst.rows == a.rows);
    GNN_ASSERT(dst.cols == b.rows);

    dim3 threads(TILE, TILE);
    dim3 blocks((unsigned int) ((b.rows + TILE - 1) / TILE),
                (unsigned int) ((a.rows + TILE - 1) / TILE));
    mat_dot_tb_kernel<<<blocks, threads>>>(dst, a, b);
    CUDA_CHECK(cudaGetLastError());
}

//slow for now while i port and test the rest (around 15%ish of cuBLAS)
__global__ void mat_dot_kernel(Mat dst, Mat a, Mat b) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    size_t i = (size_t) blockIdx.y * TILE + threadIdx.y;
    size_t j = (size_t) blockIdx.x * TILE + threadIdx.x;
    float res = 0;

    for (size_t t = 0; t < a.cols; t += TILE) {
        size_t a_col = t + threadIdx.x;
        size_t b_row = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (i < a.rows && a_col < a.cols) ? a.es[i * a.cols + a_col] : 0;
        Bs[threadIdx.y][threadIdx.x] = (b_row < a.cols && j < b.cols) ? b.es[b_row * b.cols + j] : 0;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            res += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (i < a.rows && j < b.cols) dst.es[i * b.cols + j] = res;
}

void mat_dot(Mat dst, Mat a, Mat b) {
    GNN_ASSERT(a.cols == b.rows);
    GNN_ASSERT(dst.rows == a.rows);
    GNN_ASSERT(dst.cols == b.cols);

    dim3 threads(TILE, TILE);
    dim3 blocks((unsigned int) ((b.cols + TILE - 1) / TILE),
                (unsigned int) ((a.rows + TILE - 1) / TILE));

    mat_dot_kernel<<<blocks, threads>>>(dst, a, b);
    CUDA_CHECK(cudaGetLastError());
}

//ACTIVATIONS

__global__ void mat_sig_contiguous_kernel(float* p, size_t area) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride) p[i] = sigmoidf_d(p[i]);
}

void mat_sig_contiguous(Mat m) {
    unsigned int threads = 256;
    size_t area = (size_t) m.rows * m.cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    mat_sig_contiguous_kernel<<<blocks, threads>>>(m.es, area);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_sig_noncontiguous_kernel(Mat m) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = sigmoidf_d(MAT_AT(m, i, j));
}

void mat_sig_noncontiguous(Mat m) {
    dim3 threads(32, 8);
    dim3 blocks((m.cols + threads.x - 1) / threads.x, (m.rows + threads.y - 1) / threads.y);
    mat_sig_noncontiguous_kernel<<<blocks, threads>>>(m);
    CUDA_CHECK(cudaGetLastError());
}

void mat_sig(Mat m) {
    (m.stride == m.cols) ? mat_sig_contiguous(m) : mat_sig_noncontiguous(m);
}

__global__ void mat_tanh_contiguous_kernel(float* p, size_t area) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride) p[i] = tanhf_d(p[i]);
}

void mat_tanh_contiguous(Mat m) {
    unsigned int threads = 256;
    size_t area = (size_t) m.rows * m.cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    mat_tanh_contiguous_kernel<<<blocks, threads>>>(m.es, area);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_tanh_noncontiguous_kernel(Mat m) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = tanhf_d(MAT_AT(m, i, j));
}

void mat_tanh_noncontiguous(Mat m) {
    dim3 threads(32, 8);
    dim3 blocks((m.cols + threads.x - 1) / threads.x, (m.rows + threads.y - 1) / threads.y);
    mat_tanh_noncontiguous_kernel<<<blocks, threads>>>(m);
    CUDA_CHECK(cudaGetLastError());
}

void mat_tanh(Mat m) {
    (m.stride == m.cols) ? mat_tanh_contiguous(m) : mat_tanh_noncontiguous(m);
}

__global__ void mat_relu_contiguous_kernel(float* p, size_t area) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride) p[i] = reluf_d(p[i]);
}

void mat_relu_contiguous(Mat m) {
    unsigned int threads = 256;
    size_t area = (size_t) m.rows * m.cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    mat_relu_contiguous_kernel<<<blocks, threads>>>(m.es, area);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_relu_noncontiguous_kernel(Mat m) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = reluf_d(MAT_AT(m, i, j));
}

void mat_relu_noncontiguous(Mat m) {
    dim3 threads(32, 8);
    dim3 blocks((m.cols + threads.x - 1) / threads.x, (m.rows + threads.y - 1) / threads.y);
    mat_relu_noncontiguous_kernel<<<blocks, threads>>>(m);
    CUDA_CHECK(cudaGetLastError());
}

void mat_relu(Mat m) {
    (m.stride == m.cols) ? mat_relu_contiguous(m) : mat_relu_noncontiguous(m);
}

__global__ void mat_leaky_relu_contiguous_kernel(float* p, size_t area) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride) p[i] = leaky_reluf_d(p[i]);
}

void mat_leaky_relu_contiguous(Mat m) {
    unsigned int threads = 256;
    size_t area = (size_t) m.rows * m.cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    mat_leaky_relu_contiguous_kernel<<<blocks, threads>>>(m.es, area);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mat_leaky_relu_noncontiguous_kernel(Mat m) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = leaky_reluf_d(MAT_AT(m, i, j));
}

void mat_leaky_relu_noncontiguous(Mat m) {
    dim3 threads(32, 8);
    dim3 blocks((m.cols + threads.x - 1) / threads.x, (m.rows + threads.y - 1) / threads.y);
    mat_leaky_relu_noncontiguous_kernel<<<blocks, threads>>>(m);
    CUDA_CHECK(cudaGetLastError());
}

void mat_leaky_relu(Mat m) {
    (m.stride == m.cols) ? mat_leaky_relu_contiguous(m) : mat_leaky_relu_noncontiguous(m);
}

__global__ void mat_softmax_kernel(Mat m) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m.rows) softmaxf_d(m, i);
}

MAT_HOST_DEVICE void mat_softmax(Mat m) {
    unsigned int threads = 256;
    unsigned int blocks = (unsigned int) ((m.rows + threads - 1) / threads);
    mat_softmax_kernel<<<blocks, threads>>>(m);
    CUDA_CHECK(cudaGetLastError());
}

void mat_print(Mat m, const char *name) {
    float* host_es = (float*)malloc(m.rows * m.stride * sizeof(float));
    CUDA_CHECK(cudaMemcpy(host_es, m.es, m.rows * m.stride * sizeof(float), cudaMemcpyDeviceToHost));

    printf("%s\n", name);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("    %f ", host_es[i*m.stride + j]);
        }
        printf("\n");
    }
    printf("\n");
    free(host_es);
}

#endif //GMAT_IMPLEMENTATION

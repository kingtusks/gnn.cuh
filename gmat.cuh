#pragma once

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef GNN_ASSERT
#include <assert.h>
#define GNN_ASSERT assert
#endif //GNN_ASSERT

#ifndef MAT_RAND_SEED
#define MAT_RAND_SEED 16122008ULL
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

void cuda_check(cudaError_t err, const char* file, int line);
#define CUDA_CHECK(call) cuda_check(call, __FILE__, __LINE__)

Mat mat_alloc(size_t rows, size_t cols);

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

Mat mat_row(Mat m, size_t row);

__global__ void mat_copy_contiguous_kernel(float* dst, const float* src, size_t area);
void mat_copy_contiguous(Mat dst, Mat m);
__global__ void mat_copy_noncontiguous_kernel(Mat dst, Mat m);
void mat_copy_noncontiguous(Mat dst, Mat m);
void mat_copy(Mat dst, Mat m);

__global__ void mat_dot_kernel(Mat dst, Mat a, Mat b);
void mat_dot(Mat dst, Mat a, Mat b);

__global__ void mat_sum_contiguous_kernel(float* dst, const float* src, size_t area);
void mat_sum_contiguous(Mat dst, Mat m);
__global__ void mat_sum_noncontiguous_kernel(Mat dst, Mat m);
void mat_sum_noncontiguous(Mat dst, Mat m);
void mat_sum(Mat dst, Mat m);

__global__ void mat_sig_contiguous_kernel(float* p, size_t area);
void mat_sig_contiguous(Mat m);
__global__ void mat_sig_noncontiguous_kernel(Mat m);
void mat_sig_noncontiguous(Mat m);
void mat_sig(Mat m);

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

void mat_print(Mat m, const char *name);

__device__ __forceinline__ float __sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

__device__ __forceinline__ float __tanhf(float x) {
    return (expf(2*x) - 1) / (expf(2*x) + 1);
}

__device__ __forceinline__ float __reluf(float x) {
    return MAX(0, x);
}

__host__ __device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
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

void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s:%d CUDA Error: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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

__global__ void mat_fill_contiguous_kernel(float* p, float n, size_t area) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride) p[i] = n;
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
    //for testing
#if 0
    uint64_t seed = mat_rand_seed++; //new seed per call
#else
    uint64_t seed = mat_rand_seed;
#endif
    (m.stride == m.cols) ? mat_rand_contiguous(m, low, high, seed) : mat_rand_noncontiguous(m, low, high, seed);
}

Mat mat_row(Mat m, size_t row) {
    return (Mat) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0),
    };
}

__global__ void mat_copy_contiguous_kernel(float* dst, const float* src, size_t area) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride) dst[i] = src[i];
}

void mat_copy_contiguous(Mat dst, Mat m) {
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

void mat_copy_noncontiguous(Mat dst, Mat m) {
    dim3 threads(32, 8);
    dim3 blocks((m.cols + threads.x - 1) / threads.x, (m.rows + threads.y - 1) / threads.y);
    mat_copy_noncontiguous_kernel<<<blocks, threads>>>(dst, m);
    CUDA_CHECK(cudaGetLastError());
}

void mat_copy(Mat dst, Mat m) {
    (m.stride == m.cols && dst.stride == dst.cols) ? mat_copy_contiguous(dst, m) : mat_copy_noncontiguous(dst, m);
}

//MATRIX OPS

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

//ACTIVATIONS

__global__ void mat_sig_contiguous_kernel(float* p, size_t area) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;

    for (; i < area; i += stride) p[i] = __sigmoidf(p[i]);
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

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = __sigmoidf(MAT_AT(m, i, j));
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

    for (; i < area; i += stride) p[i] = __tanhf(p[i]);
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

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = __tanhf(MAT_AT(m, i, j));
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

    for (; i < area; i += stride) p[i] = __reluf(p[i]);
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

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = __reluf(MAT_AT(m, i, j));
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

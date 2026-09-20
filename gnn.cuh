#pragma once

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#ifndef GNN_ASSERT
#include <assert.h>
#define GNN_ASSERT assert
#endif //GNN_ASSERT

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

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

__device__ float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

__device__ float tanhf(float x) {
    return (expf(2*x) - 1) / (expf(2*x) + 1);
}

__device__ float reluf(float x) {
    return MAX(0, x);
}

#ifdef GNN_IMPLEMENTATION

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
    cudaMalloc((void**)&m.es, rows * cols * sizeof(float));
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

__global__ void mat_dot_kernel(Mat dst, Mat a, Mat b) {}

void mat_dot(Mat dst, Mat a, Mat b) {}

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

    for (; i < area; i += stride) p[i] = sigmoidf(p[i]);
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

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
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

    for (; i < area; i += stride) p[i] = tanhf(p[i]);
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

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = tanhf(MAT_AT(m, i, j));
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

    for (; i < area; i += stride) p[i] = reluf(p[i]);
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

    if (i < m.rows && j < m.cols) MAT_AT(m, i, j) = reluf(MAT_AT(m, i, j));
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
    cudaMemcpy(host_es, m.es, m.rows * m.stride * sizeof(float), cudaMemcpyDeviceToHost);

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

#endif //GNN_IMPLEMENTATION

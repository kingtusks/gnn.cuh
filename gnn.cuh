#pragma once

#include <cstdio>
#include <cmath>

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
#define MAT_PRINT(m) mat_print(m, #m)

Mat mat_alloc(size_t rows, size_t cols);

__global__ void mat_fill_kernel(Mat m, float n);
void mat_fill(Mat m, float n);

Mat mat_row(Mat m, size_t row);

__global__ void mat_copy_kernel(Mat dst, Mat m);
void mat_copy(Mat dst, Mat m);

// __global__ void mat_dot_kernel(Mat dst, Mat a, Mat b);
// void mat_dot(Mat dst, Mat a, Mat b);

__global__ void mat_sum_kernel(Mat dst, Mat a);
void mat_sum(Mat dst, Mat a);

__global__ void mat_sig_kernel(Mat m);
void mat_sig(Mat m);

__global__ void mat_tanh_kernel(Mat m);
void mat_tanh(Mat m);

__global__ void mat_relu_kernel(Mat m);
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

//GENERAL PURPOSE

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    cudaMalloc((void**)&m.es, rows * cols * sizeof(float));
    return m;
}

__global__ void mat_fill_kernel(Mat m, float n) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    MAT_AT(m, tx, ty) = n;
}

void mat_fill(Mat m, float n) {
    dim3 dimGrid(1, 1);
    dim3 dimBlock(m.rows, m.cols);
    mat_fill_kernel<<<dimGrid, dimBlock>>>(m, n);
}

Mat mat_row(Mat m, size_t row) {
    return (Mat) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0),
    };
}

__global__ void mat_copy_kernel(Mat dst, Mat m) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    MAT_AT(dst, tx, ty) = MAT_AT(m, tx, ty);
}

void mat_copy(Mat dst, Mat m) {
    dim3 dimGrid(1, 1);
    dim3 dimBlock(m.rows, m.cols);
    mat_copy_kernel<<<dimGrid, dimBlock>>>(dst, m);
}

//MATRIX OPS

//USE WARPTILING (whatever that is)
// __global__ void mat_dot_kernel(Mat dst, Mat a, Mat b) {}
// void mat_dot(Mat dst, Mat a, Mat b) {};

__global__ void mat_sum_kernel(Mat dst, Mat a) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    MAT_AT(dst, tx, ty) += MAT_AT(a, tx, ty);
}

void mat_sum(Mat dst, Mat a) {
    GNN_ASSERT(dst.rows == a.rows);
    GNN_ASSERT(dst.cols == a.cols);

    dim3 dimGrid(1, 1);
    dim3 dimBlock(dst.rows, dst.cols);
    mat_sum_kernel<<<dimGrid, dimBlock>>>(dst, a);
}

//ACTIVATIONS

__global__ void mat_sig_kernel(Mat m) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    MAT_AT(m, tx, ty) = sigmoidf(MAT_AT(m, tx, ty));
}

void mat_sig(Mat m) {
    dim3 dimGrid(1, 1);
    dim3 dimBlock(m.rows, m.cols);
    mat_sig_kernel<<<dimGrid, dimBlock>>>(m);
}

__global__ void mat_tanh_kernel(Mat m) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    MAT_AT(m, tx, ty) = tanhf(MAT_AT(m, tx, ty));
}

void mat_tanh(Mat m) {
    dim3 dimGrid(1, 1);
    dim3 dimBlock(m.rows, m.cols);
    mat_tanh_kernel<<<dimGrid, dimBlock>>>(m);
}

__global__ void mat_relu_kernel(Mat m) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    MAT_AT(m, tx, ty) = reluf(MAT_AT(m, tx, ty));
}

void mat_relu(Mat m) {
    dim3 dimGrid(1, 1);
    dim3 dimBlock(m.rows, m.cols);
    mat_relu_kernel<<<dimGrid, dimBlock>>>(m);
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

#pragma once

#include <cstdio>

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m)

Mat mat_alloc(size_t rows, size_t cols);
__global__ void mat_fill_kernel(Mat m, float n);
void mat_fill(Mat* m, float n);
void mat_row(Mat m, size_t row);
// __global__ void MatRandKernel(Mat m, float low, float high);
// void MatRand(Mat* m, float low, float high);

#ifdef GNN_IMPLEMENTATION

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

void mat_fill(Mat* m, float n) {
    dim3 dimGrid(1, 1);
    dim3 dimBlock(m->rows, m->cols);
    mat_fill_kernel<<<dimGrid, dimBlock>>>(*m, n);
}

// __global__ void MatRandKernel(Mat m, float low, float high) {}
// void MatRand(Mat* m, float low, float high) {}

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

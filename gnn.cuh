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

__global__ void MatFillKernel(Mat m, float n);
void MatFill(Mat* m, float n);
__global__ void MatRandKernel(Mat m, float low, float high);
void MatRand(Mat* m, float low, float high);

#ifdef GNN_IMPLEMENTATION

__global__ void MatFillKernel(Mat m, float n) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    MAT_AT(m, tx, ty) = n;
}

void MatFill(Mat* m, float n) {
    float* d_es;
    int size = m->rows * m->cols * sizeof(float);
    int width = m->stride;

    cudaMalloc((void**)&d_es, size);
    Mat d_mat = *m;
    d_mat.es = d_es;

    dim3 dimGrid(1, 1);
    dim3 dimBlock(width, width);

    MatFillKernel<<<dimGrid, dimBlock>>>(d_mat, n);

    cudaMemcpy(m->es, d_es, size, cudaMemcpyDeviceToHost);
    cudaFree(d_es);
}

__global__ void MatRandKernel(Mat m, float low, float high) {}
void MatRand(Mat* m, float low, float high) {}

void mat_print(Mat m, const char *name) {
    printf("%s\n", name);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("    %f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

#endif //GNN_IMPLEMENTATION

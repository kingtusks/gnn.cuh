#pragma once

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

void MatFill(Mat m, float n);
__global__ void MatFillKernel(Mat m, float n);

#ifdef GNN_IMPLEMENTATION

void MatFill(Mat m, float n) {
    //stub
}

__global__ void MatFillKernel(Mat m, float n) {
    //stub
}

#endif //GNN_IMPLEMENTATION

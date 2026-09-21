#pragma once

#ifdef GNN_IMPLEMENTATION
#define GMAT_IMPLEMENTATION
#endif //GNN_IMPLEMENTATION

#include "gmat.cuh"

#ifndef GNN_ASSERT
#include <assert.h>
#define GNN_ASSERT assert
#endif //GNN_ASSERT

#ifndef GNN_MALLOC
#include <stdlib.h>
#define GNN_MALLOC malloc
#endif //GNN_MALLOC

typedef struct {
    size_t count;
    Mat *w;
    Mat *b;
    Mat *a;
} NN;

#define ARRAY_LEN(arr) sizeof((arr)) / sizeof((arr)[0])
#define NN_INPUT(nn) (nn).a[0]
#define NN_OUTPUT(nn) (nn).a[(nn).count]
#define NN_PRINT(nn) nn_print(nn, #nn)

NN nn_alloc(size_t* dim, size_t dim_len);
void nn_rand(NN nn, float low, float high);
void nn_fill(NN nn, float n);
void nn_print(NN nn, const char* name);

#ifdef GNN_IMPLEMENTATION

NN nn_alloc(size_t* dim, size_t dim_len) {
    NN nn;
    nn.count = dim_len - 1;

    size_t sizeof_wb = sizeof(Mat) * nn.count;
    size_t sizeof_a = sizeof(Mat) * (nn.count + 1);

    Mat* hw = (Mat*) malloc(sizeof_wb);
    Mat* hb = (Mat*) malloc(sizeof_wb);
    Mat* ha = (Mat*) malloc(sizeof_a);
    GNN_ASSERT(hw && hb && ha);

    ha[0] = mat_alloc(1, dim[0]);
    for (size_t i = 1; i < dim_len; ++i) {
        hw[i - 1] = mat_alloc(dim[i - 1], dim[i]);
        hb[i - 1] = mat_alloc(1, dim[i]);
        ha[i] = mat_alloc(1, dim[i]);
    }

    CUDA_CHECK(cudaMalloc((void**)&nn.w, sizeof_wb));
    CUDA_CHECK(cudaMemcpy(nn.w, hw, sizeof_wb, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&nn.b, sizeof_wb));
    CUDA_CHECK(cudaMemcpy(nn.b, hb, sizeof_wb, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&nn.a, sizeof_a));
    CUDA_CHECK(cudaMemcpy(nn.a, ha, sizeof_a, cudaMemcpyHostToDevice));

    free(hw);
    free(hb);
    free(ha);

    return nn;
}

void nn_rand(NN nn, float low, float high) {
    size_t sizeof_wb = sizeof(Mat) * nn.count;

    Mat* hw = (Mat*) malloc(sizeof_wb);
    Mat* hb = (Mat*) malloc(sizeof_wb);
    GNN_ASSERT(hw && hb);

    CUDA_CHECK(cudaMemcpy(hw, nn.w, sizeof_wb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb, nn.b, sizeof_wb, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nn.count; ++i) {
        mat_rand(hw[i], low, high);
        mat_rand(hb[i], low, high);
    }

    free(hw);
    free(hb);
}

void nn_fill(NN nn, float n) {
    size_t sizeof_wb = sizeof(Mat) * nn.count;
    size_t sizeof_a = sizeof(Mat) * (nn.count + 1);

    Mat* hw = (Mat*) malloc(sizeof_wb);
    Mat* hb = (Mat*) malloc(sizeof_wb);
    Mat* ha = (Mat*) malloc(sizeof_a);
    GNN_ASSERT(hw && hb && ha);

    CUDA_CHECK(cudaMemcpy(hw, nn.w, sizeof_wb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb, nn.b, sizeof_wb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ha, nn.a, sizeof_a, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nn.count; ++i) {
        mat_fill(hw[i], n);
        mat_fill(hb[i], n);
        mat_fill(ha[i], n);
    }
    mat_fill(ha[nn.count], n);

    free(hw);
    free(hb);
    free(ha);
}

void nn_print(NN nn, const char* name) {
    size_t sizeof_wb = sizeof(Mat) * nn.count;
    Mat* hw = (Mat*) malloc(sizeof_wb);
    Mat* hb = (Mat*) malloc(sizeof_wb);
    GNN_ASSERT(hw && hb);

    CUDA_CHECK(cudaMemcpy(hw, nn.w, sizeof_wb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb, nn.b, sizeof_wb, cudaMemcpyDeviceToHost));

    printf("%s\n", name);
    for (size_t i = 0; i < nn.count; ++i) {
        mat_print(hw[i], "w");
        mat_print(hb[i], "b");
    }
    printf("\n");

    free(hw);
    free(hb);
}

#endif //GNN_IMPLEMENTATION

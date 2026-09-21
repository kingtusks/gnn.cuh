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
void nn_print(NN nn, const char* name);

#ifdef GNN_IMPLEMENTATION

NN nn_alloc(size_t* dim, size_t dim_len) {
    NN nn;
    nn.count = dim_len - 1;

    size_t sizeof_wb = sizeof(Mat) * nn.count;
    size_t sizeof_a = sizeof(Mat) * (nn.count + 1);

    Mat* w = (Mat*) malloc(sizeof_wb);
    Mat* b = (Mat*) malloc(sizeof_wb);
    Mat* a = (Mat*) malloc(sizeof_a);
    GNN_ASSERT(w && b && a);

    a[0] = mat_alloc(1, dim[0]);
    for (size_t i = 1; i < dim_len; ++i) {
        w[i - 1] = mat_alloc(dim[i - 1], dim[i]);
        b[i - 1] = mat_alloc(1, dim[i]);
        a[i] = mat_alloc(1, dim[i]);
    }

    CUDA_CHECK(cudaMalloc((void**)&nn.w, sizeof_wb));
    CUDA_CHECK(cudaMemcpy(nn.w, w, sizeof_wb, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&nn.b, sizeof_wb));
    CUDA_CHECK(cudaMemcpy(nn.b, b, sizeof_wb, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&nn.a, sizeof_a));
    CUDA_CHECK(cudaMemcpy(nn.a, a, sizeof_a, cudaMemcpyHostToDevice));

    free(w);
    free(b);
    free(a);

    return nn;
}

void nn_print(NN nn, const char* name) {
    printf("%s\n", name);
    for (size_t i = 0; i < nn.count; ++i) {
        mat_print(nn.w[i], "w");
        mat_print(nn.b[i], "b");
    }
    printf("\n");
}

#endif //GNN_IMPLEMENTATION

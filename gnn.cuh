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
void nn_forward(NN nn);
__global__ void nn_cost_kernel(Mat out, Mat y, float* dc);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, Mat ti, Mat to, float eps);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
void nn_train(NN nn, NN g, Mat ti, Mat to, float rate, size_t iter);
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

void nn_forward(NN nn) {
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
        mat_dot(ha[i + 1], ha[i], hw[i]);
        mat_sum(ha[i + 1], hb[i]);
        mat_sig(ha[i + 1]);
    }

    free(hw);
    free(hb);
    free(ha);
}

__global__ void nn_cost_kernel(Mat out, Mat y, float* dc) {
    extern __shared__ float sd[];
    size_t tx = threadIdx.x;
    size_t idx = (size_t) blockIdx.x * blockDim.x + tx;
    size_t area = out.rows * out.cols;

    float v = 0;
    if (idx < area) {
        size_t i = idx / out.cols;
        size_t j = idx % out.cols;
        float d = MAT_AT(out, i, j) - MAT_AT(y, i, j);
        v += d*d;
    }

    sd[tx] = v;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (s > tx) sd[tx] += sd[tx + s];
        __syncthreads();
    }

    if (tx == 0) 
        dc[blockIdx.x] = sd[0];
}

float nn_cost(NN nn, Mat ti, Mat to) {
    GNN_ASSERT(ti.rows == to.rows);
    size_t sizeof_a = sizeof(Mat) * (nn.count + 1);
    Mat* ha = (Mat*) malloc(sizeof_a);
    GNN_ASSERT(ha);
    CUDA_CHECK(cudaMemcpy(ha, nn.a, sizeof_a, cudaMemcpyDeviceToHost));
    GNN_ASSERT(to.cols == ha[nn.count].cols);
    size_t n = ti.rows;

    mat_copy(ha[0], ti);
    nn_forward(nn);
    Mat out = ha[nn.count];

    size_t area = (size_t) out.rows * out.cols;
    unsigned int threads = 256;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    float* dc;
    size_t sizeof_c = blocks * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&dc, sizeof_c));

    nn_cost_kernel<<<blocks, threads, threads * sizeof(float)>>>(out, to, dc);
    CUDA_CHECK(cudaGetLastError());

    float* hc = (float*) malloc(sizeof_c);
    CUDA_CHECK(cudaMemcpy(hc, dc, sizeof_c, cudaMemcpyDeviceToHost));

    float c = 0;
    for (unsigned int i = 0; i < blocks; ++i)
        c += hc[i];

    free(hc);
    CUDA_CHECK(cudaFree(dc));

    return c / n;
}

void nn_finite_diff(NN nn, NN g, Mat ti, Mat to, float eps) {
    size_t sizeof_wb = sizeof(Mat) * nn.count;
    Mat* hw = (Mat*) malloc(sizeof_wb);
    Mat* hb = (Mat*) malloc(sizeof_wb);
    Mat* hgw = (Mat*) malloc(sizeof_wb);
    Mat* hgb = (Mat*) malloc(sizeof_wb);
    GNN_ASSERT(hw && hb && hgw && hgb);

    CUDA_CHECK(cudaMemcpy(hw, nn.w, sizeof_wb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb, nn.b, sizeof_wb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hgw, g.w, sizeof_wb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hgb, g.b, sizeof_wb, cudaMemcpyDeviceToHost));

    float saved;
    float c = nn_cost(nn, ti, to);

    for (size_t i = 0; i < nn.count; ++i) {
        for (size_t j = 0; j < hw[i].rows; ++j) {
            for (size_t k = 0; k < hw[i].cols; ++k) {
                saved = MAT_AT(hw[i], j, k);
                MAT_AT(hw[i], j, k) += eps;
                MAT_AT(hgw[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(hw[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < hb[i].rows; ++j) {
            for (size_t k = 0; k < hb[i].cols; ++k) {
                saved = MAT_AT(hb[i], j, k);
                MAT_AT(hb[i], j, k) += eps;
                MAT_AT(hgb[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(hb[i], j, k) = saved;
            }
        }
    }
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to) {
    GNN_ASSERT(ti.rows == to.rows);
    size_t sizeof_wb = sizeof(Mat) * nn.count;
    size_t sizeof_a = sizeof(Mat) * (nn.count + 1);

    Mat* ha = (Mat*) malloc(sizeof_a);
    Mat* hw = (Mat*) malloc(sizeof_wb);
    Mat* hga = (Mat*) malloc(sizeof_a);
    Mat* hgw = (Mat*) malloc(sizeof_wb);
    Mat* hgb = (Mat*) malloc(sizeof_wb);

    GNN_ASSERT(ha && hw && hga && hgw && hgb);
    GNN_ASSERT(ha[nn.count].cols == to.cols);

    CUDA_CHECK(cudaMemcpy(ha, nn.a, sizeof_a, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hw, nn.w, sizeof_wb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hga, g.a, sizeof_a, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hgw, g.w, sizeof_wb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hgb, g.b, sizeof_wb, cudaMemcpyDeviceToHost));

    size_t n = ti.rows;
    nn_fill(g, 0);
    for (size_t i = 0; i < n; ++i) {
        mat_copy(ha[0], mat_row(ti, i));
        nn_forward(nn);

        for (size_t j = 0; j <= nn.count; ++j)
            mat_fill(hga[j], 0);
        for (size_t j = 0; j < to.cols; ++j)
            MAT_AT(hga[g.count], 0, j) = MAT_AT(ha[nn.count], 0, j) - MAT_AT(to, i, j);

        for (size_t l = nn.count; l > 0; --l) {
            for (size_t j = 0; j < ha[l].cols; ++j) {
                float a = MAT_AT(ha[l], 0, j);
                float da = MAT_AT(hga[l], 0, j);
                MAT_AT(hgb[l - 1], 0, j) += 2 * da * a * (1 - a);
                for (size_t k = 0; k < ha[l - 1].cols; ++k) {
                    float pa = MAT_AT(ha[l - 1], 0, k);
                    float w = MAT_AT(hw[l - 1], k, j);
                    MAT_AT(hgw[l - 1], k, j) += 2 * da * a * (1 - a) * pa;
                    MAT_AT(hga[l - 1], 0, k) += 2 * da * a * (1 - a) * w;
                }
            }
        }
    }

    for (size_t i = 0; i < g.count; ++i) {
        for (size_t j = 0; j < hgw[i].rows; ++j) {
            for (size_t k = 0; k < hgw[i].cols; ++k)
                MAT_AT(hgw[i], j, k) /= n;
        }
        for (size_t j = 0; j < hgb[i].rows; ++j) {
            for (size_t k = 0; k < hgb[i].cols; ++k)
                MAT_AT(hgb[i], j, k) /= n;
        }
    }
}

__global__ void nn_learn_kernel(NN nn, NN g, float rate) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = (size_t) blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nn.count) return;

    if (j < nn.w[i].rows && k < nn.w[i].cols)
        MAT_AT(nn.w[i], j, k) -= rate * MAT_AT(g.w[i], j, k);

    if (j < nn.b[i].rows && k < nn.b[i].cols)
        MAT_AT(nn.b[i], j, k) -= rate * MAT_AT(g.b[i], j, k);
}

void nn_learn(NN nn, NN g, float rate) {
    Mat* hw_0 = (Mat*) malloc(sizeof(Mat));
    GNN_ASSERT(hw_0);
    CUDA_CHECK(cudaMemcpy(hw_0, nn.w[0], sizeof(Mat), cudaMemcpyDeviceToHost));

    dim3 threads(8, 8, 8);
    dim3 blocks(
        (nn.count + threads.x - 1) / threads.x,
        (hw_0->rows + threads.y - 1) / threads.y,
        (hw_0->cols + threads.z - 1) / threads.z
    );

    free(hw_0);
    nn_learn_kernel<<<blocks, threads>>>(nn, g, rate);
    CUDA_CHECK(cudaGetLastError());
}

void nn_train(NN nn, NN g, Mat ti, Mat to, float rate, size_t iter) {
    for (size_t i = 0; i < iter; ++i) {
        g = nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        printf("%zu: cost: %f", i, nn_cost(nn, ti, to));
    }
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

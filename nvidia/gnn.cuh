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
    size_t count, batch;
    Mat *w, *b, *a;
    float *dc;
} NN;

#define ARRAY_LEN(arr) (sizeof((arr)) / sizeof((arr)[0]))
#define NN_INPUT(nn) (nn).a[0]
#define NN_OUTPUT(nn) (nn).a[(nn).count]
#define NN_PRINT(nn) nn_print(nn, #nn)

NN nn_alloc(size_t* dim, size_t dim_len, size_t batch);
void nn_rand(NN nn, float low, float high);
void nn_fill(NN nn, float n);
void nn_forward(NN nn);
__global__ void nn_cost_kernel(Mat out, Mat y);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, Mat ti, Mat to, float eps);
__global__ void nn_backprop_output_kernel(Mat ha_out, Mat hga_out, Mat to_row);
__global__ void nn_backprop_gradient_kernel(Mat ha_curr, Mat ga_curr, Mat ha_prev, Mat hw_prev, Mat hgw_prev, Mat hgb_prev, Mat hga_prev);
__global__ void nn_backprop_divider_kernel(NN g, size_t n);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
__global__ void nn_learn_kernel(NN nn, NN g, float rate);
void nn_learn(NN nn, NN g, float rate);
void nn_train(NN nn, NN g, Mat ti, Mat to, float rate, size_t iter);
void nn_print(NN nn, const char* name);

#ifdef GNN_IMPLEMENTATION

NN nn_alloc(size_t* dim, size_t dim_len, size_t batch) {
    NN nn;
    nn.count = dim_len - 1;
    nn.batch = batch;
    size_t sizeof_wb = sizeof(Mat) * nn.count;
    size_t sizeof_a = sizeof(Mat) * (nn.count + 1);

    nn.hw = (Mat*) malloc(sizeof_wb);
    nn.hb = (Mat*) malloc(sizeof_wb);
    nn.ha = (Mat*) malloc(sizeof_a);
    GNN_ASSERT(nn.hw && nn.hb && nn.ha);

    nn.ha[0] = mat_alloc(batch, dim[0]);
    for (size_t i = 1; i < dim_len; ++i) {
        nn.hw[i - 1] = mat_alloc(dim[i - 1], dim[i]);
        nn.hb[i - 1] = mat_alloc(1, dim[i]);
        nn.ha[i] = mat_alloc(batch, dim[i]);
    }

    CUDA_CHECK(cudaMalloc((void**)&nn.w, sizeof_wb));
    CUDA_CHECK(cudaMemcpy(nn.w, nn.hw, sizeof_wb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&nn.b, sizeof_wb));
    CUDA_CHECK(cudaMemcpy(nn.b, nn.hb, sizeof_wb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&nn.a, sizeof_a));
    CUDA_CHECK(cudaMemcpy(nn.a, nn.ha, sizeof_a, cudaMemcpyHostToDevice));
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

    Mat out = ha[nn.count];

    unsigned int threads = 256;
    size_t area = (size_t) out.rows * out.cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    size_t sizeof_c = blocks * sizeof(float);
    float* dc;
    CUDA_CHECK(cudaMalloc((void**)&dc, sizeof_c));
    float* hc = (float*) malloc(sizeof_c);

    float c = 0;
    for (size_t i = 0; i < ti.rows; ++i) {
        mat_copy(ha[0], mat_row(ti, i));
        nn_forward(nn);
        nn_cost_kernel<<<blocks, threads, threads * sizeof(float)>>>(out, mat_row(to, i), dc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hc, dc, sizeof_c, cudaMemcpyDeviceToHost));
        for (unsigned int b = 0; b < blocks; ++b)
            c += hc[b];
    }

    free(ha);
    free(hc);
    CUDA_CHECK(cudaFree(dc));

    return c / ti.rows;
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

    float saved, cp, grad;
    float c = nn_cost(nn, ti, to);

    for (size_t i = 0; i < nn.count; ++i) {
        for (size_t j = 0; j < hw[i].rows; ++j) {
            for (size_t k = 0; k < hw[i].cols; ++k) {
                float* w = &MAT_AT(hw[i], j, k);

                CUDA_CHECK(cudaMemcpy(&saved, w, sizeof(float), cudaMemcpyDeviceToHost));

                float nudged = saved + eps;
                CUDA_CHECK(cudaMemcpy(w, &nudged, sizeof(float), cudaMemcpyHostToDevice));

                cp = nn_cost(nn, ti, to);
                grad = (cp - c) / eps;

                CUDA_CHECK(cudaMemcpy(w, &saved, sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(&MAT_AT(hgw[i], j, k), &grad, sizeof(float), cudaMemcpyHostToDevice));
            }
        }

        for (size_t j = 0; j < hb[i].rows; ++j) {
            for (size_t k = 0; k < hb[i].cols; ++k) {
                float* b = &MAT_AT(hb[i], j, k);

                CUDA_CHECK(cudaMemcpy(&saved, b, sizeof(float), cudaMemcpyDeviceToHost));

                float nudged = saved + eps;
                CUDA_CHECK(cudaMemcpy(b, &nudged, sizeof(float), cudaMemcpyHostToDevice));

                cp = nn_cost(nn, ti, to);
                grad = (cp - c) / eps;

                CUDA_CHECK(cudaMemcpy(b, &saved, sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(&MAT_AT(hgb[i], j, k), &grad, sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }

    free(hw);
    free(hb);
    free(hgw);
    free(hgb);
}

__global__ void nn_backprop_output_kernel(Mat ha_out, Mat hga_out, Mat to_row) {
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ha_out.cols)
        MAT_AT(hga_out, 0, j) = MAT_AT(ha_out, 0, j) - MAT_AT(to_row, 0, j);
}

__global__ void nn_backprop_gradient_kernel(Mat ha_curr, Mat ga_curr, Mat ha_prev, Mat hw_prev, Mat hgw_prev, Mat hgb_prev, Mat hga_prev) {
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t k = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ha_curr.cols || k >= ha_prev.cols) return;

    float d = 2 * MAT_AT(ga_curr, 0, j) * MAT_AT(ha_curr, 0, j) * (1 - MAT_AT(ha_curr, 0, j));
    MAT_AT(hgw_prev, k, j) += d * MAT_AT(ha_prev, 0, k);
    atomicAdd(&MAT_AT(hga_prev, 0, k), d * MAT_AT(hw_prev, k, j));
    if (k == 0) MAT_AT(hgb_prev, 0, j) += d;
}

__global__ void nn_backprop_divider_kernel(NN g, size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = (size_t) blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= g.count) return;

    if (j < g.w[i].rows && k < g.w[i].cols)
        MAT_AT(g.w[i], j, k) /= n;

    if (j < g.b[i].rows && k < g.b[i].cols)
        MAT_AT(g.b[i], j, k) /= n;
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to) {
    GNN_ASSERT(ti.rows == to.rows);
    size_t sizeof_a = sizeof(Mat) * (nn.count + 1);
    size_t sizeof_wb = sizeof(Mat) * nn.count;

    Mat* ha = (Mat*) malloc(sizeof_a);
    GNN_ASSERT(ha);
    CUDA_CHECK(cudaMemcpy(ha, nn.a, sizeof_a, cudaMemcpyDeviceToHost));
    GNN_ASSERT(ha[nn.count].cols == to.cols);

    Mat* hw = (Mat*) malloc(sizeof_wb);
    Mat* hga = (Mat*) malloc(sizeof_a);
    Mat* hgw = (Mat*) malloc(sizeof_wb);
    Mat* hgb = (Mat*) malloc(sizeof_wb);
    GNN_ASSERT(hw && hga && hgw && hgb);
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

        unsigned int output_threads = 256;
        unsigned int output_blocks = (unsigned int) ((to.cols + output_threads - 1) / output_threads);
        nn_backprop_output_kernel<<<output_blocks, output_threads>>>(ha[nn.count], hga[nn.count], mat_row(to, i));
        CUDA_CHECK(cudaGetLastError());

        for (size_t l = nn.count; l > 0; --l) {
            dim3 gradient_threads(16, 16);
            dim3 gradient_blocks(
                (unsigned int) ((ha[l].cols + gradient_threads.x - 1) / gradient_threads.x),
                (unsigned int) ((ha[l - 1].cols + gradient_threads.y - 1) / gradient_threads.y)
            );
            nn_backprop_gradient_kernel<<<gradient_blocks, gradient_threads>>>(
                ha[l], hga[l], ha[l - 1], hw[l - 1], hgw[l - 1], hgb[l - 1], hga[l - 1]
            );
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t max_r = 0;
    size_t max_c = 0;
    for (size_t i = 0; i < g.count; ++i) {
        if (hgw[i].rows > max_r) max_r = hgw[i].rows;
        if (hgw[i].cols > max_c) max_c = hgw[i].cols;
    }

    dim3 threads(8, 8, 8);
    dim3 blocks(
        (unsigned int) ((nn.count + threads.x - 1) / threads.x),
        (unsigned int) ((max_r + threads.y - 1) / threads.y),
        (unsigned int) ((max_c + threads.z - 1) / threads.z)
    );
    nn_backprop_divider_kernel<<<blocks, threads>>>(g, n);
    CUDA_CHECK(cudaGetLastError());

    free(ha);
    free(hw);
    free(hga);
    free(hgw);
    free(hgb);
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
    size_t sizeof_wb = sizeof(Mat) * nn.count;
    Mat* hw = (Mat*) malloc(sizeof_wb);
    CUDA_CHECK(cudaMemcpy(hw, nn.w, sizeof_wb, cudaMemcpyDeviceToHost));
    GNN_ASSERT(hw);

    dim3 threads(8, 8, 8);
    dim3 blocks(
        (nn.count + threads.x - 1) / threads.x,
        (hw[0].rows + threads.y - 1) / threads.y,
        (hw[0].cols + threads.z - 1) / threads.z
    );

    free(hw);
    nn_learn_kernel<<<blocks, threads>>>(nn, g, rate);
    CUDA_CHECK(cudaGetLastError());
}

void nn_train(NN nn, NN g, Mat ti, Mat to, float rate, size_t iter) {
    for (size_t i = 0; i < iter; ++i) {
#if 1
        nn_backprop(nn, g, ti, to);
#else
        nn_finite_diff(nn, g, ti, to, 1e-2);
#endif
        nn_learn(nn, g, rate);
        printf("%zu: cost: %f\n", i, nn_cost(nn, ti, to));
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

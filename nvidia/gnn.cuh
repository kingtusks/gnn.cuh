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

#ifndef NN_PRINT_INTERVAL
#define NN_PRINT_INTERVAl 5
#endif //NN_PRINT_INTERVAL

typedef struct {
    size_t count, batch;
    Mat *w, *b, *a;
    float *dc;
} NN;

#define ARRAY_LEN(arr) (sizeof((arr)) / sizeof((arr)[0]))
#define NN_INPUT(nn) (nn).ha[0]
#define NN_OUTPUT(nn) (nn).ha[(nn).count]
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

    nn.w = (Mat*) malloc(sizeof_wb);
    nn.b = (Mat*) malloc(sizeof_wb);
    nn.a = (Mat*) malloc(sizeof_a);
    GNN_ASSERT(nn.w && nn.b && nn.a);

    nn.ha[0] = mat_alloc(batch, dim[0]);
    for (size_t i = 1; i < dim_len; ++i) {
        nn.hw[i - 1] = mat_alloc(dim[i - 1], dim[i]);
        nn.hb[i - 1] = mat_alloc(1, dim[i]);
        nn.ha[i] = mat_alloc(batch, dim[i]);
    }

    CUDA_CHECK(cudaMalloc((void**)&dc, sizeof(float));
    return nn;
}

void nn_rand(NN nn, float low, float high) {
    for (size_t i = 0; i < nn.count; ++i) {
        mat_rand(nn.w[i], low, high);
        mat_rand(nn.b[i], low, high);
    }
}

void nn_fill(NN nn, float n) {
    for (size_t i = 0; i < nn.count; ++i) {
        mat_fill(nn.w[i], n);
        mat_fill(nn.b[i], n);
        mat_fill(nn.a[i], n);
    }
    mat_fill(NN_OUTPUT(nn), n);
}

void nn_forward(NN nn) {
    for (size_t i = 0; i < nn.count; ++i) {
        mat_dot(nn.a[i + 1], nn.a[i], nn.w[i]);
        mat_sum_bias(nn.a[i + 1], nn.b[i]);
        mat_sig(nn.a[i + 1]);
    }
}

__global__ void nn_cost_kernel(Mat out, Mat y, float* dcost) {
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
        atomicAdd(dcost, sd[0])
}

float nn_cost(NN nn, Mat ti, Mat to) {
    GNN_ASSERT(ti.rows == to.rows);
    GNN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    unsigned int threads = 256;
    size_t area = (size_t) NN_OUTPUT(nn).rows * NN_OUTPUT(nn).cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    CUDA_CHECK(cudaMemset(nn.dc, 0, sizeof(float)));

    size_t nb = 0;
    for (size_t i = 0; i + nn.batch <= ti.rows; i += nn.batch; ++nb) {
        mat_copy(NN_INPUT(nn), mat_rows(ti, i, nn.batch));
        nn_forward(nn);
        nn_cost_kernel<<<blocks, threads, threads * sizeof(float)>>>(NN_OUTPUT(nn), mat_rows(to, i, nn.batch), nn.dc);
        CUDA_CHECK(cudaGetLastError());
    }
    GNN_ASSERT(nb > 0);

    float c;
    CUDA_CHECK(cudaMemcpy(&c, nn.dc, sizeof(float), cudaMemcpyDeviceToHost));
    return c / (float) (nb * nn.batch);
}
//finite diff

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

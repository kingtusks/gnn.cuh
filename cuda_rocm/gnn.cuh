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
#define NN_PRINT_INTERVAL 5
#endif //NN_PRINT_INTERVAL

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
void nn_rand_he(NN nn);
void nn_fill(NN nn, float n);
void nn_forward(NN nn);
__global__ void nn_cost_kernel(Mat out, Mat y, float* dcost);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, Mat ti, Mat to, float eps);
__global__ void nn_out_grad_kernel(Mat a, Mat y, Mat da, float scale);
__global__ void nn_activation_grad_kernel(Mat da, Mat a);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_sgd(NN nn, NN g, float rate);
void nn_momentum(NN nn, NN g, float rate);
void nn_train(NN nn, NN g, Mat ti, Mat to, float rate, size_t epochs);
__global__ void nn_test_kernel(Mat out, Mat y, float* dcost);
float nn_test(NN nn, Mat ti, Mat to);
void nn_free(NN nn);
void nn_save(NN nn, const char* path);
void nn_load(NN nn, const char* path);
void nn_print(NN nn, const char* name);

static void nn_finite_diff_nudge(NN nn, Mat m, Mat gm, Mat ti, Mat to, float eps, float c) {
    for (size_t j = 0; j < m.rows; ++j) {
        for (size_t k = 0; k < m.cols; ++k) {
            float* p = &MAT_AT(m, j, k);
            float saved, nudged, grad;

            CUDA_CHECK(cudaMemcpy(&saved, p, sizeof(float), cudaMemcpyDeviceToHost));
            nudged = saved + eps;
            CUDA_CHECK(cudaMemcpy(p, &nudged, sizeof(float), cudaMemcpyHostToDevice));

            grad = (nn_cost(nn, ti, to) - c) / eps;

            CUDA_CHECK(cudaMemcpy(p, &saved, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(&MAT_AT(gm, j, k), &grad, sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

static void nn_write_u64(FILE* f, uint64_t n) {
    size_t ok = fwrite(&v, sizeof(v), 1, f);
    GNN_ASSERT(ok == 1);
    (void) ok;
}

static uint64_t nn_read_u64(FILE* f) {
    uint64_t v = 0;
    size_t ok = fread(&v, sizeof(v), 1, f);
    GNN_ASSERT(ok == 1);
    (void) ok;
}

static void nn_mat_write(FILE* f, Mat m) {
    size_t n = m.rows * m.cols;
    float* h = (float*) malloc(n * sizeof(float));
    GNN_ASSERT(h);
    CUDA_CHECK(cudaMemcpy(h, m.es, n * sizeof(float), cudaMemcpyDeviceToHost));
    size_t ok = fwrite(h, sizeof(float), n, f);
    GNN_ASSERT(ok == n);
    (void) ok;
    free(h);
}

static void nn_mat_read(FILE* f, Mat m) {
    size_t n = m.rows * m.cols;
    float* h = (float*) malloc(n * sizeof(float));
    GNN_ASSERT(h);
    size_t ok = fread(h, sizeof(float), n, f);
    GNN_ASSERT(ok == n);
    (void) ok;
    CUDA_CHECK(cudaMemcpy(m.es, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

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

    NN_INPUT(nn) = mat_alloc(batch, dim[0]);
    for (size_t i = 1; i < dim_len; ++i) {
        nn.w[i - 1] = mat_alloc(dim[i - 1], dim[i]);
        nn.b[i - 1] = mat_alloc(1, dim[i]);
        nn.a[i] = mat_alloc(batch, dim[i]);
    }

    CUDA_CHECK(cudaMalloc((void**)&nn.dc, sizeof(float)));
    return nn;
}

void nn_rand(NN nn, float low, float high) {
    for (size_t i = 0; i < nn.count; ++i) {
        mat_rand(nn.w[i], low, high);
        mat_rand(nn.b[i], low, high);
    }
}

void nn_rand_he(NN nn) {
    for (size_t i = 0; i < nn.count; ++i) {
        float lim = sqrtf(6.f / (float) nn.w[i].rows);
        mat_rand(nn.w[i], -lim, lim);
        mat_fill(nn.b[i], 0.f);
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

void nn_forward(NN nn) { for (size_t i = 0; i < nn.count; ++i) {
        mat_dot(nn.a[i + 1], nn.a[i], nn.w[i]);
        mat_sum_bias(nn.a[i + 1], nn.b[i]);
#ifdef GNN_SIGMOID
        mat_sig(nn.a[i + 1]);
#else
        if (i + 1 < nn.count) mat_leaky_relu(nn.a[i + 1]);
        // if (i + 1 < nn.count) mat_relu(nn.a[i + 1]);
        else mat_softmax(nn.a[i + 1]);
#endif
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
#ifdef GNN_SIGMOID
        float d = MAT_AT(out, i, j) - MAT_AT(y, i, j);
        v += d*d;
#else
        // below code makes the cost return 20.723... except of nan (which is good but weird as it gets stuck there)
        // we can deduce MAT_AT(out, i, j) is less than 1e-9 thus it hangs on 1e-9
        // v += -MAT_AT(y, i, j) * logf(fmaxf(MAT_AT(out, i, j), 1e-9)) + 1e-7f;
        v += -MAT_AT(y, i, j) * logf(MAT_AT(out, i, j) + 1e-7f);
#endif
    }

    sd[tx] = v;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (s > tx) sd[tx] += sd[tx + s];
        __syncthreads();
    }

    if (tx == 0)
        atomicAdd(dcost, sd[0]);
}

float nn_cost(NN nn, Mat ti, Mat to) {
    GNN_ASSERT(ti.rows == to.rows);
    GNN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    unsigned int threads = 256;
    size_t area = (size_t) NN_OUTPUT(nn).rows * NN_OUTPUT(nn).cols;
    unsigned int blocks = (unsigned int) ((area + threads - 1) / threads);

    CUDA_CHECK(cudaMemset(nn.dc, 0, sizeof(float)));

    size_t nb = 0;
    for (size_t i = 0; i + nn.batch <= ti.rows; i += nn.batch, ++nb) {
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

void nn_finite_diff(NN nn, NN g, Mat ti, Mat to, float eps) {
    float c = nn_cost(nn, ti, to);
    for (size_t i = 0; i < nn.count; ++i) {
        nn_finite_diff_nudge(nn, nn.w[i], g.w[i], ti, to, eps, c);
        nn_finite_diff_nudge(nn, nn.b[i], g.b[i], ti, to, eps, c);
    }
}

__global__ void nn_out_grad_kernel(Mat a, Mat y, Mat da, float scale) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.rows && j < a.cols)
        MAT_AT(da, i, j) = scale * (MAT_AT(a, i, j) - MAT_AT(y, i, j));
}

__global__ void nn_activation_grad_kernel(Mat da, Mat a) {
    size_t i = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
#ifdef GNN_SIGMOID
    if (i < a.rows && j < a.cols) {
        float v = MAT_AT(a, i, j);
        MAT_AT(da, i, j) *= v * (1.f - v);
    }
#else
    if (i < a.rows && j < a.cols && MAT_AT(a, i, j) <= 0.f) MAT_AT(da, i, j) *= 0.01f;
#endif
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to) {
    GNN_ASSERT(ti.rows == nn.batch && to.rows == nn.batch);
    GNN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    mat_copy(NN_INPUT(nn), ti);
    nn_forward(nn);

    dim3 t(32, 8);
    Mat out = NN_OUTPUT(nn);
    dim3 blk((unsigned int) ((out.cols + t.x - 1) / t.x), (unsigned int) ((out.rows + t.y - 1) / t.y));
#ifdef GNN_SIGMOID
    nn_out_grad_kernel<<<blk, t>>>(out, to, g.a[nn.count], 2.f / (float) nn.batch);
#else
    nn_out_grad_kernel<<<blk, t>>>(out, to, g.a[nn.count], 1.f / (float) nn.batch);
#endif
    CUDA_CHECK(cudaGetLastError());

    for (size_t l = nn.count; l > 0; --l) {
        Mat a = nn.a[l];
        Mat dz = g.a[l];
#ifdef GNN_SIGMOID
        dim3 b2((unsigned int) ((a.cols + t.x - 1) / t.x), (unsigned int) ((a.rows + t.y - 1) / t.y));
        nn_activation_grad_kernel<<<b2, t>>>(dz, a);
        CUDA_CHECK(cudaGetLastError());
#else
        if (l < nn.count) {
            dim3 b2((unsigned int) ((a.cols + t.x - 1) / t.x), (unsigned int) ((a.rows + t.y - 1) / t.y));
            nn_activation_grad_kernel<<<b2, t>>>(dz, a);
            CUDA_CHECK(cudaGetLastError());
        }
#endif
        mat_dot_ta(g.w[l - 1], nn.a[l - 1], dz);
        mat_sum_collapse(g.b[l - 1], dz);
        if (l > 1)
            mat_dot_tb(g.a[l - 1], dz, nn.w[l - 1]);
    }
}

void nn_sgd(NN nn, NN g, float rate) {
    for (size_t i = 0; i < nn.count; ++i) {
        mat_sub_scaled(nn.w[i], g.w[i], rate);
        mat_sub_scaled(nn.b[i], g.b[i], rate);
    }
}

void nn_momentum(NN nn, NN g, float rate) {
    float vw = 0;
    float vb = 0;
    for (size_t i = 0; i < nn.count; ++i) {
        vw = 0.9 * v + g.w[i];
        vb = 0.9 * v + g.b[i];
        mat_sub_scaled(nn.w[i], vw, rate);
        mat_sub_scaled(nn.b[i], vb, rate);
    }
}

void nn_train(NN nn, NN g, Mat ti, Mat to, float rate, size_t epochs) {
    for (size_t i = 1; i <= epochs; ++i) {
        for (size_t j = 0; j + nn.batch <= ti.rows; j += nn.batch) {
            nn_backprop(nn, g, mat_rows(ti, j, nn.batch), mat_rows(to, j, nn.batch));
            nn_sgd(nn, g, rate);
        }
        if (i % NN_PRINT_INTERVAL == 0)
            printf("%zu: cost: %f\n", i, nn_cost(nn, ti, to));
    }
}

__global__ void nn_test_kernel(Mat out, Mat y, float* dcost) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out.rows) return;

    size_t po = 0, py = 0;
    for (size_t j = 1; j < out.cols; ++j) {
        if (MAT_AT(out, i, j) > MAT_AT(out, i, po)) po = j;
        if (MAT_AT(y, i, j) > MAT_AT(y, i, py)) py = j;
    }

    if (po == py) atomicAdd(dcost, 1.f);
}

float nn_test(NN nn, Mat ti, Mat to) {
    GNN_ASSERT(ti.rows == to.rows);
    GNN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    unsigned int threads = 256;
    unsigned int blocks = (unsigned int) ((nn.batch + threads - 1) / threads);

    CUDA_CHECK(cudaMemset(nn.dc, 0, sizeof(float)));

    size_t nb = 0;
    for (size_t i = 0; i + nn.batch <= ti.rows; i += nn.batch, ++nb) {
        mat_copy(NN_INPUT(nn), mat_rows(ti, i, nn.batch));
        nn_forward(nn);
        nn_test_kernel<<<blocks, threads>>>(NN_OUTPUT(nn), mat_rows(to, i, nn.batch), nn.dc);
        CUDA_CHECK(cudaGetLastError());
    }
    GNN_ASSERT(nb > 0);

    float correct;
    CUDA_CHECK(cudaMemcpy(&correct, nn.dc, sizeof(float), cudaMemcpyDeviceToHost));

    size_t total = nb * nn.batch;
    float acc = 100.f * correct / (float) total;
    printf("accuracy: %.2f%% (%d / %zu)\n", acc, (int) correct, total);
    return acc;
}

void nn_free(NN nn) {
    for (size_t i = 0; i < nn.count; ++i) {
        CUDA_CHECK(cudaFree(nn.w[i].es));
        CUDA_CHECK(cudaFree(nn.b[i].es));
        CUDA_CHECK(cudaFree(nn.a[i].es));
    }
    CUDA_CHECK(cudaFree(nn.a[nn.count].es));
    CUDA_CHECK(cudaFree(nn.dc));
    free(nn.w);
    free(nn.b);
    free(nn.a);
}

void nn_save(NN nn, const char* path) {
    FILE* f = fopen(path, "wb");
    GNN_ASSERT(f);

    nn_write_u64(f, nn.count);
    for (size_t i = 0; i < nn.count; ++i) {
        nn_write_u64(f, nn.w[i].rows);
        nn_write_u64(f, nn.w[i].cols);
        nn_mat_write(f, nn.w[i]);
        nn_mat_write(f, nn.b[i]);
    }

    fclose(f);
}

void nn_load(NN nn, const char* path) {
    FILE* f = fopen(path, "rb");
    GNN_ASSERT(f);

    uint64_t count = nn_read_u64(f);
    GNN_ASSERT(count == nn.count);
    (void) count;

    for (size_t i = 0; i < nn.count; ++i) {
        uint64_t r = nn_read_u64(f);
        uint64_t c = nn_read_u64(f);
        GNN_ASSERT(r == nn.w[i].rows && c == nn.w[i].cols);
        (void) r;
        (void) c;
        nn_mat_read(f, nn.w[i]);
        nn_mat_read(f, nn.b[i]);
    }

    fclose(f);
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

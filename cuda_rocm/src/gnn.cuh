#pragma once
#include "gmat.cuh"

#ifndef GNN_MALLOC
#include <stdlib.h>
#define GNN_MALLOC malloc
#endif //GNN_MALLOC

namespace NeuralNetwork {
#define NN_PRINT(nn) nn_print(nn, #nn)
#define ARRAY_LEN(arr) (sizeof((arr)) / sizeof((arr)[0]))
#define NN_INPUT(nn) (nn).a[0]
#define NN_OUTPUT(nn) (nn).a[(nn).count]
    inline constexpr unsigned int NN_PRINT_INTERVAL{5};
    struct NN {
        size_t count, batch;
        Matrix::Mat *w, *b, *a;
        float *dc;
    };

    enum class Optimizer {
        SGD,
        SGDMomentum,
        Adam
    };

    NN nn_alloc(size_t* dim, size_t dim_len, size_t batch);
    NN nn_alloc_copy(NN nn);
    void nn_rand(NN nn, float low, float high);
    void nn_rand_he(NN nn);
    void nn_fill(NN nn, float n);
    void nn_forward(NN nn);
    __global__ void nn_cost_kernel(Matrix::Mat out, Matrix::Mat y, float* dcost);
    float nn_cost(NN nn, Matrix::Mat ti, Matrix::Mat to);
    void nn_finite_diff(NN nn, NN g, Matrix::Mat ti, Matrix::Mat to, float eps);
    __global__ void nn_out_grad_kernel(Matrix::Mat a, Matrix::Mat y, Matrix::Mat da, float scale);
    __global__ void nn_activation_grad_kernel(Matrix::Mat da, Matrix::Mat a);
    void nn_backprop(NN nn, NN g, Matrix::Mat ti, Matrix::Mat to);
    void nn_sgd(NN nn, NN g, float rate);
    void nn_momentum(NN nn, NN g, NN v, float rate);
    void nn_adam(NN nn, NN g, NN m1, NN m2, size_t t, float rate);
    void nn_train(NN nn, Matrix::Mat ti, Matrix::Mat to, size_t epochs = 1000, float rate = 3e-4, Optimizer opt = Optimizer::Adam);
    __global__ void nn_test_kernel(Matrix::Mat out, Matrix::Mat y, float* dcost);
    float nn_test(NN nn, Matrix::Mat ti, Matrix::Mat to);
    void nn_infer(NN nn, Matrix::Mat ti);
    void nn_free(NN nn);
    void nn_save(NN nn, const char* path);
    void nn_load(NN nn, const char* path);
    void nn_print(NN nn, const char* name);

    static void nn_finite_diff_nudge(NN nn, Matrix::Mat m, Matrix::Mat gm, Matrix::Mat ti, Matrix::Mat to, float eps, float c) {
        for (size_t j{}; j < m.rows; ++j) {
            for (size_t k{}; k < m.cols; ++k) {
                float* p{&MAT_AT(m, j, k)};
                float saved, nudged, grad;

                GNN_CUDA_CHECK(cudaMemcpy(&saved, p, sizeof(float), cudaMemcpyDeviceToHost));
                nudged = saved + eps;
                GNN_CUDA_CHECK(cudaMemcpy(p, &nudged, sizeof(float), cudaMemcpyHostToDevice));

                grad = (nn_cost(nn, ti, to) - c) / eps;

                GNN_CUDA_CHECK(cudaMemcpy(p, &saved, sizeof(float), cudaMemcpyHostToDevice));
                GNN_CUDA_CHECK(cudaMemcpy(&MAT_AT(gm, j, k), &grad, sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }

    static void nn_write_u64(FILE* f, uint64_t n) {
        size_t ok{fwrite(&n, sizeof(n), 1, f)};
        GNN_ASSERT(ok == 1);
        (void) ok;
    }

    static uint64_t nn_read_u64(FILE* f) {
        uint64_t v{};
        size_t ok{fread(&v, sizeof(v), 1, f)};
        GNN_ASSERT(ok == 1);
        (void) ok;
        return v;
    }

    static void nn_mat_write(FILE* f, Matrix::Mat m) {
        size_t n{m.rows * m.cols};
        float* h{(float*) malloc(n * sizeof(float))};
        GNN_ASSERT(h);
        GNN_CUDA_CHECK(cudaMemcpy(h, m.es, n * sizeof(float), cudaMemcpyDeviceToHost));
        size_t ok{fwrite(h, sizeof(float), n, f)};
        GNN_ASSERT(ok == n);
        (void) ok;
        free(h);
    }

    static void nn_mat_read(FILE* f, Matrix::Mat m) {
        size_t n{m.rows * m.cols};
        float* h{(float*) malloc(n * sizeof(float))};
        GNN_ASSERT(h);
        size_t ok{fread(h, sizeof(float), n, f)};
        GNN_ASSERT(ok == n);
        (void) ok;
        GNN_CUDA_CHECK(cudaMemcpy(m.es, h, n * sizeof(float), cudaMemcpyHostToDevice));
        free(h);
    }
}

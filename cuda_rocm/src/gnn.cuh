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
    inline constexpr unsigned int NN_PRINT_INTERVAL{};

    struct NN {
        size_t count, batch;
        Mat *w, *b, *a;
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
    __global__ void nn_cost_kernel(Mat out, Mat y, float* dcost);
    float nn_cost(NN nn, Mat ti, Mat to);
    void nn_finite_diff(NN nn, NN g, Mat ti, Mat to, float eps);
    __global__ void nn_out_grad_kernel(Mat a, Mat y, Mat da, float scale);
    __global__ void nn_activation_grad_kernel(Mat da, Mat a);
    void nn_backprop(NN nn, NN g, Mat ti, Mat to);
    void nn_sgd(NN nn, NN g, float rate);
    void nn_momentum(NN nn, NN g, NN v, float rate);
    void nn_adam(NN nn, NN g, NN m1, NN m2, size_t t, float rate);
    void nn_train(NN nn, Mat ti, Mat to, size_t epochs = 1000, float rate = 3e-4, Optimizer opt = Optimizer::Adam);
    __global__ void nn_test_kernel(Mat out, Mat y, float* dcost);
    float nn_test(NN nn, Mat ti, Mat to);
    void nn_infer(NN nn, Mat ti);
    void nn_free(NN nn);
    void nn_save(NN nn, const char* path);
    void nn_load(NN nn, const char* path);
    void nn_print(NN nn, const char* name);

    static void nn_finite_diff_nudge(NN nn, Mat m, Mat gm, Mat ti, Mat to, float eps, float c) {
        for (size_t j{0}; j < m.rows; ++j) {
            for (size_t k{0}; k < m.cols; ++k) {
                float* p{&MAT_AT(m, j, k)};
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

    static void nn_mat_write(FILE* f, Mat m) {
        size_t n{m.rows * m.cols};
        float* h{(float*) malloc(n * sizeof(float))};
        GNN_ASSERT(h);
        CUDA_CHECK(cudaMemcpy(h, m.es, n * sizeof(float), cudaMemcpyDeviceToHost));
        size_t ok{fwrite(h, sizeof(float), n, f)};
        GNN_ASSERT(ok == n);
        (void) ok;
        free(h);
    }

    static void nn_mat_read(FILE* f, Mat m) {
        size_t n{m.rows * m.cols};
        float* h{(float*) malloc(n * sizeof(float))};
        GNN_ASSERT(h);
        size_t ok{fread(h, sizeof(float), n, f)};
        GNN_ASSERT(ok == n);
        (void) ok;
        CUDA_CHECK(cudaMemcpy(m.es, h, n * sizeof(float), cudaMemcpyHostToDevice));
        free(h);
    }
}

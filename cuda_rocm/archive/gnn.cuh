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
    void nn_finite_diff_nudge(NN nn, Matrix::Mat m, Matrix::Mat gm, Matrix::Mat ti, Matrix::Mat to, float eps, float c);
    void nn_write_u64(FILE* f, uint64_t n);
    uint64_t nn_read_u64(FILE* f);
    void nn_mat_write(FILE* f, Matrix::Mat m);
    void nn_mat_read(FILE* f, Matrix::Mat m);
}

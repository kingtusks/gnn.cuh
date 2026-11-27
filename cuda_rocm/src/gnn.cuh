#pragma once
#include "gmat.cuh"

namespace gnn
{
    #define NN_PRINT(nn) nn_print(#nn);

    enum class Optimizer
    {
        SGD,
        SGDMomentum,
        Adam
    };

    struct NN
    {
        size_t count_ {};
        unsigned int batch_ {};
        std::vector<Tensor> w_, b_, a_;
        Tensor dc_ {1};

        NN(const std::vector<unsigned int>& dims, unsigned int batch);

        Tensor& input() { return a_.front(); };
        Tensor& output() { return a_.back(); };

        void nn_rand(float low = {0.0f}, float high = {1.0f});
        void nn_rand_he();
        void nn_fill(float n);
        void nn_forward();
        float nn_cost(const Tensor& ti, const Tensor& to);
        void finite_diff(NN& g, const Tensor& ti, const Tensor& to, float eps = {1e-3f});
        void backprop(NN& g, const Tensor& ti, const Tensor& to);
        void nn_sgd(const NN& g, float rate);
        void nn_momentum(const NN& g, NN& v, float rate);
        void nn_adam(const NN& g, NN& m1, NN& m2, size_t t, float rate = {1e-3f});
        void nn_train(const Tensor& ti, const Tensor& to, size_t epochs, float rate, Optimizer opt);
        float nn_test(const Tensor& ti, const Tensor& to);
        void nn_infer(const Tensor& ti);
        void nn_print(const char* name);
    }; //struct nn
} //namespace gnn

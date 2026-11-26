#pragma once
#include "gmat.cuh"

namespace gnn
{
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
    }; //struct nn
} //namespace gnn

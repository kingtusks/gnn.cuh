#pragma once
#include "gmat.cuh"

namespace gnn
{
    struct NN
    {
        size_t count_ {};
        unsigned int batch_ {};
        std::vector<Tensor> w_, b_, a_;

        NN(const std::vector<unsigned int>& dims, unsigned int batch);

        Tensor& input() { return a_.front(); };
        Tensor& output() { return a_.back(); };

        void nn_rand();
        void nn_rand_he();
        void nn_fill(float n);
    }; //struct nn
} //namespace gnn

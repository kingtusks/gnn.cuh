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
    }; //struct nn
} //namespace gnn

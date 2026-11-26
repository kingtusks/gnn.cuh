#include "gnn.cuh"

namespace gnn
{
    NN::NN(const std::vector<unsigned int>& dims, unsigned int batch) :
        count_(dims.size() - 1), batch_(batch)
    {
        GNN_ASSERT(dims.size() >= 2);

        w_.reserve(count_);
        b_.reserve(count_);
        a_.reserve(count_ + 1);

        a_.push_back(Tensor({dims[0], batch}));
        for (size_t i {1}; i < dims.size(); ++i)
        {
            w_.push_back(Tensor({dims[i], dims[i - 1]}));
            b_.push_back(Tensor({dims[i], 1}));
            a_.push_back(Tensor({dims[i], batch}));
        }
    }

    void NN::nn_rand()
    {
        for (size_t i {}; i < count_; ++i)
        {
            float lim {sqrtf(6.0f / (float) w_[i].shape_[1])};
            w_[i].rand(-lim, lim);
            b_[i].fill(0.0f);
        }
    }

    void NN::nn_rand_he()
    {
        for (size_t i {}; i < count_; ++i)
        {
            a_[i + 1].dot(w_[i], a_[i]);
            a_[i + 1].sum_bias(b_[i]);

            if (i + 1 < count_) a_[i + 1].leaky_relu();
            else a_[i + 1].softmax();
        }
    }

    void NN::nn_fill(float n)
    {
        for (size_t i {}; i < count_; ++i)
        {
            w_[i].fill(n);
            b_[i].fill(n);
            a_[i].fill(n);
        }
        output().fill(n);
    }
} //namespace gnn

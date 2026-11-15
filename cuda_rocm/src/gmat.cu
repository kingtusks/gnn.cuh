#include "gmat.cuh"

namespace gnn
{
    // non-Tensor functions
    void cuda_check(cudaError_t err, const char* file, int line)
    {
        if (err != cudaSuccess)
        {
            printf("%s:%d CUDA Error: %s\n", file, line, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    void set_seed(std::uint64_t seed)
    {
        tensor_rand_seed = seed;
    }

    namespace kernels
    {

    } // namespace kernels

    // constructors

    Tensor::Tensor(std::initializer_list<unsigned int> dims) :
        dims_(0), count_(1), es_(nullptr)
    {
        GNN_ASSERT(dims.size() > 0 && dims.size() < max_dims);
        dims_ = dims.size();

        size_t i {};
        for (unsigned int dim : dims)
        {
            shape_[i++] = dim;
            count_ *= dim;
        }

        strides_[0] = shape_[0];
        for (size_t i {1}; i++ < dims_;)
            strides_[i] = shape_[i] * strides_[i - 1];

        CUDA_CHECK(cudaMalloc((void**)&es_, count_ * sizeof(float)));
    }

    // public impl

    void Tensor::print(const char* name) const
    {

    }
} // namespace gnn

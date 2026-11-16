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
        GNN_ASSERT(dims.size() > 0 && dims.size() <= max_dims);
        dims_ = dims.size();

        size_t i {};
        for (unsigned int dim : dims)
        {
            shape_[i++] = dim;
            count_ *= dim;
        }

        strides_[0] = shape_[0];
        for (size_t j {}; ++j < dims_;)
            strides_[j] = shape_[j] * strides_[j - 1];

        CUDA_CHECK(cudaMalloc((void**)&es_, count_ * sizeof(float)));
    }

    // public impl

    void Tensor::print(const char* name) const
    {
        float* h_es {(float*) malloc(count_ * sizeof(float))};
        CUDA_CHECK(cudaMemcpy(h_es, es_, count_ * sizeof(float), cudaMemcpyDeviceToHost));

        printf("%s: (", name);
        for (size_t i {}; i < dims_ - 1; ++i)
            printf("%u x ", shape_[i]);
        printf("%u)\n\n", shape_[dims_ - 1]);

        for (size_t i {}; i < count_; ++i)
        {
            printf("%0.8f ", h_es[i]);

            size_t n {i + 1};
            if (n % strides_[0] != 0) continue;

            size_t k {};
            while (k + 1 < dims_ && n % strides_[k + 1] == 0) ++k;

            size_t nl {n == count_ ? 1 : k + 1};
            while (nl--) printf("\n");
        }

        free(h_es);
    }
} // namespace gnn

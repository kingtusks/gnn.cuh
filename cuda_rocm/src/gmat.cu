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

    // constructors + ro5

    Tensor::~Tensor()
    {
        if (owns_) cudaFree(es_);
    }

    Tensor::Tensor(const Tensor& other) :
        dims_(other.dims_), count_(other.count_), es_(nullptr)
    {
        for (size_t i {}; i < max_dims; ++i)
        {
            shape_[i] = other.shape_[i];
            strides_[i] = other.strides_[i];
        }

        if (count_ > 0)
        {
            CUDA_CHECK(cudaMalloc((void**)&es_, count_ * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(es_, other.es_, count_ * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }

    Tensor::Tensor(Tensor&& other) noexcept :
        dims_(other.dims_), count_(other.count_), es_(other.es_), owns_(other.owns_)
    {
        for (size_t i {}; i < max_dims; ++i)
        {
            shape_[i] = other.shape_[i];
            strides_[i] = other.strides_[i];
        }
        other.es_ = nullptr; other.dims_ = 0; other.count_ = 0; other.owns_ = true;
    }

    Tensor& Tensor::operator=(const Tensor& other)
    {
        if (this != &other) *this = Tensor(other);
        return *this;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept
    {
        if (this != &other)
        {
            cudaFree(es_);

            dims_ = other.dims_;
            count_ = other.count_;
            es_ = other.es_;
            owns_ = other.owns_;

            for (size_t i {}; i < max_dims; ++i)
            {
                shape_[i] = other.shape_[i];
                strides_[i] = other.strides_[i];
            }

            other.es_ = nullptr; other.dims_ = 0; other.count_ = 0; other.owns_ = true;
        }

        return *this;
    }

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

        strides_[0] = 1;
        for (size_t j {}; ++j < dims_;)
            strides_[j] = strides_[j - 1] * shape_[j - 1];

        CUDA_CHECK(cudaMalloc((void**)&es_, count_ * sizeof(float)));
    }

    Tensor::Tensor(std::initializer_list<unsigned int> dims, std::initializer_list<float> data) :
        Tensor(dims)
    {
        GNN_ASSERT(data.size() == count_);
        CUDA_CHECK(cudaMemcpy(es_, data.begin(), count_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    Tensor::Tensor(std::initializer_list<unsigned int> dims, const float* host_data) :
        Tensor(dims)
    {
        CUDA_CHECK(cudaMemcpy(es_, host_data, count_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    // public impl

    bool Tensor::is_contiguous() const
    {
        size_t expected {1};
        for (size_t i {}; i < dims_; ++i)
        {
            if (shape_[i] > 1 && strides_[i] != expected) return false;
            expected *= shape_[i];
        }
        return true;
    }

    Tensor& Tensor::fill(float n)
    {

    }

    Tensor Tensor::slice(unsigned int dim, unsigned int start, unsigned int len) const
    {
        GNN_ASSERT(dim < dims_ && len > 0 && start + len <= shape_[dim]);

        Tensor t;
        t.owns_ = false;
        t.dims_ = dims_;
        t.es_ = es_ + start * strides_[dim];
        t.count_ = count_ / shape_[dim] * len;

        for (size_t i {}; i < dims_; ++i)
        {
            t.shape_[i] = shape_[i];
            t.strides_[i] = strides_[i];
        }

        t.shape_[dim] = len;
        return t;
    }

    void Tensor::info(const char* name) const
    {
        printf("%s\n", name);
        printf(" owns: %s\n", owns_ ? "true" : "false");
        printf(" dims: %u\n", dims_);
        printf(" count: %u\n", count_);

        printf(" shape: { ");
        for (size_t i {}; i < dims_; ++i)
            printf("%u ", shape_[i]);
        printf("}\n");

        printf(" strides: { ");
        for (size_t i {}; i < dims_; ++i)
            printf("%u ", shape_[i]);
        printf("}\n");
    }

    void Tensor::print(const char* name) const
    {
        if (count_ == 0)
        {
            printf("%s: empty\n", name);
            return;
        }

        size_t extent {1};
        for (size_t i {}; i < dims_; ++i)
            extent += (size_t) (shape_[i] - 1) * strides_[i];

        float* h_es {(float*) malloc(extent * sizeof(float))};
        CUDA_CHECK(cudaMemcpy(h_es, es_, extent * sizeof(float), cudaMemcpyDeviceToHost));

        printf("%s: (", name);
        for (size_t i {}; i < dims_ - 1; ++i)
            printf("%u x ", shape_[i]);
        printf("%u)\n\n", shape_[dims_ - 1]);

        for (size_t i {}; i < count_; ++i)
        {
            size_t idx {}, remainder {i};
            for (size_t j {}; j < dims_; ++j)
            {
                idx += (remainder % shape_[j]) * strides_[j];
                remainder /= shape_[j];
            }
            printf("%0.6f ", h_es[idx]);

            size_t block {1};
            for (size_t j {}; j < dims_; ++j)
            {
                block *= shape_[j];
                if ((i + 1) % block != 0) break;
                printf("\n");
            }
        }

        free(h_es);
    }
} // namespace gnn

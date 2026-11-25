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
        __global__ void fill(float* es, KernelTensorInfo kti, size_t count, float n)
        {
            size_t i {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            if (i >= count) return;

            size_t idx {};
            for (size_t j {}; j < kti.dims_; ++j)
            {
                idx += (i % kti.shape_[j]) * kti.strides_[j];
                i /= kti.shape_[j];
            }
            es[idx] = n;
        }

        __global__ void rand(float* es, KernelTensorInfo kti,
                size_t count, std::uint64_t seed, float low, float high)
        {
            size_t i {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t i2 {i};
            if (i >= count) return;

            size_t idx {};
            for (size_t j {}; j < kti.dims_; ++j)
            {
                idx += (i % kti.shape_[j]) * kti.strides_[j];
                i /= kti.shape_[j];
            }
            es[idx] = rand_float(seed, i2) * (high - low) + low;
        }

        __global__ void pad(float* es, float* src_es,
                KernelTensorInfo kti, KernelTensorInfo src_kti,
                size_t count, unsigned int p_size, float p_num)
        {
            size_t i {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            if (i >= count) return;

            size_t d_idx {}, src_idx {};
            bool inside {true};

            for (size_t j {}; j < kti.dims_; ++j)
            {
                unsigned int c {(unsigned int) (i % kti.shape_[j])};
                i /= kti.shape_[j];

                d_idx += (size_t) c * kti.strides_[j];

                unsigned int p_append {j < 2 ? p_size : 0};
                if (c < p_append || c >= p_append + src_kti.shape_[j])
                    inside = false;
                else
                    src_idx += (size_t) (c - p_append) * src_kti.strides_[j];
            }

            es[d_idx] = inside ? src_es[src_idx] : p_num;
        }

        __global__ void sgd(float* es, float* g_es, KernelTensorInfo kti,
                size_t count, float rate)
        {
            size_t i {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            if (i >= count) return;

            size_t idx {};
            for (size_t j {}; j < kti.dims_; ++j)
            {
                idx += (i % kti.shape_[j]) * kti.strides_[j];
                i /= kti.shape_[j];
            }
            es[idx] -= rate * g_es[idx];
        }

        __global__ void momentum(float* es, float* g_es, float* v_es,
                KernelTensorInfo kti, size_t count, float rate)
        {
            size_t i {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            if (i >= count) return;

            size_t idx {};
            for (size_t j {}; j < kti.dims_; ++j)
            {
                idx += (i % kti.shape_[j]) * kti.strides_[j];
                i /= kti.shape_[j];
            }
            float v {0.9f * v_es[idx] + g_es[idx]};
            v_es[idx] = v;
            es[idx] -= rate * v;
        }

        __global__ void adam(float* es, float* g_es, float* m1_es, float* m2_es,
                KernelTensorInfo kti, size_t count, float b1, float b2, float c1, float c2,
                float rate, float eps)
        {
            size_t i {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            if (i >= count) return;

            size_t idx {};
            for (size_t j {}; j < kti.dims_; ++j)
            {
                idx += (i % kti.shape_[j]) * kti.strides_[j];
                i /= kti.shape_[j];
            }
            float grad {g_es[idx]};
            float a {b1 * m1_es[idx] + (1.0f - b1) * grad};
            float b {b2 * m2_es[idx] + (1.0f - b2) * grad * grad};
            m1_es[idx] = a;
            m2_es[idx] = b;
            es[idx] -= rate * (a / c1) / (sqrtf(b / c2) + eps);
        }
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
            if (owns_) cudaFree(es_);

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

    void Tensor::same_structure(const Tensor& other) const
    {
        GNN_ASSERT(dims_ == other.dims_);
        for (size_t i {}; i < dims_; ++i)
        {
            GNN_ASSERT(shape_[i] == other.shape_[i]);
            GNN_ASSERT(strides_[i] == other.strides_[i]);
        }
    }

    Tensor& Tensor::fill(float n)
    {
        KernelTensorInfo kti {kernel_info()};

        unsigned int threads {256};
        unsigned int blocks {(unsigned int) (count_ + threads - 1) / threads};
        kernels::fill<<<blocks, threads>>>(es_, kti, count_, n);
        CUDA_CHECK(cudaGetLastError());
        return *this;
    }

    Tensor& Tensor::rand(float low, float high)
    {
        KernelTensorInfo kti {kernel_info()};

        std::uint64_t seed {++tensor_rand_seed};
        unsigned int threads {256};
        unsigned int blocks {(unsigned int) (count_ + threads - 1) / threads};
        kernels::rand<<<blocks, threads>>>(es_, kti, count_, seed, low, high);
        CUDA_CHECK(cudaGetLastError());
        return *this;
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

    Tensor& Tensor::pad(const Tensor& src, float p_num, unsigned int p_size)
    {
        GNN_ASSERT(dims_ == src.dims_ && dims_ >= 2);
        for (size_t j {}; j < dims_; ++j)
            GNN_ASSERT(shape_[j] == src.shape_[j] + (j < 2 ? 2 * p_size : 0));

        unsigned int threads {256};
        unsigned int blocks {(unsigned int) (count_ + threads - 1) / threads};
        kernels::pad<<<blocks, threads>>>(es_, src.es_, kernel_info(), src.kernel_info(), count_, p_size, p_num);
        CUDA_CHECK(cudaGetLastError());
        return *this;
    }

    Tensor& Tensor::sgd(const Tensor& g, float rate)
    {
        same_structure(g);

        unsigned int threads {256};
        unsigned int blocks {(unsigned int) (count_ + threads - 1) / threads};
        kernels::sgd<<<blocks, threads>>>(es_, g.es_, kernel_info(), count_, rate);
        CUDA_CHECK(cudaGetLastError());
        return *this;
    }

    Tensor& Tensor::momentum(const Tensor& g, Tensor& v, float rate)
    {
        same_structure(g);
        same_structure(v);

        unsigned int threads {256};
        unsigned int blocks {(unsigned int) (count_ + threads - 1) / threads};
        kernels::momentum<<<blocks, threads>>>(es_, g.es_, v.es_, kernel_info(), count_, rate);
        CUDA_CHECK(cudaGetLastError());
        return *this;
    }

    Tensor& adam(const Tensor& g, Tensor& m1, Tensor& m2, size_t t, float b1, float b2, float rate, float eps)
    {
        GNN_ASSERT(t > 0);
        same_structure(g);
        same_structure(m1);
        same_structure(m2);

        float c1 {1.0f - powf(b1, (float) t)};
        float c2 {1.0f - powf(b2, (float) t)};

        unsigned int threads {256};
        unsigned int blocks {(unsigned int) (count_ + threads - 1) / threads};
        kernels::adam<<<blocks, threads>>>(es_, g.es_, m1.es_, m2.es_, kernel_info(),
                count_, b1, b2, c1, c2, rate, eps);
        CUDA_CHECK(cudaGetLastError());
        return *this;
    }

    KernelTensorInfo Tensor::kernel_info() const
    {
        KernelTensorInfo kti {};
        kti.dims_ = dims_;
        for (size_t i {}; i < dims_; ++i)
        {
            kti.shape_[i] = shape_[i];
            kti.strides_[i] = strides_[i];
        }
        return kti;
    }

    void Tensor::info(const char* name) const
    {
        printf("%s\n", name);
        printf(" owns: %s\n", owns_ ? "true" : "false");
        printf(" dims: %u\n", dims_);
        printf(" count: %zu\n", count_);

        printf(" shape: { ");
        for (size_t i {}; i < dims_; ++i)
            printf("%u ", shape_[i]);
        printf("}\n");

        printf(" strides: { ");
        for (size_t i {}; i < dims_; ++i)
            printf("%u ", strides_[i]);
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

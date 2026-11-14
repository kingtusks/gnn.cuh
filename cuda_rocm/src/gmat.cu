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
        __global__ void fill(Tensor t, float n)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < t.rows_ && j < t.cols_ && k < t.depth_)
                t.at(i, j, k) = n;
        }

        __global__ void pad_inject(Tensor dst, const Tensor t, size_t p_size)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < t.rows_ && j < t.cols_ && k < t.depth_)
                dst.at(i + p_size, j + p_size, k) = t.at(i, j, k);
        }

        __global__ void pad_fill(Tensor dst, float p_num, size_t p_size)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < dst.rows_ && j < dst.cols_ && dst.depth_)
                if (i < p_size || j < p_size || (i >= dst.rows_ - p_size && i < dst.rows_) || (j >= dst.cols_ - p_size && j < dst.cols_))
                    dst.at(i, j, k) = p_num;
        }

        __global__ void rand(Tensor t, float low, float high, std::uint64_t seed)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < t.rows_ && j < t.cols_ && k < t.depth_)
            {
                std::uint64_t true_i {k * t.dstride_ + i * t.stride_ + j};
                t.at(i, j, k) = rand_float(seed, true_i) * (high - low) + low;
            }
        }

        __global__ void sgd(Tensor t, const Tensor g, float rate)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < t.rows_ && j < t.cols_ && k < t.depth_)
                t.at(i, j, k) -= rate * g.at(i, j, k);
        }

        __global__ void momentum(Tensor t, const Tensor g, Tensor v, float rate)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < t.rows_ && j < t.cols_ && k < t.depth_)
            {
                v.at(i, j, k) = 0.9f * v.at(i, j, k) + g.at(i, j, k);
                t.at(i, j, k) -= rate * v.at(i, j, k);
            }
        }

        __global__ void adam(Tensor t, const Tensor g, Tensor m1, Tensor m2,
                float c1, float c2, size_t t, float rate)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < t.rows_ && j < t.cols_ && k < t.depth_)
            {
                float g {g.at(i, j, k)};
                float a {0.9f * m1.at(i, j, k) + 0.1f * g};
                float b {0.999f * m2.at(i, j, k) + 0.001f * g * g};
                m1.at(i, j, k) = a;
                m2.at(i, j, k) = b;
                t.at(i, j, k) -= rate * (a / c1) / (sqrtf(b / c2) + 1e-8f);
            }
        }
    } // namespace kernels

    // constructors

    Tensor::Tensor(size_t rows, size_t cols, size_t depth) :
        rows_(rows), cols_(cols), depth_(depth), stride_(cols), dstride_(rows * cols), es_(nullptr)
    {
        CUDA_CHECK(cudaMalloc((void**)&es_, rows * cols * depth * sizeof(float)));
    }

    Tensor::Tensor(size_t rows, size_t cols, size_t depth, size_t stride, size_t dstride, const float *h_es) :
        Tensor(rows, cols, depth)
    {
        for (size_t i {}; i < depth; ++i)
        {
            CUDA_CHECK(cudaMemcpy2D(
                es_ + i * dstride_, stride_ * sizeof(float),
                h_es + i * dstride, stride * sizeof(float),
                cols * sizeof(float),
                rows, cudaMemcpyHostToDevice
            ));
        }
    }

    // public impl

    Tensor Tensor::clone() const
    {
        Tensor t(rows_, cols_, depth_);
        for (size_t i {}; i < depth_; ++i)
        {
            CUDA_CHECK(cudaMemcpy2D(
                t.es_ + i * t.dstride_, t.stride_ * sizeof(float),
                es_ + i * dstride_, stride_ * sizeof(float),
                cols_ * sizeof(float),
                rows_, cudaMemcpyDeviceToDevice
            ));
        }
        return t;
    }

    void Tensor::pad(Tensor dst, float p_num, size_t p_size)
    {
        GNN_ASSERT(dst.rows_ == 2 * p_size + rows_);
        GNN_ASSERT(dst.cols_ == 2 * p_size + cols_);
        GNN_ASSERT(dst.depth_ == depth_);

        dim3 threads {32, 8};
        dim3 i_blocks
        {
            (unsigned int) (cols_ + threads.x - 1) / threads.x,
            (unsigned int) (rows_ + threads.y - 1) / threads.y,
            depth_
        };
        kernels::pad_inject<<<i_blocks, threads>>>(dst, *this, p_size);
        CUDA_CHECK(cudaGetLastError());

        dim3 f_blocks
        {
            (unsigned int) (dst.cols_ + threads.x - 1) / threads.x,
            (unsigned int) (dst.rows_ + threads.y - 1) / threads.y,
            dst.depth_
        };
        kernels::pad_fill<<<f_blocks, threads>>>(dst, p_num, p_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void Tensor::fill(float n)
    {
        dim3 threads {32, 8};
        dim3 blocks
        {
            (unsigned int) (cols_ + threads.x - 1) / threads.x,
            (unsigned int) (rows_ + threads.y - 1) / threads.y,
            depth_
        };

        kernels::fill<<<blocks, threads>>>(*this, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void Tensor::rand(float low, float high)
    {
        std::uint64_t seed {++tensor_rand_seed};
        dim3 threads {32, 8};
        dim3 blocks
        {
            (unsigned int) (cols_ + threads.x - 1) / threads.x,
            (unsigned int) (rows_ + threads.y - 1) / threads.y,
            depth_
        };
        kernels::rand<<<blocks, threads>>>(*this, low, high, seed);
        CUDA_CHECK(cudaGetLastError());
    }

    Tensor Tensor::mat(size_t depth)
    {
        return mats(depth, 1);
    }

    Tensor Tensor::mats(size_t depth, size_t n)
    {
        Tensor t;
        t.rows_ = rows_;
        t.cols_ = cols_;
        t.depth_ = n;
        t.stride_ = stride_;
        t.dstride_ = dstride_;
        t.es_ = &at(0, 0, depth);
        return t;
    }

    Tensor Tensor::row(size_t depth, size_t row)
    {
        return rows(row, 1, depth);
    }

    Tensor Tensor::rows(size_t depth, size_t row, size_t n)
    {
        Tensor t;
        t.rows_ = n;
        t.cols_ = cols_;
        t.depth_ = 1;
        t.stride_ = stride_;
        t.dstride_ = dstride_;
        t.es_ = &at(row, 0, depth);
        return t;
    }

    void Tensor::sgd(const Tensor g, float rate)
    {
        GNN_ASSERT(rows_ == g.rows_);
        GNN_ASSERT(cols_ == g.cols_);

        dim3 threads {32, 8};
        dim3 blocks
        {
            (unsigned int) (cols_ + threads.x - 1) / threads.x,
            (unsigned int) (rows_ + threads.x - 1) / threads.x,
            depth_
        };
        kernels::sgd<<<blocks, threads>>>(*this, g, rate);
        CUDA_CHECK(cudaGetLastError());
    }

    void Tensor::momentum(const Tensor g, Tensor v, float rate)
    {
        GNN_ASSERT(rows_ == g.rows_);
        GNN_ASSERT(cols_ == g.cols_);
        GNN_ASSERT(rows_ == v.rows_);
        GNN_ASSERT(cols_ == v.cols_);

        dim3 threads {32, 8};
        dim3 blocks
        {
            (unsigned int) (cols_ + threads.x - 1) / threads.x,
            (unsigned int) (rows_ + threads.x - 1) / threads.x,
            depth_
        };
        kernels::momentum<<<blocks, threads>>>(*this, g, v, rate);
        CUDA_CHECK(cudaGetLastError());
    }

    void Tensor::adam(const Tensor g, Tensor m1, Tensor m2, size_t t, float rate)
    {
        GNN_ASSERT(t > 0);
        GNN_ASSERT(rows_ == g.rows && cols_ == g.cols);
        GNN_ASSERT(rows_ == m1.rows && cols_ == m1.cols);
        GNN_ASSERT(rows_ == m2.rows && cols_ == m2.cols);

        float c1 {1.0f - powf(0.9f, t)};
        float c1 {1.0f - powf(0.999f, t)};

        dim3 threads {32, 8};
        dim3 blocks
        {
            (unsigned int) (cols_ + threads.x - 1) / threads.x,
            (unsigned int) (rows_ + threads.y - 1) / threads.y,
            depth_
        };
        kernels::adam<<<blocks, threads>>>(*this, g, m1, m2, c1, c2, t, rate);
        CUDA_CHECK(cudaGetLastError());
    }

    void Tensor::print(const char* name) const
    {
        size_t count {(size_t) (depth_ - 1) * dstride_ + (size_t) rows_ * stride_};
        float* h_es {(float*) malloc(count * sizeof(float))};
        CUDA_CHECK(cudaMemcpy(h_es, es_, count * sizeof(float), cudaMemcpyDeviceToHost));

        printf("%s (%u x %u x %u)\n", name, rows_, cols_, depth_);
        for (size_t d {}; d < depth_; ++d) {
            printf("  [depth %zu]\n", d);
            for (size_t i {}; i < rows_; ++i) {
                for (size_t j {}; j < cols_; ++j)
                    printf("    %f ", h_es[d * dstride_ + i * stride_ + j]);
                printf("\n");
            }
        }
        printf("\n");
        free(h_es);
    }
} // namespace gnn

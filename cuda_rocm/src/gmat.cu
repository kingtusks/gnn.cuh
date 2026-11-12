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

            if (i < t.rows && j < t.cols && k < t.depth)
            {
                t.at(i, j, k) = n;
            }
        }

        __global__ void pad_inject(Tensor dst, const Tensor t, size_t p_size)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < t.rows && j < t.cols && k < t.depth)
            {
                dst.at(i + p_size, j + p_size, k) = t.at(i, j, k);
            }
        }

        __global__ void pad_fill(Tensor dst, float p_num, size_t p_size)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < dst.rows && j < dst.cols && dst.depth)
                if (i < p_size || j < p_size || (i >= dst.rows - p_size && i < dst.rows) || (j >= dst.cols - p_size && j < dst.cols))
                    dst.at(i, j, k) = p_num;
        }

        __global__ void rand(Tensor t, float low, float high, std::uint64_t seed)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < t.rows && j < t.cols && k < t.depth)
            {
                std::uint64_t true_i {k * t.dstride + i * t.stride + j};
                t.at(i, j, k) = rand_float(seed, true_i) * (high - low) + low;
            }
        }
    } // namespace kernels

    // constructors
    Tensor::Tensor(size_t rows, size_t cols, size_t depth) :
        rows(rows), cols(cols), depth(depth), stride(cols), dstride(rows * cols), es(nullptr)
    {
        CUDA_CHECK(cudaMalloc((void**)&es, rows * cols * depth * sizeof(float)));
    }

    Tensor::Tensor(size_t rows, size_t cols, size_t depth, size_t stride, size_t dstride, const float *h_es) :
        Tensor(rows, cols, depth)
    {
        for (size_t d {}; d < depth; ++d)
        {
            CUDA_CHECK(cudaMemcpy2D(
                es + d * this->dstride, this->stride * sizeof(float),
                h_es + d * dstride, stride * sizeof(float),
                cols * sizeof(float),
                rows, cudaMemcpyHostToDevice
            ));
        }
    }

    // public impl

    void Tensor::pad(Tensor dst, float p_num, size_t p_size)
    {
        GNN_ASSERT(dst.rows == 2 * p_size + rows);
        GNN_ASSERT(dst.cols == 2 * p_size + cols);
        GNN_ASSERT(dst.depth == depth);

        dim3 threads {32, 8};
        dim3 i_blocks
        {
            (cols + threads.x - 1) / threads.x,
            (rows + threads.y - 1) / threads.y,
            depth
        };
        kernels::pad_inject<<<i_blocks, threads>>>(dst, *this, p_size);
        CUDA_CHECK(cudaGetLastError());

        dim3 f_blocks
        {
            (dst.cols + threads.x - 1) / threads.x,
            (dst.rows + threads.y - 1) / threads.y,
            dst.depth
        };
        kernels::pad_fill<<<f_blocks, threads>>>(dst, p_num, p_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void Tensor::fill(float n)
    {
        dim3 threads {32, 8};
        dim3 blocks
        {
            (cols + threads.x - 1) / threads.x,
            (rows + threads.y - 1) / threads.y,
            depth
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
            (cols + threads.x - 1) / threads.x,
            (rows + threads.y - 1) / threads.y,
            depth
        };
        kernels::rand<<<blocks, threads>>>(*this, low, high, seed);
        CUDA_CHECK(cudaGetLastError());
    }

    void Tensor::print(const char *name) const
    {
        size_t count {(size_t) (depth - 1) * dstride + (size_t) rows * stride};
        float* h_es {(float*) malloc(count * sizeof(float))};
        CUDA_CHECK(cudaMemcpy(h_es, es, count * sizeof(float), cudaMemcpyDeviceToHost));

        printf("%s (%u x %u x %u)\n", name, rows, cols, depth);
        for (size_t d {}; d < depth; ++d) {
            printf("  [depth %zu]\n", d);
            for (size_t i {}; i < rows; ++i) {
                for (size_t j {}; j < cols; ++j)
                    printf("    %f ", h_es[d * dstride + i * stride + j]);
                printf("\n");
            }
        }
        printf("\n");
        free(h_es);
    }
} // namespace gnn

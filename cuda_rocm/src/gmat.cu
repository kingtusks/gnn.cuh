#include "gmat.cuh"

namespace gnn
{
    namespace kernels
    {
        __global__ void tensor_fill(Tensor t, float n)
        {
            size_t i {(size_t) blockIdx.y * blockDim.y + threadIdx.y};
            size_t j {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            size_t k {(size_t) blockIdx.z};

            if (i < t.rows && j < t.cols && k < t.depth) t.at(i, j, k) = n;
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
    void Tensor::tensor_fill(float n)
    {
        dim3 threads {32, 8};
        dim3 blocks
        {
            (cols + threads.x - 1) / threads.x,
            (rows + threads.y - 1) / threads.y,
            depth
        };

        kernels::tensor_fill<<<blocks, threads>>>(*this, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void Tensor::tensor_print(const char *name) const
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

    // non-Tensor functions
    void cuda_check(cudaError_t err, const char* file, int line)
    {
        if (err != cudaSuccess)
        {
            printf("%s:%d CUDA Error: %s\n", file, line, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
} // namespace gnn

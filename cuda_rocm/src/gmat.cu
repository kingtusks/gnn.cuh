#include "gmat.cuh"

namespace gnn
{
    namespace kernels
    {
        __global__ void tensor_fill_contiguous(float* p, float n, size_t area)
        {

        }

        __global__ void tensor_fill_noncontiguous(Tensor t, float n)
        {

        }
    }

    struct Tensor
    {
        Tensor(size_t rows, size_t cols, size_t depth) :
            rows(rows), cols(cols), depth(depth), stride(cols), dstride(rows * cols), es(nullptr)
        {
            CUDA_CHECK(cudaMalloc((void**)&es, rows * cols * depth * sizeof(float)));
        }

        Tensor(size_t rows, size_t cols, size_t depth, size_t stride, size_t dstride, const float *h_es) :
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

        void tensor_fill(float n)
        {
            GNN_ASSERT(depth == 1);
            (stride == cols) ? tensor_fill_contiguous(n) : tensor_fill_noncontiguous(n)
        }

        private:
            void tensor_fill_contiguous(float n)
            {

            }

            void tensor_fill_noncontiguous(float n)
            {

            }
    };

    void cuda_check(cudaError_t err, const char* file, int line)
    {
        if (err != cudaSuccess)
        {
            printf("%s:%d CUDA Error: %s\n", file, line, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
}

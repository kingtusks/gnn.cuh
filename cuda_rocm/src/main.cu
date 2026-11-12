#include "gmat.cuh"

int main() {
    size_t p_size {2};
    gnn::Tensor t {3, 4, 2};
    gnn::Tensor dst {t.rows + 2 * p_size, t.cols + 2 * p_size, 2};

    t.fill(3.0f);
    TENSOR_PRINT(t);

    t.pad(dst, 1.0f, p_size);
    TENSOR_PRINT(t);
    TENSOR_PRINT(dst);

    // destructor wont work for individual matrices (bc of the kernels)
    CUDA_CHECK(cudaFree(t.es));
    CUDA_CHECK(cudaFree(dst.es));
    return 0;
}

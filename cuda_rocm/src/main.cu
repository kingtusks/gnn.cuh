#include "gmat.cuh"

int main() {
    gnn::Tensor t {3, 4, 2};
    gnn::Tensor g {t.clone()};
    float s {0.2f};

    t.fill(1.0f);
    g.fill(2.0f);

    TENSOR_PRINT(t);
    t.sgd(g, s);
    TENSOR_PRINT(t);

    // destructor wont work for individual matrices (bc of the kernels)
    CUDA_CHECK(cudaFree(t.es_));
    CUDA_CHECK(cudaFree(g.es_));
    return 0;
}

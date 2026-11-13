#include "gmat.cuh"

int main() {
    gnn::Tensor t {3, 4, 2};
    gnn::Tensor g {t.clone()};
    gnn::Tensor v {t.clone()};
    float rate {2e-1};

    t.fill(1.0f);
    g.fill(2.0f);
    v.fill(3.0f);

    TENSOR_PRINT(t);
    t.momentum(g, v, rate);
    TENSOR_PRINT(t);

    // destructor wont work for individual matrices (bc of the kernels)
    CUDA_CHECK(cudaFree(t.es_));
    CUDA_CHECK(cudaFree(g.es_));
    CUDA_CHECK(cudaFree(v.es_));
    return 0;
}

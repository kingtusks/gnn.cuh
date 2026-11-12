#include "gmat.cuh"

int main() {
    gnn::Tensor t {3, 4, 2};
    t.tensor_fill(3.0f);
    TENSOR_PRINT(t);
    CUDA_CHECK(cudaFree(t.es)); //need a destructor asap
    return 0;
}

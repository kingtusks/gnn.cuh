#include "gmat.cuh"

int main() {
    gnn::Tensor t {3, 2, 4, 2};
    TENSOR_PRINT(t);
    return 0;
}

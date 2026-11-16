#include "gmat.cuh"

int main() {
    gnn::Tensor t {2, 3, 5, 2};
    TENSOR_PRINT(t);
    return 0;
}

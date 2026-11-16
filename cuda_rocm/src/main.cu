#include "gmat.cuh"

int main() {
    gnn::Tensor t {3, 2, 4, 2};
    gnn::Tensor t2(t);
    TENSOR_PRINT(t2);

    return 0;
}

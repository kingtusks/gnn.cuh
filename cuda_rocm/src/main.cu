#include "gmat.cuh"

int main() {
    gnn::Tensor td(
        {3, 4},
        {0, 0, 0,
         0, 1, 1,
         1, 0, 1,
         1, 1, 0}
    );

    TENSOR_PRINT(td);

    gnn::Tensor ti = td.slice(1, 0, 2);
    TENSOR_PRINT(ti);
    return 0;
}

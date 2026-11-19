#include "gmat.cuh"

int main() {
    gnn::Tensor td(
        {3, 4},
        {0, 0, 0,
         0, 1, 1,
         1, 0, 1,
         1, 1, 0}
    );

    gnn::Tensor ti = td.slice(0, 0, 2);
    ti.fill(2.0f);
    TENSOR_PRINT(ti);
    TENSOR_PRINT(td);
    return 0;
}

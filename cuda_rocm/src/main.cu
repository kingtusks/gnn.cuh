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
    ti.rand();
    TENSOR_PRINT(ti);
    return 0;
}

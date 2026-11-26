#include "gnn.cuh"

int main() {
    gnn::Tensor t({3, 4, 2, 2});
    gnn::Tensor u({5, 6, 2, 2});

    t.fill(1.0f);
    u.pad(t);
    TENSOR_PRINT(u);
    return 0;
}

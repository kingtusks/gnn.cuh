#define GNN_IMPLEMENTATION
#include "gnn.cuh"

int main() {
    // Mat m = mat_alloc(3, 3);
    // mat_rand(m, 5, 10);
    // MAT_PRINT(m);

    size_t dim[] = {2, 2, 1};
    NN nn = nn_alloc(dim, 3);
    // NN_PRINT(nn);
    return 0;
}

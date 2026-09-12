#define GNN_IMPLEMENTATION
#include "gnn.cuh"

int main() {
    Mat m = mat_alloc(3, 3);
    mat_fill(m, 2);
    MAT_PRINT(m);

    mat_sig(m);
    MAT_PRINT(m);
    mat_fill(m, 2);

    mat_tanh(m);
    MAT_PRINT(m);
    mat_fill(m, 2);

    mat_relu(m);
    MAT_PRINT(m);
}

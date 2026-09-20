#define GNN_IMPLEMENTATION
#include "gnn.cuh"

int main() {
    Mat m = mat_alloc(3, 3);
    mat_fill(m, 2);
    Mat b = mat_alloc(3, 3);

    mat_copy(b, m);

    MAT_PRINT(m);
    MAT_PRINT(b);
}

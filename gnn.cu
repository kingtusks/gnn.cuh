#define GNN_IMPLEMENTATION
#include "gnn.cuh"

int main() {
    Mat m = mat_alloc(3, 3);
    mat_fill(m, 2);
    Mat b = mat_alloc(3, 3);
    mat_fill(b, 3);

    mat_sum(b, m);

    MAT_PRINT(b);
    MAT_PRINT(m);
}

#define GNN_IMPLEMENTATION
#include "gnn.cuh"

int main() {
    Mat m = mat_alloc(3, 3);
    mat_fill(m, 2);
    MAT_PRINT(m);

    Mat dst = mat_alloc(3, 3);
    mat_copy(dst, m);
    MAT_PRINT(dst);
}

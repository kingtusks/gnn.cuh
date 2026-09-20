#define GNN_IMPLEMENTATION
#include "gnn.cuh"

int main() {
    Mat dst = mat_alloc(2, 2);
    Mat a = mat_alloc(2, 3);
    mat_fill(a, 2);
    Mat b = mat_alloc(3, 2);
    mat_fill(b, 2);
    mat_dot(dst, a, b);

    MAT_PRINT(dst);

    return 0;
}

#define GNN_IMPLEMENTATION
#include "gnn.cuh"

float d[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

int main() {
    Mat m = mat_alloc(3, 3);

    MAT_PRINT(m);
    MatFill(&m, 1);
    MAT_PRINT(m);
}

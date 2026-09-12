#define GNN_IMPLEMENTATION
#include "gnn.cuh"

float d[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

int main() {
    Mat m = {
        .rows = 3,
        .cols = 3,
        .stride = 3,
        .es = d,
    };

    MAT_PRINT(m);
    MatFill(&m, 1);
    MAT_PRINT(m);
}

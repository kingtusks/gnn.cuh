#define GNN_IMPLEMENTATION
#include "gnn.cuh"

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

int main() {
    size_t stride = 3;
    size_t n = sizeof(td) / sizeof(td[0]) / stride;

    Mat ti = mat_alloc_from(n, 2, stride, td);
    Mat to = mat_alloc_from(n, 1, stride, td + 2);

    size_t dim[] = {2, 2, 1};
    NN nn = nn_alloc(dim, 3);

    nn_fill(nn, 1);
    NN_PRINT(nn);
    float cost = nn_cost(nn, ti, to);
    printf("%f\n", cost);

    return 0;
}

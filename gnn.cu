#define GNN_IMPLEMENTATION
#include "gnn.cuh"

#define RATE 1
#define ITER 100

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
    NN g = nn_alloc(dim, 3);

    nn_rand(nn, 0, 1);
    nn_train(nn, g, ti, to, RATE, ITER);

    // for (size_t i = 0; i < 2; ++i) {
    //     for (size_t j = 0; j < 2; ++j) {
    //         MAT_AT(NN_INPUT(nn), 0, 0) = i;
    //         MAT_AT(NN_INPUT(nn), 0, 1) = j;
    //         nn_forward(nn);
    //         float y = *NN_OUTPUT(nn).es;
    //         printf("%zu ^ %zu = %f\n", i, j, y);
    //     }
    // }

    return 0;
}

#define GNN_IMPLEMENTATION
#include "gnn.cuh"
// #include "time.h"

#define RATE 1
#define ITER 10000

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

int main() {
    // mat_set_seed((uint64_t) time(NULL));
    size_t stride = 3;
    size_t n = sizeof(td) / sizeof(td[0]) / stride;

    Mat ti = mat_alloc_from(n, 2, stride, td);
    Mat to = mat_alloc_from(n, 1, stride, td + 2);

    size_t dim[] = {2, 2, 1};
    // size_t dim[] = {784, 128, 64, 10};
    NN nn = nn_alloc(dim, ARRAY_LEN(dim));
    NN g = nn_alloc(dim, ARRAY_LEN(dim));

    nn_rand(nn, -1, 1);
    nn_train(nn, g, ti, to, RATE, ITER);

    size_t sizeof_a = sizeof(Mat) * (nn.count + 1);
    Mat* ha = (Mat*) malloc(sizeof_a);
    GNN_ASSERT(ha);
    CUDA_CHECK(cudaMemcpy(ha, nn.a, sizeof_a, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float in0 = (float) i;
            float in1 = (float) j;
            CUDA_CHECK(cudaMemcpy(&MAT_AT(ha[0], 0, 0), &in0, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(&MAT_AT(ha[0], 0, 1), &in1, sizeof(float), cudaMemcpyHostToDevice));
            nn_forward(nn);
            float y;
            CUDA_CHECK(cudaMemcpy(&y, ha[nn.count].es, sizeof(float), cudaMemcpyDeviceToHost));
            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }

    // NN_PRINT(nn);

    free(ha);
    return 0;
}

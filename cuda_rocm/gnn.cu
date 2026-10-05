#define GNN_IMPLEMENTATION
#include "gnn.cuh"
#define MNIST_IMPLEMENTATION
#include "../mnist/mnist.h"

#define RATE 1
#define EPOCHS 1000

float xor_td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

void xor_nn() {
    size_t stride = 3;
    size_t n = sizeof(xor_td) / sizeof(xor_td[0]) / stride;

    Mat ti = mat_alloc_from(n, 2, stride, xor_td);
    Mat to = mat_alloc_from(n, 1, stride, xor_td + 2);

    size_t dim[] = {2, 2, 1};
    NN nn = nn_alloc(dim, ARRAY_LEN(dim), 2);
    NN g = nn_alloc(dim, ARRAY_LEN(dim), 2);

    nn_train(nn, g, ti, to, RATE, EPOCHS);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float in[2] = {(float) i, (float) j};
            CUDA_CHECK(cudaMemcpy(NN_INPUT(nn).es, in, sizeof(in), cudaMemcpyHostToDevice));
            nn_forward(nn);
            float y;
            CUDA_CHECK(cudaMemcpy(&y, NN_OUTPUT(nn).es, sizeof(float), cudaMemcpyDeviceToHost));
            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }
}

void mnist_nn() {
    std::vector<float> mnist_td = mnist_load_td("../mnist/data/train-images.idx3-ubyte", "../mnist/data/train-labels.idx1-ubyte", 60000);
    std::vector<float> mnist_ttd = mnist_load_td("../mnist/data/t10k-images.idx3-ubyte", "../mnist/data/t10k-labels.idx1-ubyte", 10000);
    GNN_ASSERT(!mnist_td.empty() && !mnist_ttd.empty());

    Mat ti = mat_alloc_from(60000, MNIST_IMG_SIZE, MNIST_STRIDE, mnist_td.data());
    Mat to = mat_alloc_from(60000, MNIST_N_CLASSES, MNIST_STRIDE, mnist_td.data() + MNIST_IMG_SIZE);
    Mat tti = mat_alloc_from(10000, MNIST_IMG_SIZE, MNIST_STRIDE, mnist_ttd.data());
    Mat tto = mat_alloc_from(10000, MNIST_N_CLASSES, MNIST_STRIDE, mnist_ttd.data() + MNIST_IMG_SIZE);

    size_t batch = 50;
    size_t dim[] = {784, 128, 64, 10};
    NN nn = nn_alloc(dim, ARRAY_LEN(dim), batch);
    NN g = nn_alloc(dim, ARRAY_LEN(dim), batch);

    nn_train(nn, g, ti, to, RATE, EPOCHS);
    nn_test(nn, tti, tto);
}

int main() {
#if 0
    xor_nn();
#else
    mnist_nn();
#endif
    return 0;
}

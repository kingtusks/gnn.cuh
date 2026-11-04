#include "gnn.cuh"
#include "../../datasets/mnist/mnist.h"

float xor_td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

void xor_nn() {
    size_t stride = 3;
    size_t n = sizeof(xor_td) / sizeof(xor_td[0]) / stride;

    Matrix::Mat ti = Matrix::mat_alloc_from(n, 2, stride, xor_td);
    Matrix::Mat to = Matrix::mat_alloc_from(n, 1, stride, xor_td + 2);

    size_t dim[] = {2, 2, 1};
    NeuralNetwork::NN nn{NeuralNetwork::nn_alloc(dim, ARRAY_LEN(dim), 2)};
    NeuralNetwork::NN g{NeuralNetwork::nn_alloc(dim, ARRAY_LEN(dim), 2)};
    NeuralNetwork::NN v{NeuralNetwork::nn_alloc(dim, ARRAY_LEN(dim), 2)};

    NeuralNetwork::nn_rand_he(nn);
    NeuralNetwork::nn_fill(v, 0);
    // nn_train(nn, g, v, ti, to, EPOCHS);

    for (size_t i{}; i < 2; ++i) {
        for (size_t j{}; j < 2; ++j) {
            float in[2] = {(float) i, (float) j};
            GNN_CUDA_CHECK(cudaMemcpy(NN_INPUT(nn).es, in, sizeof(in), cudaMemcpyHostToDevice));
            NeuralNetwork::nn_forward(nn);
            float y;
            GNN_CUDA_CHECK(cudaMemcpy(&y, NN_OUTPUT(nn).es, sizeof(float), cudaMemcpyDeviceToHost));
            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }

    NeuralNetwork::nn_free(nn); NeuralNetwork::nn_free(g); NeuralNetwork::nn_free(v);
}

void mnist_nn() {
    std::vector<float> mnist_td {
        MNIST::mnist_load_td(
            "../datasets/mnist/data/train-images.idx3-ubyte",
            "../datasets/mnist/data/train-labels.idx1-ubyte",
            60000
        )
    };

    std::vector<float> mnist_ttd {
        MNIST::mnist_load_td(
            "../datasets/mnist/data/t10k-images.idx3-ubyte",
            "../datasets/mnist/data/t10k-labels.idx1-ubyte",
            10000)
    };

    GNN_ASSERT(!mnist_td.empty() && !mnist_ttd.empty());

    Matrix::Mat ti {
        Matrix::mat_alloc_from(
            60000,
            MNIST::MNIST_IMG_SIZE,
            MNIST::MNIST_STRIDE,
            mnist_td.data()
        )
    };

    Matrix::Mat to {
        Matrix::mat_alloc_from(
            60000,
            MNIST::MNIST_IMG_SIZE,
            MNIST::MNIST_STRIDE,
            mnist_td.data() + MNIST::MNIST_IMG_SIZE
        )
    };

    Matrix::Mat tti {
        Matrix::mat_alloc_from(
            60000,
            MNIST::MNIST_IMG_SIZE,
            MNIST::MNIST_STRIDE,
            mnist_ttd.data()
        )
    };

    Matrix::Mat tto {
        Matrix::mat_alloc_from(
            60000,
            MNIST::MNIST_IMG_SIZE,
            MNIST::MNIST_STRIDE,
            mnist_ttd.data() + MNIST::MNIST_IMG_SIZE
        )
    };

    size_t batch{100};
    size_t epochs{100};
    float rate{1e-1};
    size_t dim[]{784, 128, 64, 10};

    NeuralNetwork::NN nn {
        NeuralNetwork::nn_alloc(dim, ARRAY_LEN(dim), batch)
    };

    NeuralNetwork::nn_train(nn, ti, to, epochs, rate, NeuralNetwork::Optimizer::SGD);
    NeuralNetwork::nn_test(nn, tti, tto);
    NeuralNetwork::nn_free(nn);
}

//1 for 10, 2 for 100
void cifar_x(int lb) {}

int main() {
    return 0;
}

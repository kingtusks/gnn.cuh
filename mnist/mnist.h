#pragma once

#include <cstdio>
#include <fstream>
#include <vector>

#define MNIST_IMG_SIZE 784
#define MNIST_N_CLASSES 10
#define MNIST_STRIDE (MNIST_IMG_SIZE + MNIST_N_CLASSES)

inline std::vector<float> mnist_load_td(const char* images_path, const char* labels_path, size_t n);

#ifdef MNIST_IMPLEMENTATION

inline std::vector<float> mnist_load_td(const char* images_path, const char* labels_path, size_t n) {
    std::ifstream images_file(images_path, std::ios::binary);
    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!images_file || !labels_file) {
        fprintf(stderr, "mnist: could not open %s or %s\n", images_path, labels_path);
        return {};
    }

    images_file.seekg(16);
    labels_file.seekg(8);

    std::vector<unsigned char> raw_labels(n);
    labels_file.read(reinterpret_cast<char*>(raw_labels.data()), n);

    std::vector<float> td(n * MNIST_STRIDE, 0.f);
    std::vector<unsigned char> raw(MNIST_IMG_SIZE);

    for (size_t i = 0; i < n; ++i) {
        images_file.read(reinterpret_cast<char*>(raw.data()), MNIST_IMG_SIZE);
        float* row = &td[i * MNIST_STRIDE];
        for (size_t j = 0; j < MNIST_IMG_SIZE; ++j)
            row[j] = raw[j] / 255.f;
        row[MNIST_IMG_SIZE + raw_labels[i]] = 1.f;
    }
    return td;
}

#endif //MNIST_IMPLEMENTATION

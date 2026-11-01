#pragma once
#include <vector>

#define MNIST_IMG_SIZE 784
#define MNIST_N_CLASSES 10
#define MNIST_STRIDE (MNIST_IMG_SIZE + MNIST_N_CLASSES)

inline std::vector<float> mnist_load_td(const char* images_path, const char* labels_path, size_t n);

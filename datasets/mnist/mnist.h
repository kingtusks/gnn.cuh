#pragma once
#include <vector>

namespace MNIST {
    inline constexpr size_t MNIST_IMG_SIZE{784};
    inline constexpr size_t MNIST_N_CLASSES{10};
    inline constexpr size_t MNIST_STRIDE{MNIST_IMG_SIZE + MNIST_N_CLASSES};
    std::vector<float> mnist_load_td(const char* images_path, const char* labels_path, size_t n);
}

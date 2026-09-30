#include <cstdio>
#include <fstream>
#include <vector>

#define N_TRAIN 60000
#define N_TEST 10000
#define IDX 45

std::vector<std::vector<float>> extract_images(std::ifstream& file, size_t iter) {
    file.seekg(16);
    std::vector<std::vector<float>> images;

    for (size_t n = 0; n < iter; ++n) {
        std::vector<unsigned char> raw(784);
        file.read(reinterpret_cast<char*>(raw.data()), 784);

        std::vector<float> image(784);
        for (size_t i = 0; i < 784; ++i)
            image[i] = raw[i] / 255.f;
        images.push_back(image);
    }
    return images;
}

//instead of being a float we need it to be an array[10] with one 1 and rest 0 corresponding to the #
std::vector<float> extract_labels(std::ifstream& file, size_t iter) {
    file.seekg(8);
    std::vector<unsigned char> raw(iter);
    file.read(reinterpret_cast<char*>(raw.data()), iter);

    std::vector<float> labels(iter);
    for (size_t i = 0; i < iter; ++i)
        labels[i] = (float) raw[i];
    return labels;
}

int main() {
    std::ifstream train_images_file("data/train-images.idx3-ubyte", std::ios::binary);
    std::ifstream test_images_file("data/t10k-images.idx3-ubyte", std::ios::binary);
    std::ifstream train_labels_file("data/train-labels.idx1-ubyte", std::ios::binary);
    std::ifstream test_labels_file("data/t10k-labels.idx1-ubyte", std::ios::binary);

    auto train_images = extract_images(train_images_file, N_TRAIN);
    auto test_images = extract_images(test_images_file, N_TEST);
    auto train_labels = extract_labels(train_labels_file, N_TRAIN);
    auto test_labels = extract_labels(test_labels_file, N_TEST);

    std::vector<float> image = train_images[IDX];
    // for (size_t row = 0; row < 28; ++row) {
    //     for (size_t col = 0; col < 28; ++col) {
    //         printf("%c", (image[row * 28 + col] > 0.5f ? '#' : '.'));
    //     }
    //     printf("\n");
    // }

    //std::vector<size_t>
    printf("%f\n", train_labels[IDX]);

    return 0;
}

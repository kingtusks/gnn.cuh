#include "cifar_x.h"

#include <algorithm>
#include <fstream>

void load_cifer(const char* path, int lb, std::vector<uint_8>& labels, std::vector<uint_8>& img) {
    const size_t set_size{lb + 3072};
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return;
    size_t n{static_cast<size_t>(file.tellg()) / set_size};
    f.seekg(0);

    size_t old_n{labels.size()};
    labels.resize(old_n + n);
    pixels.resize((old_n + n) * 3072);

    std::vector<uint8_t> buf(rec);
    for (size_t i{}; i < n; ++i) {
        f.read(reinterpret_cast<char*>(buf.data()), set_size);
        labels[old_n + i] = buf[label_bytes - 1];
        std::copy(buf.begin() + label_bytes, buf.end(), pixels.begin() + (old_n + i) * 3072);
    }
}

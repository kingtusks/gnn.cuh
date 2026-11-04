#pragma once

#include <cstdint>
#include <vector>

namespace Cifar {
    void load_cifar(const char* path, int lb, std::vector<uint_8>& labels, std::vector<uint_8>& img);
}

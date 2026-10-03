# gnn

A small GPU neural network library in two single-header files, written from scratch with no cuBLAS, rocBLAS, or ML frameworks. Builds for NVIDIA (CUDA) or AMD (HIP) from the same source.

- `gmat.cuh`: GPU matrix primitives
- `gnn.cuh`: fully connected network, backprop, and mini-batch training

Reaches **[97.11]%** on MNIST ([training / test] set).

## Credits

Originally supposed to be a CUDA port of [tsoding/nn.h](https://github.com/tsoding/nn.h), following his [Machine Learning in C series](https://www.youtube.com/watch?v=PGSba51aRYU&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw) but later added new things other than FNNs.

## Build

### NVIDIA (CUDA)

Requires the CUDA toolkit and an NVIDIA GPU.

```
nvcc -O2 -arch=native gnn.cu -o gnn
```

### AMD (HIP)

Requires ROCm and a supported AMD GPU. Define `GMAT_HIP` to switch the CUDA runtime calls over to HIP, and set `--offload-arch` to your GPU's target (`rocminfo` prints it, for example `gfx1100`).

```
hipcc -O2 -DGMAT_HIP --offload-arch=gfxXXXX gnn.cu -o gnn
```

Tested on: [GPU / ROCm version, or "CUDA only so far"].

## Usage

```cpp
#define GNN_IMPLEMENTATION
#include "gnn.cuh"

size_t dim[] = {784, 128, 64, 10};
NN nn = nn_alloc(dim, ARRAY_LEN(dim), 100);   // 100 = batch size
NN g  = nn_alloc(dim, ARRAY_LEN(dim), 100);   // gradient buffers, same shape

nn_train(nn, g, ti, to, 1.0f, 20);            // rate, epochs
nn_test(nn, ti, to);                          // prints accuracy
```

`ti` and `to` are `Mat`s on the GPU with one sample per row.

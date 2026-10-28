# gnn

A small GPU neural network library in two single-header files, written from scratch with no cuBLAS, rocBLAS, or ML frameworks. Builds for NVIDIA (CUDA) or AMD (HIP) from the same source.

- `gmat.cuh`: GPU matrix primitives
- `gnn.cuh`: fully connected network, backprop, and mini-batch training

Reaches **[98.13]%** on MNIST (test set).
> Rate: 1e-1, Epochs: 450, Batch Size: 100

## Credits

Originally supposed to be a CUDA port of [tsoding/nn.h](https://github.com/tsoding/nn.h), following his [Machine Learning in C series](https://www.youtube.com/watch?v=PGSba51aRYU&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw) but later added new things other than FNNs.

`CIFER_10` and `CIFER_100` datasets are from this paper: [Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.](https://cave.cs.toronto.edu/kriz/learning-features-2009-TR.pdf).

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

Tested on: NVIDIA RTX 5050 and GTX 1060 (need an AMD tester)

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

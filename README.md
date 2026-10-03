# gnn

A small CUDA neural network library in two single-header files, written from scratch with no cuBLAS or ML frameworks.

- `gmat.cuh`: GPU matrix primitives
- `gnn.cuh`: fully connected network, backprop, and mini-batch training

Reaches **[97.11]%** on MNIST ([training / test] set).

## Credits

A CUDA port of [tsoding/nn.h](https://github.com/tsoding/nn.h), following his [Machine Learning in C series](https://www.youtube.com/watch?v=PGSba51aRYU&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw).

## Build

Requires the CUDA toolkit and an NVIDIA GPU.

```
nvcc -O2 gnn.cu -o gnn
```

## Usage

```c
#define GNN_IMPLEMENTATION
#include "gnn.cuh"

size_t dim[] = {784, 128, 64, 10};
NN nn = nn_alloc(dim, ARRAY_LEN(dim), 100);   // 100 = batch size
NN g  = nn_alloc(dim, ARRAY_LEN(dim), 100);   // gradient buffers, same shape

nn_train(nn, g, ti, to, 1.0f, 20);            // rate, epochs
nn_test(nn, ti, to);                          // prints accuracy
```

`ti` and `to` are `Mat`s on the GPU with one sample per row.

## Notes

- Sigmoid activation, mean squared error, mini-batch SGD
- Each layer is one tiled matrix multiply over the whole batch
- `mat_dot` is hand-written, roughly 15% of cuBLAS.

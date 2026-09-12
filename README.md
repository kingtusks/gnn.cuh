# gnn.cuh

A CUDA/HIP port of [Tsoding's `nn.h`](https://github.com/tsoding/nn.h), following his
["Machine Learning in C"](https://www.youtube.com/watch?v=PGSba51aRYU&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw)
series on YouTube. All credit for the original design and approach goes to him. This is just my attempt to reimplement it targeting the GPU instead of plain C.

> **Note:** This is a learning project, not a production-ready library. I'm using it to understand CUDA (memory management, kernel launches, grid/block sizing) and the fundamentals of ML (matrices, forward/backward passes, gradient descent) from the ground up. Expect rough edges, missing error handling, and API decisions that trade correctness/robustness for "does this teach me something."

#include "gnn.cuh"

int main() {
    gnn::Tensor in({4, 4, 1, 1}, {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16});
    gnn::Tensor delta({2, 2, 1, 1});  delta.fill(1.0f);
    gnn::Tensor gmax({4, 4, 1, 1}), gavg({4, 4, 1, 1});

    gmax.max_pool_grad(in, delta, 2, 2, 2, 2);
    gavg.avg_pool_grad(delta, 2, 2, 2, 2);
    TENSOR_PRINT(gmax);
    TENSOR_PRINT(gavg);

    return 0;
}

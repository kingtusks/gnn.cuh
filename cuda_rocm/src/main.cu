#include "gnn.cuh"

int main() {
    gnn::Tensor in({3, 3, 1, 1}, {1,2,3, 4,5,6, 7,8,9});
    gnn::Tensor w({2, 2, 1, 1}, {1,1, 1,1});
    gnn::Tensor b({1, 1}, {0});
    gnn::Tensor out({2, 2, 1, 1});

    out.conv(in, w, b);

    gnn::Tensor delta({2, 2, 1, 1});  delta.fill(1.0f);
    gnn::Tensor gb({1, 1}), gw({2, 2, 1, 1}), gin({3, 3, 1, 1});

    gb.conv_grad_b(delta);
    gw.conv_grad_w(in, delta);
    gin.conv_grad_in(delta, w);

    TENSOR_PRINT(gb);
    TENSOR_PRINT(gw);
    TENSOR_PRINT(gin);
    return 0;
}

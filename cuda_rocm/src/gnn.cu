#include "gnn.cuh"

namespace NeuralNetwork {
    NN nn_alloc(size_t* dim, size_t dim_len, size_t batch) {
        NN nn;
        nn.count = dim_len - 1;
        nn.batch = batch;
        size_t sizeof_wb{sizeof(Matrix::Mat) * nn.count};
        size_t sizeof_a{sizeof(Matrix::Mat) * (nn.count + 1)};

        nn.w = (Matrix::Mat*) malloc(sizeof_wb);
        nn.b = (Matrix::Mat*) malloc(sizeof_wb);
        nn.a = (Matrix::Mat*) malloc(sizeof_a);
        GNN_ASSERT(nn.w && nn.b && nn.a);

        NN_INPUT(nn) = Matrix::mat_alloc(batch, dim[0]);
        for (size_t i{1}; i < dim_len; ++i) {
            nn.w[i - 1] = Matrix::mat_alloc(dim[i - 1], dim[i]);
            nn.b[i - 1] = Matrix::mat_alloc(1, dim[i]);
            nn.a[i] = Matrix::mat_alloc(batch, dim[i]);
        }

        GNN_CUDA_CHECK(cudaMalloc((void**)&nn.dc, sizeof(float)));
        return nn;
    }

    NN nn_alloc_copy(NN nn) {
        size_t n{nn.count + 1};
        size_t* dim{(size_t*) malloc(sizeof(size_t) * n)};
        GNN_ASSERT(dim);

        dim[0] = NN_INPUT(nn).cols;
        for (size_t i{}; i < nn.count; ++i)
            dim[i + 1] = nn.w[i].cols;

        NN a{nn_alloc(dim, n, nn.batch)};
        free(dim);
        return a;
    }

    void nn_rand(NN nn, float low, float high) {
        for (size_t i{}; i < nn.count; ++i) {
            mat_rand(nn.w[i], low, high);
            mat_rand(nn.b[i], low, high);
        }
    }

    void nn_rand_he(NN nn) {
        float lim;
        for (size_t i{}; i < nn.count; ++i) {
            lim = sqrtf(6.f / (float) nn.w[i].rows);
            mat_rand(nn.w[i], -lim, lim);
            mat_fill(nn.b[i], 0.f);
        }
    }

    void nn_fill(NN nn, float n) {
        for (size_t i{}; i < nn.count; ++i) {
            mat_fill(nn.w[i], n);
            mat_fill(nn.b[i], n);
            mat_fill(nn.a[i], n);
        }
        mat_fill(NN_OUTPUT(nn), n);
    }

    void nn_forward(NN nn) {
        for (size_t i{}; i < nn.count; ++i) {
            mat_dot(nn.a[i + 1], nn.a[i], nn.w[i]);
            mat_sum_bias(nn.a[i + 1], nn.b[i]);
            if (i + 1 < nn.count) mat_leaky_relu(nn.a[i + 1]);
            else mat_softmax(nn.a[i + 1]);
        }
    }

    __global__ void nn_cost_kernel(Matrix::Mat out, Matrix::Mat y, float* dcost) {
        extern __shared__ float sd[];
        size_t tx{threadIdx.x};
        size_t idx{(size_t) blockIdx.x * blockDim.x + tx};
        size_t area{out.rows * out.cols};

        float v{};
        if (idx < area) {
            size_t i{idx / out.cols};
            size_t j{idx % out.cols};
            v += -MAT_AT(y, i, j) * logf(MAT_AT(out, i, j) + 1e-7f);
        }

        sd[tx] = v;
        __syncthreads();

        for (size_t s{blockDim.x / 2}; s > 0; s >>= 1) {
            if (s > tx) sd[tx] += sd[tx + s];
            __syncthreads();
        }

        if (tx == 0)
            atomicAdd(dcost, sd[0]);
    }

    float nn_cost(NN nn, Matrix::Mat ti, Matrix::Mat to) {
        GNN_ASSERT(ti.rows == to.rows);
        GNN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

        unsigned int threads{256};
        size_t area{(size_t) NN_OUTPUT(nn).rows * NN_OUTPUT(nn).cols};
        unsigned int blocks{(unsigned int) ((area + threads - 1) / threads)};

        GNN_CUDA_CHECK(cudaMemset(nn.dc, 0, sizeof(float)));

        size_t nb{};
        for (size_t i{}; i + nn.batch <= ti.rows; i += nn.batch, ++nb) {
            mat_copy(NN_INPUT(nn), mat_rows(ti, i, nn.batch));
            nn_forward(nn);
            nn_cost_kernel<<<blocks, threads, threads * sizeof(float)>>>(NN_OUTPUT(nn), Matrix::mat_rows(to, i, nn.batch), nn.dc);
            GNN_CUDA_CHECK(cudaGetLastError());
        }
        GNN_ASSERT(nb > 0);

        float c;
        GNN_CUDA_CHECK(cudaMemcpy(&c, nn.dc, sizeof(float), cudaMemcpyDeviceToHost));
        return c / (float) (nb * nn.batch);
    }

    void nn_finite_diff(NN nn, NN g, Matrix::Mat ti, Matrix::Mat to, float eps) {
        float c{nn_cost(nn, ti, to)};
        for (size_t i{}; i < nn.count; ++i) {
            nn_finite_diff_nudge(nn, nn.w[i], g.w[i], ti, to, eps, c);
            nn_finite_diff_nudge(nn, nn.b[i], g.b[i], ti, to, eps, c);
        }
    }

    __global__ void nn_out_grad_kernel(Matrix::Mat a, Matrix::Mat y, Matrix::Mat da, float scale) {
        size_t i{(size_t) blockIdx.y * blockDim.y + threadIdx.y};
        size_t j{(size_t) blockIdx.x * blockDim.x + threadIdx.x};
        if (i < a.rows && j < a.cols)
            MAT_AT(da, i, j) = scale * (MAT_AT(a, i, j) - MAT_AT(y, i, j));
    }

    __global__ void nn_activation_grad_kernel(Matrix::Mat da, Matrix::Mat a) {
        size_t i{(size_t) blockIdx.y * blockDim.y + threadIdx.y};
        size_t j{(size_t) blockIdx.x * blockDim.x + threadIdx.x};
        if (i < a.rows && j < a.cols && MAT_AT(a, i, j) <= 0.f) MAT_AT(da, i, j) *= 0.01f;
    }

    void nn_backprop(NN nn, NN g, Matrix::Mat ti, Matrix::Mat to) {
        GNN_ASSERT(ti.rows == nn.batch && to.rows == nn.batch);
        GNN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

        Matrix::mat_copy(NN_INPUT(nn), ti);
        nn_forward(nn);

        dim3 threads{32, 8};
        Matrix::Mat out{NN_OUTPUT(nn)};
        dim3 blocks_out{(unsigned int) ((out.cols + threads.x - 1) / threads.x), (unsigned int) ((out.rows + threads.y - 1) / threads.y)};
        nn_out_grad_kernel<<<blocks_out, threads>>>(out, to, g.a[nn.count], 1.f / (float) nn.batch);
        GNN_CUDA_CHECK(cudaGetLastError());

        Matrix::Mat a;
        Matrix::Mat dz;
        for (size_t l{nn.count}; l > 0; --l) {
            a = nn.a[l];
            dz = g.a[l];
            if (l < nn.count) {
                dim3 blocks_activation{(unsigned int) ((a.cols + threads.x - 1) / threads.x), (unsigned int) ((a.rows + threads.y - 1) / threads.y)};
                nn_activation_grad_kernel<<<blocks_activation, threads>>>(dz, a);
                GNN_CUDA_CHECK(cudaGetLastError());
            }
            Matrix::mat_dot_ta(g.w[l - 1], nn.a[l - 1], dz);
            Matrix::mat_sum_collapse(g.b[l - 1], dz);
            if (l > 1)
                Matrix::mat_dot_tb(g.a[l - 1], dz, nn.w[l - 1]);
        }
    }

    void nn_sgd(NN nn, NN g, float rate) {
        for (size_t i{}; i < nn.count; ++i) {
            Matrix::mat_sgd_helper(nn.w[i], g.w[i], rate);
            Matrix::mat_sgd_helper(nn.b[i], g.b[i], rate);
        }
    }

    void nn_momentum(NN nn, NN g, NN v, float rate) {
        for (size_t i{}; i < nn.count; ++i) {
            Matrix::mat_momentum_helper(nn.w[i], g.w[i], v.w[i], rate);
            Matrix::mat_momentum_helper(nn.b[i], g.b[i], v.b[i], rate);
        }
    }

    void nn_adam(NN nn, NN g, NN m1, NN m2, size_t t, float rate) {
        for (size_t i{}; i < nn.count; ++i) {
            Matrix::mat_adam_helper(nn.w[i], g.w[i], m1.w[i], m2.w[i], t, rate);
            Matrix::mat_adam_helper(nn.b[i], g.b[i], m1.b[i], m2.b[i], t, rate);
        }
    }

    void nn_train(NN nn, Matrix::Mat ti, Matrix::Mat to, size_t epochs, float rate, Optimizer opt) {
        nn_rand_he(nn);
        NN g{nn_alloc_copy(nn)};
        NN v{}, m1{}, m2{};
        size_t t{};

        switch (opt) {
            case Optimizer::SGD:
                break;
            case Optimizer::SGDMomentum:
                v = nn_alloc_copy(nn);
                nn_fill(v, 0);
                break;
            case Optimizer::Adam:
                m1 = nn_alloc_copy(nn);
                m2 = nn_alloc_copy(nn);
                nn_fill(m1, 0);
                nn_fill(m2, 0);
                break;
        }

        for (size_t i{1}; i <= epochs; ++i) {
            for (size_t j{}; j + nn.batch <= ti.rows; j += nn.batch) {
                nn_backprop(nn, g, mat_rows(ti, j, nn.batch), mat_rows(to, j, nn.batch));
                switch (opt) {
                    case Optimizer::SGD:
                        nn_sgd(nn, g, rate);
                        break;
                    case Optimizer::SGDMomentum:
                        nn_momentum(nn, g, v, rate);
                        break;
                    case Optimizer::Adam:
                        nn_adam(nn, g, m1, m2, ++t, rate);
                        break;
                }
            }
            if (i % NN_PRINT_INTERVAL == 0)
                printf("%zu: cost: %f\n", i, nn_cost(nn, ti, to));
        }

        nn_free(g);
        switch (opt) {
            case Optimizer::SGD:
                break;
            case Optimizer::SGDMomentum:
                nn_free(v);
                break;
            case Optimizer::Adam:
                nn_free(m1); nn_free(m2);
                break;
        }
    }

    __global__ void nn_test_kernel(Matrix::Mat out, Matrix::Mat y, float* dcost) {
        size_t i{(size_t) blockIdx.x * blockDim.x + threadIdx.x};
        if (i >= out.rows) return;

        size_t po{}, py{};
        for (size_t j{1}; j < out.cols; ++j) {
            if (MAT_AT(out, i, j) > MAT_AT(out, i, po)) po = j;
            if (MAT_AT(y, i, j) > MAT_AT(y, i, py)) py = j;
        }

        if (po == py) atomicAdd(dcost, 1.f);
    }

    float nn_test(NN nn, Matrix::Mat ti, Matrix::Mat to) {
        GNN_ASSERT(ti.rows == to.rows);
        GNN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

        unsigned int threads{256};
        unsigned int blocks{(unsigned int) ((nn.batch + threads - 1) / threads)};

        GNN_CUDA_CHECK(cudaMemset(nn.dc, 0, sizeof(float)));

        size_t nb{};
        for (size_t i{}; i + nn.batch <= ti.rows; i += nn.batch, ++nb) {
            Matrix::mat_copy(NN_INPUT(nn), mat_rows(ti, i, nn.batch));
            nn_forward(nn);
            nn_test_kernel<<<blocks, threads>>>(NN_OUTPUT(nn), mat_rows(to, i, nn.batch), nn.dc);
            GNN_CUDA_CHECK(cudaGetLastError());
        }
        GNN_ASSERT(nb > 0);

        float correct;
        GNN_CUDA_CHECK(cudaMemcpy(&correct, nn.dc, sizeof(float), cudaMemcpyDeviceToHost));

        size_t total{nb * nn.batch};
        float acc{100.f * correct / (float) total};
        printf("accuracy: %.2f%% (%d / %zu)\n", acc, (int) correct, total);
        return acc;
    }

    void nn_infer(NN nn, Matrix::Mat ti) {
        for (size_t i{}; i < ti.rows; ++i) {
            Matrix::mat_copy(NN_INPUT(nn), mat_rows(ti, i, 1));
            nn_forward(nn);
            Matrix::mat_print(mat_row(NN_OUTPUT(nn), i), "output");
        }
    }

    void nn_free(NN nn) {
        for (size_t i{}; i < nn.count; ++i) {
            GNN_CUDA_CHECK(cudaFree(nn.w[i].es));
            GNN_CUDA_CHECK(cudaFree(nn.b[i].es));
            GNN_CUDA_CHECK(cudaFree(nn.a[i].es));
        }
        GNN_CUDA_CHECK(cudaFree(nn.a[nn.count].es));
        GNN_CUDA_CHECK(cudaFree(nn.dc));
        free(nn.w);
        free(nn.b);
        free(nn.a);
    }

    void nn_save(NN nn, const char* path) {
        FILE* f{fopen(path, "wb")};
        GNN_ASSERT(f);

        nn_write_u64(f, nn.count);
        for (size_t i{}; i < nn.count; ++i) {
            nn_write_u64(f, nn.w[i].rows);
            nn_write_u64(f, nn.w[i].cols);
            nn_mat_write(f, nn.w[i]);
            nn_mat_write(f, nn.b[i]);
        }

        fclose(f);
    }

    void nn_load(NN nn, const char* path) {
        FILE* f{fopen(path, "rb")};
        GNN_ASSERT(f);

        [[maybe_unused]] const uint64_t count{nn_read_u64(f)};
        GNN_ASSERT(count == nn.count);

        for (size_t i{}; i < nn.count; ++i) {
            [[maybe_unused]] const uint64_t r{nn_read_u64(f)};
            [[maybe_unused]] const uint64_t c{nn_read_u64(f)};
            GNN_ASSERT(r == nn.w[i].rows && c == nn.w[i].cols);
            nn_mat_read(f, nn.w[i]);
            nn_mat_read(f, nn.b[i]);
        }

        fclose(f);
    }

    void nn_print(NN nn, const char* name) {
        printf("%s\n", name);
        for (size_t i{}; i < nn.count; ++i) {
            Matrix::mat_print(nn.w[i], "w");
            Matrix::mat_print(nn.b[i], "b");
        }
        printf("\n");
    }

    void nn_finite_diff_nudge(NN nn, Matrix::Mat m, Matrix::Mat gm, Matrix::Mat ti, Matrix::Mat to, float eps, float c) {
        for (size_t j{}; j < m.rows; ++j) {
            for (size_t k{}; k < m.cols; ++k) {
                float* p{&MAT_AT(m, j, k)};
                float saved, nudged, grad;

                GNN_CUDA_CHECK(cudaMemcpy(&saved, p, sizeof(float), cudaMemcpyDeviceToHost));
                nudged = saved + eps;
                GNN_CUDA_CHECK(cudaMemcpy(p, &nudged, sizeof(float), cudaMemcpyHostToDevice));

                grad = (nn_cost(nn, ti, to) - c) / eps;

                GNN_CUDA_CHECK(cudaMemcpy(p, &saved, sizeof(float), cudaMemcpyHostToDevice));
                GNN_CUDA_CHECK(cudaMemcpy(&MAT_AT(gm, j, k), &grad, sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }

    void nn_write_u64(FILE* f, uint64_t n) {
        size_t ok{fwrite(&n, sizeof(n), 1, f)};
        GNN_ASSERT(ok == 1);
        (void) ok;
    }

    uint64_t nn_read_u64(FILE* f) {
        uint64_t v{};
        size_t ok{fread(&v, sizeof(v), 1, f)};
        GNN_ASSERT(ok == 1);
        (void) ok;
        return v;
    }

    void nn_mat_write(FILE* f, Matrix::Mat m) {
        size_t n{m.rows * m.cols};
        float* h{(float*) malloc(n * sizeof(float))};
        GNN_ASSERT(h);
        GNN_CUDA_CHECK(cudaMemcpy(h, m.es, n * sizeof(float), cudaMemcpyDeviceToHost));
        size_t ok{fwrite(h, sizeof(float), n, f)};
        GNN_ASSERT(ok == n);
        (void) ok;
        free(h);
    }

    void nn_mat_read(FILE* f, Matrix::Mat m) {
        size_t n{m.rows * m.cols};
        float* h{(float*) malloc(n * sizeof(float))};
        GNN_ASSERT(h);
        size_t ok{fread(h, sizeof(float), n, f)};
        GNN_ASSERT(ok == n);
        (void) ok;
        GNN_CUDA_CHECK(cudaMemcpy(m.es, h, n * sizeof(float), cudaMemcpyHostToDevice));
        free(h);
    }
}

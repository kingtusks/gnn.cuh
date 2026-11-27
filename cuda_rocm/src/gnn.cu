#include "gnn.cuh"

namespace gnn
{
    namespace kernels
    {
        __global__ void cost(const float* o_es, const float* y_es, float* dc_es,
                KernelTensorInfo o_kti, KernelTensorInfo y_kti, size_t count)
        {
            size_t i {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            extern __shared__ float sd[];
            size_t tx {threadIdx.x};

            float v {};
            if (i < count)
            {
                size_t c {i % o_kti.shape_[0]};
                size_t r {i / o_kti.shape_[0]};
                v = -y_es[c * y_kti.strides_[0] + r * y_kti.strides_[1]]
                    * logf(o_es[c * o_kti.strides_[0] + r * o_kti.strides_[1]] + 1e-7f);
            }
            sd[tx] = v;
            __syncthreads();

            for (size_t s {blockDim.x / 2}; s > 0; s >>= 1)
            {
                if (tx < s) sd[tx] += sd[tx + s];
                __syncthreads();
            }
            if (tx == 0) atomicAdd(dc_es, sd[0]);
        }

        __global__ void out_grad(const float* a_es, float* ga_es, const float* y_es,
                KernelTensorInfo a_kti, KernelTensorInfo ga_kti, KernelTensorInfo y_kti,
                size_t count, float scale)
        {
            size_t i {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            if (i >= count) return;

            size_t c {i % ga_kti.shape_[0]};
            size_t r {i / ga_kti.shape_[0]};

            ga_es[c * ga_kti.strides_[0] + r * ga_kti.strides_[1]] =
                scale * (a_es[c * a_kti.strides_[0] + r * a_kti.strides_[1]]
                        - y_es[c * y_kti.strides_[0] + r * y_kti.strides_[1]]);
        }

        //leaky relu for now (may add options later)
        __global__ void activation_grad(const float* a_es, float* ga_es,
                KernelTensorInfo a_kti, KernelTensorInfo ga_kti,
                size_t count, float slope)
        {
            size_t i {(size_t) blockIdx.x * blockDim.x + threadIdx.x};
            if (i >= count) return;

            size_t c {i % ga_kti.shape_[0]};
            size_t r {i / ga_kti.shape_[0]};

            if (a_es[c * a_kti.strides_[0] + r * a_kti.strides_[1]] <= 0.f)
                ga_es[c * ga_kti.strides_[0] + r * ga_kti.strides_[1]] *= slope;
        }
    } //namespace kernels

    NN::NN(const std::vector<unsigned int>& dims, unsigned int batch) :
        count_(dims.size() - 1), batch_(batch)
    {
        GNN_ASSERT(dims.size() >= 2);

        w_.reserve(count_);
        b_.reserve(count_);
        a_.reserve(count_ + 1);

        a_.push_back(Tensor({dims[0], batch}));
        for (size_t i {1}; i < dims.size(); ++i)
        {
            w_.push_back(Tensor({dims[i], dims[i - 1]}));
            b_.push_back(Tensor({dims[i], 1}));
            a_.push_back(Tensor({dims[i], batch}));
        }
    }

    void NN::nn_rand(float low, float high)
    {
        for (size_t i {}; i < count_; ++i)
        {
            w_[i].rand(low, high);
            b_[i].rand(low, high);
        }
    }

    void NN::nn_rand_he()
    {
        for (size_t i {}; i < count_; ++i)
        {
            float lim {sqrtf(6.0f / (float) w_[i].shape_[1])};
            w_[i].rand(-lim, lim);
            b_[i].fill(0.0f);
        }
    }

    void NN::nn_fill(float n)
    {
        for (size_t i {}; i < count_; ++i)
        {
            w_[i].fill(n);
            b_[i].fill(n);
            a_[i].fill(n);
        }
        output().fill(n);
    }

    void NN::nn_forward()
    {
        for (size_t i {}; i < count_; ++i)
        {
            a_[i + 1].dot(w_[i], a_[i]);
            a_[i + 1].sum_bias(b_[i]);

            if (i + 1 < count_) a_[i + 1].leaky_relu();
            else a_[i + 1].softmax();
        }
    }

    float NN::nn_cost(const Tensor& ti, const Tensor& to)
    {
        GNN_ASSERT(ti.shape_[1] == to.shape_[1]);
        GNN_ASSERT(to.shape_[0] == output().shape_[0]);

        dc_.fill(0.0f);
        unsigned int threads {256};
        unsigned int blocks {(unsigned int) (output().count_ + threads - 1) / threads};

        size_t nb {};
        for (unsigned int i {}; i + batch_ <= ti.shape_[1]; i += batch_, ++nb)
        {
            input().copy_from(ti.slice(1, i, batch_));
            nn_forward();
            kernels::cost<<<blocks, threads, threads * sizeof(float)>>>(
                    output().es_, to.slice(1, i, batch_).es_, dc_.es_,
                    output().kernel_info(), to.slice(1, i, batch_).kernel_info(),
                    output().count_);
            CUDA_CHECK(cudaGetLastError());
        }
        GNN_ASSERT(nb > 0);

        float c;
        CUDA_CHECK(cudaMemcpy(&c, dc_.es_, sizeof(float), cudaMemcpyDeviceToHost));
        return c / (float) (nb * batch_);
    }

    static void fdnudge(NN& nn, Tensor& t, Tensor& gt,
            const Tensor& ti, const Tensor& to, float eps, float c)
    {
        for (size_t i {}; i < t.count_; ++i)
        {
            float saved;
            CUDA_CHECK(cudaMemcpy(&saved, t.es_ + i, sizeof(float), cudaMemcpyDeviceToHost));

            float nudged {saved + eps};
            CUDA_CHECK(cudaMemcpy(t.es_ + i, &nudged, sizeof(float), cudaMemcpyHostToDevice));

            float grad {(nn.nn_cost(ti, to) - c) / eps};
            CUDA_CHECK(cudaMemcpy(t.es_ + i, &saved, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gt.es_ + i, &grad, sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    void NN::finite_diff(NN& g, const Tensor& ti, const Tensor& to, float eps)
    {
        float c {nn_cost(ti, to)};
        for (size_t i {}; i < count_; ++i)
        {
            fdnudge(*this, w_[i], g.w_[i], ti, to, eps, c);
            fdnudge(*this, b_[i], g.b_[i], ti, to, eps, c);
        }
    }

    void NN::backprop(NN& g, const Tensor& ti, const Tensor& to)
    {
        GNN_ASSERT(ti.shape_[1] == batch_ && to.shape_[1] == batch_);
        input().copy_from(ti);
        nn_forward();

        unsigned int threads {256};
        Tensor& out {output()};
        Tensor& gout {g.output()};
        unsigned int o_blocks {(unsigned int) (out.count_ + threads - 1) / threads};
        kernels::out_grad<<<o_blocks, threads>>>(out.es_, gout.es_, to.es_,
                out.kernel_info(), gout.kernel_info(), to.kernel_info(),
                out.count_, 1.0f / (float) batch_);
        CUDA_CHECK(cudaGetLastError());

        for (size_t i {count_}; i > 0; --i)
        {
            Tensor& g_ai {g.a_[i]};
            const Tensor& ai {a_[i]};

            if (i < count_)
            {
                unsigned int a_blocks {(unsigned int) (ai.count_ + threads - 1) / threads};
                kernels::activation_grad<<<a_blocks, threads>>>(ai.es_, g_ai.es_,
                        ai.kernel_info(), g_ai.kernel_info(), ai.count_, 1e-2f);
                CUDA_CHECK(cudaGetLastError());
            }

            g.w_[i - 1].dot(g_ai, a_[i - 1], false, true);
            g.b_[i - 1].sum_collapse(g_ai);
            if (i > 1)
                g.a_[i - 1].dot(w_[i - 1], g_ai, true, false);
        }
    }
} //namespace gnn

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "common.h"
#include "forward_kernel.cuh"
#include <cutlass/cutlass.h>

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << msg << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static void reference_attention_cpu(const std::vector<half>& Q,
                                    const std::vector<half>& K,
                                    const std::vector<half>& V,
                                    std::vector<float>& O_ref,
                                    int B, int N, int H, int d) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < N; ++i) {
                std::vector<float> logits(N, 0.0f);
                for (int j = 0; j < N; ++j) {
                    float dot = 0.0f;
                    for (int t = 0; t < d; ++t) {
                        const int q_idx = ((b * N + i) * H + h) * d + t;
                        const int k_idx = ((b * N + j) * H + h) * d + t;
                        dot += __half2float(Q[q_idx]) * __half2float(K[k_idx]);
                    }
                    logits[j] = dot * scale;
                }

                float m = -INFINITY;
                for (float x : logits) m = std::max(m, x);
                float l = 0.0f;
                for (float &x : logits) {
                    x = std::exp(x - m);
                    l += x;
                }

                for (int t = 0; t < d; ++t) {
                    float out = 0.0f;
                    for (int j = 0; j < N; ++j) {
                        const int v_idx = ((b * N + j) * H + h) * d + t;
                        out += (logits[j] / l) * __half2float(V[v_idx]);
                    }
                    const int o_idx = ((b * N + i) * H + h) * d + t;
                    O_ref[o_idx] = out;
                }
            }
        }
    }
}

int main() {
    // 这个 demo 对应“基础版（未优化）FlashAttention forward”
    constexpr int B = 1;
    constexpr int N = 64;
    constexpr int H = 1;
    constexpr int d = 64;

    constexpr int Br = 64;
    constexpr int Bc = 64;

    const int numel = B * N * H * d;
    const size_t bytes = numel * sizeof(half);

    std::vector<half> h_in(numel);
    std::vector<half> h_q(numel);
    std::vector<half> h_k(numel);
    std::vector<half> h_v(numel);
    std::vector<half> h_out(numel, __float2half(0.0f));
    std::vector<float> h_ref(numel, 0.0f);
    for (int i = 0; i < numel; ++i) {
        const float base = static_cast<float>(i % 23) * 0.03f;
        h_q[i] = __float2half(base);
        h_k[i] = __float2half(base * 1.1f);
        h_v[i] = __float2half(base * 0.9f);
    }

    half *d_q = nullptr;
    half *d_k = nullptr;
    half *d_v = nullptr;
    half *d_out = nullptr;
    check_cuda(cudaMalloc(&d_q, bytes), "cudaMalloc d_q");
    check_cuda(cudaMalloc(&d_k, bytes), "cudaMalloc d_k");
    check_cuda(cudaMalloc(&d_v, bytes), "cudaMalloc d_v");
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");
    check_cuda(cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D Q");
    check_cuda(cudaMemcpy(d_k, h_k.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D K");
    check_cuda(cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D V");
    check_cuda(cudaMemset(d_out, 0, bytes), "cudaMemset d_out");

    const int64_t batch_stride = static_cast<int64_t>(N) * H * d;
    const int64_t seq_stride = static_cast<int64_t>(H) * d;
    const int64_t head_stride = d;

    ForwardKernelArgs args{
        /*Q=*/d_q,
        /*K=*/d_k,
        /*V=*/d_v,
        /*O=*/d_out,
        /*batch_stride=*/batch_stride,
        /*seq_stride=*/seq_stride,
        /*head_stride=*/head_stride,
        /*seq_len=*/N,
        /*n_heads=*/H,
        /*n_samples=*/B,
        /*d_head=*/d,
        /*n_Q_blocks=*/(N + Br - 1) / Br,
        /*n_KV_blocks=*/(N + Bc - 1) / Bc
    };

    check_cuda(fa_warp::launch_flash_forward_unoptimized<half, Br, Bc>(args),
               "launch_flash_forward_unoptimized");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check_cuda(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H output");

    // 计算 CPU 参考，便于快速确认功能正确
    reference_attention_cpu(h_q, h_k, h_v, h_ref, B, N, H, d);

    float max_abs_err = 0.0f;
    for (int i = 0; i < numel; ++i) {
        const float err = std::fabs(__half2float(h_out[i]) - h_ref[i]);
        max_abs_err = std::max(max_abs_err, err);
    }

    std::cout << "CUTLASS 基础版 FlashAttention demo (前 16 个元素):" << std::endl;
    for (int i = 0; i < 16; ++i) {
        std::cout << __half2float(h_out[i]) << (i == 15 ? '\n' : ' ');
    }
    std::cout << "max_abs_err = " << max_abs_err << std::endl;

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    return 0;
}

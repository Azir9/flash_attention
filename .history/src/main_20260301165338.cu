// 一个简单的 demo，演示如何调用 naive 版 flash attention：
//  - 构造很小的 Q/K/V (B=1, N=4, H=1, d=4)
//  - 在 CPU 上随机初始化
//  - 拷贝到 GPU，运行 flash_attention_forward_naive
//  - 把 O 拷回 CPU 打印前几项

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "flash_attention.cuh"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << msg << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    const int B = 1;
    const int N = 4;
    const int H = 1;
    const int d = 4;
    const int numel = B * N * H * d;

    std::cout << "Demo: naive flash attention, B=" << B
              << ", N=" << N << ", H=" << H << ", d=" << d << std::endl;

    std::vector<float> h_Q(numel);
    std::vector<float> h_K(numel);
    std::vector<float> h_V(numel);
    std::vector<float> h_O(numel, 0.f);

    // 简单初始化: Q/K/V 用一些固定值，方便你调试时手算或断点
    for (int i = 0; i < numel; ++i) {
        h_Q[i] = 0.1f * (i + 1);
        h_K[i] = 0.2f * (i + 1);
        h_V[i] = 0.3f * (i + 1);
    }

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
    size_t bytes = numel * sizeof(float);

    check_cuda(cudaMalloc(&d_Q, bytes), "cudaMalloc d_Q");
    check_cuda(cudaMalloc(&d_K, bytes), "cudaMalloc d_K");
    check_cuda(cudaMalloc(&d_V, bytes), "cudaMalloc d_V");
    check_cuda(cudaMalloc(&d_O, bytes), "cudaMalloc d_O");

    check_cuda(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy Q");
    check_cuda(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy K");
    check_cuda(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy V");

    fa_naive::flash_attention_forward_naive(d_Q, d_K, d_V, d_O,
                                            B, N, H, d,
                                            /*stream=*/0);

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check_cuda(cudaMemcpy(h_O.data(), d_O, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy O");

    std::cout << "Output O (shape [B, N, H, d] = [1, 4, 1, 4]):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "  token " << i << ": ";
        for (int j = 0; j < d; ++j) {
            int idx = ((0 * N + i) * H + 0) * d + j;
            std::cout << h_O[idx];
            if (j + 1 < d) std::cout << ", ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return 0;
}



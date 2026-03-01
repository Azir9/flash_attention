// 负责最朴素的矩阵乘法 / 向量点积（不使用 tensor core）

#pragma once

namespace fa_naive {

// 计算两个长度为 d 的向量的点积
inline __device__ float dot_product(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    int d) {
    float acc = 0.f;
    for (int k = 0; k < d; ++k) {
        acc += a[k] * b[k];
    }
    return acc;
}

// C[m, n] = A[m, k] @ B[k, n]
// A: [M, K], B: [K, N], C: [M, N]
// 行优先存储
inline __device__ void matmul_naive(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

} // namespace fa_naive

// 最朴素版本的 Flash Attention 实现，对应博客 Part-2/Part-3 中的 naive 逻辑：
// 1) 先做 QK^T 得到打分
// 2) 对每一行做稳定 softmax
// 3) 再和 V 做一次矩阵乘得到输出
//
// 这里直接把三个步骤融合到一个 CUDA kernel 里实现（但不做任何分块 / SMEM 优化）。

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "flash_attention.cuh"
#include "forward_kernel.cuh"
#include "load_store.cuh"
#include "gemm.cuh"
#include "online_softmax.cuh"

namespace fa_naive {

__global__ void flash_attention_forward_kernel_naive(const float* __restrict__ Q,
                                                     const float* __restrict__ K,
                                                     const float* __restrict__ V,
                                                     float* __restrict__ O,
                                                     int B, int N, int H, int d) {
    int b = blockIdx.z; // batch
    int h = blockIdx.y; // head
    int i = blockIdx.x; // query index

    if (b >= B || h >= H || i >= N) {
        return;
    }

    // 指向该 (b, i, h) 的 Q 行 和 O 行起始位置
    int q_offset = index_qkvd(b, i, h, B, N, H, d);
    int o_offset = index_o(b, i, h, B, N, H, d);
    const float* q_row = Q + q_offset;
    float* o_row = O + o_offset;

    // 1. 先算这一行对所有 key 的打分 S[i, j]，并做稳定 softmax
    extern __shared__ float shared[];
    float* logits = shared;      // 大小 N
    float* probs = shared + N;   // 大小 N

    // 计算 scale = 1 / sqrt(d)
    float scale = rsqrtf((float)d);

    // 遍历所有 key 位置 j，计算 dot(q_i, k_j)
    for (int j = 0; j < N; ++j) {
        int k_offset = index_kv(b, j, h, B, N, H, d);
        const float* k_row = K + k_offset;
        float score = dot_product(q_row, k_row, d) * scale;
        logits[j] = score;
    }

    // 对 logits[0..N-1] 做稳定 softmax，结果写到 probs[0..N-1]
    stable_softmax_row(logits, probs, N);

    // 2. 用 probs 做加权和，得到 O[b, i, h, :]
    //    O[i, :] = sum_j probs[j] * V[j, :]
    for (int dim = 0; dim < d; ++dim) {
        o_row[dim] = 0.f;
    }

    for (int j = 0; j < N; ++j) {
        float p = probs[j];
        int v_offset = index_kv(b, j, h, B, N, H, d);
        const float* v_row = V + v_offset;
        for (int dim = 0; dim < d; ++dim) {
            o_row[dim] += p * v_row[dim];
        }
    }
}

void flash_attention_forward_naive(const float* __restrict__ Q,
                                   const float* __restrict__ K,
                                   const float* __restrict__ V,
                                   float* __restrict__ O,
                                   int B, int N, int H, int d,
                                   cudaStream_t stream) {
    dim3 grid(N, H, B);   // (i, h, b)
    dim3 block(1);        // 一个线程负责一整行

    size_t smem_bytes = sizeof(float) * 2 * N; // logits[N] + probs[N]

    flash_attention_forward_kernel_naive<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, B, N, H, d);
}

} // namespace fa_naive




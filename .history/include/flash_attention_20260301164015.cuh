// 启动 Flash Attention naive kernel 的封装（仅 forward，单精度实现）

#pragma once

#include <cuda_runtime.h>

namespace fa_naive {

// Q, K, V, O: shape [B, N, H, d] 按 row-major 展开
// 这里实现的是最朴素的 FlashAttention 前向（实质上是稳定 softmax 的 scaled dot-product attention），
// 不做任何分块 / shared memory / tensor core 优化，只为验证逻辑正确。
//
// 计算:
//   S[b, i, h, j] = (Q[b, i, h, :] · K[b, j, h, :]) / sqrt(d)
//   P[b, i, h, j] = softmax_j(S[b, i, h, j])
//   O[b, i, h, :] = sum_j P[b, i, h, j] * V[b, j, h, :]
//
// 参数:
//  - B: batch size
//  - N: sequence length
//  - H: num heads
//  - d: head dim
//  - stream: CUDA stream
//
// 要求:
//  - 指针均指向 GPU 内存
//  - 数据类型为 float
void flash_attention_forward_naive(const float* __restrict__ Q,
                                   const float* __restrict__ K,
                                   const float* __restrict__ V,
                                   float* __restrict__ O,
                                   int B, int N, int H, int d,
                                   cudaStream_t stream = 0);

} // namespace fa_naive

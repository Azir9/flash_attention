// 核心 CUDA kernel（naive 版本）

#pragma once

#include <cuda_runtime.h>
#include <math.h>

#include "common.h"
#include "online_softmax.cuh"

namespace fa_naive {

// 为了简单，一个线程负责一个 (b, i, h) 的整行输出 O[b, i, h, :]
// 即: 每个线程要遍历所有 key 位置 j 和特征维 d。
//
// grid 配置建议:
//   dim3 grid(N, H, B);
//   dim3 block(1);
//
// 对于实际大规模应用这个实现会非常慢，但逻辑清晰，便于和博客中的 naive 推导对照。
__global__ void flash_attention_forward_kernel_naive(const float* __restrict__ Q,
                                                     const float* __restrict__ K,
                                                     const float* __restrict__ V,
                                                     float* __restrict__ O,
                                                     int B, int N, int H, int d);

} // namespace fa_naive

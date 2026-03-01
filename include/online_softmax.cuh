// Online / stable softmax 的朴素实现（仅用于说明思路）

#pragma once

#include <math.h>

// 这里给出两种形式:
// 1. 行内一次性稳定 softmax（需要一次性拿到整行 logits）
// 2. 线上 (online) softmax 更新步骤，用于分块累积时维护 m, l

namespace fa_naive {

// ---------- 1. 单行稳定 softmax ----------

// 输入 logits[0..len-1]，输出 probs[0..len-1]
// 算法:
//   m = max_j logits[j]
//   exps[j] = exp(logits[j] - m)
//   l = sum_j exps[j]
//   probs[j] = exps[j] / l
inline __device__ void stable_softmax_row(const float* __restrict__ logits,
                                          float* __restrict__ probs,
                                          int len) {
    // 求最大值
    float m = -FLT_MAX;
    for (int j = 0; j < len; ++j) {
        m = logits[j] > m ? logits[j] : m;
    }

    // 计算 exp 并累加
    float l = 0.f;
    for (int j = 0; j < len; ++j) {
        float e = expf(logits[j] - m);
        probs[j] = e;
        l += e;
    }

    // 归一化
    float inv_l = 1.f / (l + 1e-6f);
    for (int j = 0; j < len; ++j) {
        probs[j] *= inv_l;
    }
}

// ---------- 2. Online softmax building blocks ----------
//
// 假设我们按块遍历 logits:
//   第一次块得到局部的 max 和 sum_exp，记为 (m_1, l_1)
//   第二次块得到 (m_2, l_2) ...
// 通过下面两个函数，可以把各块合并成整体的 (m, l)。

// 合并一块的最大值:
//   m_new = max(m_prev, block_max)
inline __device__ void online_max_update(float& m_prev, float block_max) {
    m_prev = block_max > m_prev ? block_max : m_prev;
}

// 在已知整体最大值 m 的前提下，更新整体的 softmax 分母:
//   l_new = l_prev * exp(m_prev - m_new) + block_sum * exp(block_max - m_new)
inline __device__ void online_l_update(float& l_prev, float m_prev,
                                       float block_sum, float block_max,
                                       float m_new) {
    float scale_prev = expf(m_prev - m_new);
    float scale_blk = expf(block_max - m_new);
    l_prev = l_prev * scale_prev + block_sum * scale_blk;
}

} // namespace fa_naive

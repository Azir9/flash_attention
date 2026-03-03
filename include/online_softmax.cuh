#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace fa_warp {

// 所有线程参与 warp shuffle 的掩码
#define SHFL_ENTIRE_WARP_MASK 0xffffffff

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void scale_S_accum(
    accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    const accum_t &softmax_scale) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] *= softmax_scale;
        }
    }
}

// 你的原始版本默认做 warp 归约。这里增加 enable_warp_reduce 开关：
// - true: 适配同一行分布在多个线程（博客中的 warp 协作形式）
// - false: 适配一线程一行（当前 demo 形式）
template <bool enable_warp_reduce = true, int QO_fragments, int KV_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void calc_row_max(
    accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m_next)[QO_fragments],
    accum_t (&m_cur)[QO_fragments]) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        m_next[q] = m_cur[q];

        #pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            m_next[q] = max(m_next[q], S_accum[q][k]);
        }

        if constexpr (enable_warp_reduce) {
            m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 2), m_next[q]);
            m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 1), m_next[q]);
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void scale_l_O(
    accum_t (&m_next)[QO_fragments],
    accum_t (&m_cur)[QO_fragments],
    accum_t (&l)[QO_fragments],
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments]) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        const accum_t scale = expf(m_cur[q] - m_next[q]);
        m_cur[q] = m_next[q];
        l[q] *= scale;
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] *= scale;
        }
    }
}

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void exponentiate_tensor(
    accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m)[QO_fragments]) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] = expf(S_accum[q][k] - m[q]);
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void update_row_exp_sum(
    accum_t (&P_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            l[q] += P_accum[q][d_head];
        }
    }
}

template <bool enable_warp_reduce = true, int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void final_softmax_normalization(
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]) {
    // 行和归约（必要时）
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        if constexpr (enable_warp_reduce) {
            l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
            l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
        }
    }

    // 最终归一化
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        const accum_t inv_l = l[q] > 0 ? (accum_t(1) / l[q]) : accum_t(0);
        #pragma unroll
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] *= inv_l;
        }
    }
}

} // namespace fa_warp
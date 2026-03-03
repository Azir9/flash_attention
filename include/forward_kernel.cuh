#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

#include "common.h"
#include "gemm.cuh"
#include "online_softmax.cuh"

struct ForwardKernelArgs {
    using index_t = int64_t;
 
    void *__restrict__ Q; // Q位置
    void *__restrict__ K; //
    void *__restrict__ V;
    void *__restrict__ O;
 
    const index_t batch_stride;
    const index_t seq_stride;
    const index_t head_stride;
 
    const index_t seq_len;   // 序列长度 N
    const index_t n_heads;   // 头数 H
    const index_t n_samples; // batch 大小 B
    const index_t d_head;    // 每个 head 的维度 d
 
    const int n_Q_blocks;    // Q 方向 block 数
    const int n_KV_blocks;   // K/V 方向 block 数
};

namespace fa_warp {

template <typename value_t, typename index_t = int64_t>
struct GmemBlockPointers {
    value_t *gmem_Q;
    value_t *gmem_O;
    value_t *gmem_K;
    value_t *gmem_V;
};

template <typename value_t>
__device__ __forceinline__ float to_float(value_t v);

template <>
__device__ __forceinline__ float to_float<half>(half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename value_t>
__device__ __forceinline__ value_t from_float(float v);

template <>
__device__ __forceinline__ half from_float<half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// 先算 sample/head 基地址，再算 Q/O 与 K/V 的块偏移（与博客结构一致）
template <typename value_t, typename index_t = int64_t>
__forceinline__ __device__ constexpr GmemBlockPointers<value_t, index_t>
make_gmem_block_ptrs(const ForwardKernelArgs &args,
                     const int sample,
                     const int head,
                     const int q_seq_block,
                     const int Br = 64) {
    const index_t gmem_seq_stride = static_cast<index_t>(args.seq_stride);

    const index_t sample_head_offset =
        static_cast<index_t>(sample) * static_cast<index_t>(args.batch_stride) +
        static_cast<index_t>(head) * static_cast<index_t>(args.head_stride);

    // Q/O 每个 CTA 只处理一个 Q block
    const index_t QO_gmem_block_offset =
        sample_head_offset + static_cast<index_t>(q_seq_block) * static_cast<index_t>(Br) * gmem_seq_stride;
    // K/V 读取整段序列，从 sample/head 基地址开始
    const index_t KV_gmem_block_offset = sample_head_offset;

    GmemBlockPointers<value_t, index_t> ptrs;
    ptrs.gmem_Q = &reinterpret_cast<value_t *>(args.Q)[QO_gmem_block_offset];
    ptrs.gmem_O = &reinterpret_cast<value_t *>(args.O)[QO_gmem_block_offset];
    ptrs.gmem_K = &reinterpret_cast<value_t *>(args.K)[KV_gmem_block_offset];
    ptrs.gmem_V = &reinterpret_cast<value_t *>(args.V)[KV_gmem_block_offset];
    return ptrs;
}

// 相对基础版（未做 tensor core / double-buffer 等优化）FlashAttention forward。
// 仍然采用 FlashAttention 的块化与 online softmax 计算方式：
//   1) QK 分块点积
//   2) online softmax 累计
//   3) PV 累加
template <typename value_t, int Br = 64, int Bc = 64, int MAX_D = 128>
__global__ void flash_forward_kernel_unoptimized(const ForwardKernelArgs args) {
    const int sample = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq_block = blockIdx.x;

    const int lane_id = threadIdx.x;
    const int q_local_row = lane_id;
    if (q_local_row >= Br) {
        return;
    }

    const int q_row = q_seq_block * Br + q_local_row;
    if (q_row >= static_cast<int>(args.seq_len)) {
        return;
    }

    const int d_head = static_cast<int>(args.d_head);
    if (d_head > MAX_D) {
        return;
    }

    auto ptrs = make_gmem_block_ptrs<value_t>(args, sample, head, q_seq_block, Br);
    const int64_t gmem_seq_stride = static_cast<int64_t>(args.seq_stride);

    const value_t *q_ptr = ptrs.gmem_Q + static_cast<int64_t>(q_local_row) * gmem_seq_stride;
    const value_t *k_base = ptrs.gmem_K;
    const value_t *v_base = ptrs.gmem_V;
    value_t *o_ptr = ptrs.gmem_O + static_cast<int64_t>(q_local_row) * gmem_seq_stride;

    float O_accum[1][MAX_D];
    #pragma unroll
    for (int dh = 0; dh < MAX_D; ++dh) {
        O_accum[0][dh] = 0.0f;
    }

    const float softmax_scale = rsqrtf(static_cast<float>(d_head));
    float m[1] = {-INFINITY};
    float l[1] = {0.0f};
    float m_next[1] = {-INFINITY};
    float S_accum[1][Bc];

    // 主循环：遍历所有 K/V block
    for (int kv_block = 0; kv_block < args.n_KV_blocks; ++kv_block) {
        const int kv_start = kv_block * Bc;
        const int kv_end = min(kv_start + Bc, static_cast<int>(args.seq_len));
        const int kv_len = kv_end - kv_start;

        // 1) 计算当前 KV block 的 QK 分数
        #pragma unroll
        for (int kk = 0; kk < Bc; ++kk) {
            S_accum[0][kk] = -INFINITY;
        }

        for (int kk = 0; kk < kv_len; ++kk) {
            const int k_row = kv_start + kk;
            const value_t *k_ptr = k_base + static_cast<int64_t>(k_row) * gmem_seq_stride;

            // 使用 gemm.cuh 的优化点积路径（保持 GEMM 累加结构）
            float score = qk_dot_f32_accum<value_t, MAX_D>(q_ptr, k_ptr, d_head);
            S_accum[0][kk] = score;
        }

        // 2) online softmax（使用你提供的函数路径）
        scale_S_accum<1, Bc, float>(S_accum, softmax_scale);
        calc_row_max<false, 1, Bc, float>(S_accum, m_next, m);
        scale_l_O<1, MAX_D, float>(m_next, m, l, O_accum);
        exponentiate_tensor<1, Bc, float>(S_accum, m_next);
        update_row_exp_sum<1, Bc, float>(S_accum, l);

        // 3) PV 累加到 O_accum
        for (int kk = 0; kk < kv_len; ++kk) {
            const int k_row = kv_start + kk;
            const value_t *v_ptr = v_base + static_cast<int64_t>(k_row) * gmem_seq_stride;
            const float p = S_accum[0][kk];
            // 使用 gemm.cuh 的 PV FMA 累加路径
            pv_fma_accum<value_t, MAX_D>(O_accum[0], v_ptr, p, d_head);
        }
    }

    // 4) 最终归一化
    final_softmax_normalization<false, 1, MAX_D, float>(O_accum, l);

    #pragma unroll
    for (int dh = 0; dh < MAX_D; ++dh) {
        if (dh < d_head) {
            o_ptr[dh] = from_float<value_t>(O_accum[0][dh]);
        }
    }
}

template <typename value_t, int Br = 64, int Bc = 64>
inline cudaError_t launch_flash_forward_unoptimized(const ForwardKernelArgs &args,
                                                    cudaStream_t stream = 0) {
    dim3 block(Br, 1, 1);  // 每个线程处理一行 Q（基础版实现）
    dim3 grid(args.n_Q_blocks, args.n_heads, args.n_samples);
    flash_forward_kernel_unoptimized<value_t, Br, Bc><<<grid, block, 0, stream>>>(args);
    return cudaGetLastError();
}

} // namespace fa_warp
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace fa_warp {

// 保留与原 tensorcore 结构一致的常量命名，便于后续替换为 mma 指令路径
constexpr int MMA_M_FRAGMENTS_PER_ITER = 2;
constexpr int MMA_N_FRAGMENTS_PER_ITER = 1;
constexpr int MMA_K_FRAGMENTS_PER_ITER = 2;
constexpr int N_REGS_PER_F32_ACCUM_FRAGMENT = 2;

template <typename value_t>
__device__ __forceinline__ float gemm_to_float(value_t v);

template <>
__device__ __forceinline__ float gemm_to_float<half>(half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float gemm_to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

// 这个声明保留 tensorcore 结构入口。若未来接入真实 mma PTX，可直接提供定义。
template <typename value_t>
__device__ __forceinline__ void mma_m16n8k16_f32_accum(
    float &, float &, float &, float &,
    uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t,
    float, float, float, float);

template <typename value_t, const int M_fragments, const int N_fragments,
          const int K_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments],
    uint32_t (&regs_B)[N_fragments][K_fragments],
    accum_t (&regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT]) {
    #pragma unroll
    for (int k = 0; k < K_fragments; k += MMA_K_FRAGMENTS_PER_ITER) {
        #pragma unroll
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            #pragma unroll
            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2], regs_C[m + 1][n * 2 + 1],
                    regs_A[m][k], regs_A[m + 1][k],
                    regs_A[m][k + 1], regs_A[m + 1][k + 1],
                    regs_B[n][k], regs_B[n][k + 1],
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2], regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}

// 当前可直接落地的 QK 点积优化：
// - 保持“分块 + 累加”的 GEMM 结构
// - 使用 fmaf 与手工展开，便于编译器做向量化调度
template <typename value_t, int MAX_D = 128>
__device__ __forceinline__ float qk_dot_f32_accum(
    const value_t *q_ptr,
    const value_t *k_ptr,
    const int d_head) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    int d = 0;
    for (; d + 3 < d_head; d += 4) {
        const float q0 = gemm_to_float<value_t>(q_ptr[d + 0]);
        const float q1 = gemm_to_float<value_t>(q_ptr[d + 1]);
        const float q2 = gemm_to_float<value_t>(q_ptr[d + 2]);
        const float q3 = gemm_to_float<value_t>(q_ptr[d + 3]);

        const float k0 = gemm_to_float<value_t>(k_ptr[d + 0]);
        const float k1 = gemm_to_float<value_t>(k_ptr[d + 1]);
        const float k2 = gemm_to_float<value_t>(k_ptr[d + 2]);
        const float k3 = gemm_to_float<value_t>(k_ptr[d + 3]);

        acc0 = fmaf(q0, k0, acc0);
        acc1 = fmaf(q1, k1, acc1);
        acc2 = fmaf(q2, k2, acc2);
        acc3 = fmaf(q3, k3, acc3);
    }

    float sum = (acc0 + acc1) + (acc2 + acc3);
    for (; d < d_head; ++d) {
        sum = fmaf(gemm_to_float<value_t>(q_ptr[d]),
                   gemm_to_float<value_t>(k_ptr[d]), sum);
    }
    return sum;
}

// 当前可直接落地的 PV 累加优化：
// O += p * V(row)
template <typename value_t, int MAX_D = 128>
__device__ __forceinline__ void pv_fma_accum(
    float (&o_accum)[MAX_D],
    const value_t *v_ptr,
    const float p,
    const int d_head) {
    int d = 0;
    for (; d + 3 < d_head; d += 4) {
        o_accum[d + 0] = fmaf(p, gemm_to_float<value_t>(v_ptr[d + 0]), o_accum[d + 0]);
        o_accum[d + 1] = fmaf(p, gemm_to_float<value_t>(v_ptr[d + 1]), o_accum[d + 1]);
        o_accum[d + 2] = fmaf(p, gemm_to_float<value_t>(v_ptr[d + 2]), o_accum[d + 2]);
        o_accum[d + 3] = fmaf(p, gemm_to_float<value_t>(v_ptr[d + 3]), o_accum[d + 3]);
    }
    for (; d < d_head; ++d) {
        o_accum[d] = fmaf(p, gemm_to_float<value_t>(v_ptr[d]), o_accum[d]);
    }
}

} // namespace fa_warp
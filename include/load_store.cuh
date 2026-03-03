#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cutlass/cutlass.h>
#include <cutlass/arch/memory_sm75.h>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/tensor.hpp>
#include "common.h"

namespace fa_warp {
// 每个 fragment 为 8x8，GSM 每次迭代搬 4 行
template <typename T, int N>
inline __device__ void cute_async_copy(const T *g_ptr, T *s_ptr);

template <typename T>
struct GM2SM_async {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        // 使用 CUTE 的 SM80 cp.async 原子执行 16B 异步搬运
        constexpr int kElemsPerAccess = BYTES_PER_VEC4_ACCESS / static_cast<int>(sizeof(T));
        cute_async_copy<T, kElemsPerAccess>(gmem, smem);
    }
};

template <typename T>
struct SM2GM {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        // 直接按 16B（uint4）粒度从 SMEM 回写到 GMEM
        reinterpret_cast<uint4 *>(gmem)[0] = reinterpret_cast<uint4 *>(smem)[0];
    }
};

template <typename T, int N>
inline __device__ void cute_async_copy(const T *g_ptr, T *s_ptr) {
    using namespace cute;

    // 一维向量视图：长度为 N
    auto layout = make_layout(make_shape(Int<N>{}));
    auto g_tensor = make_tensor(make_gmem_ptr(const_cast<T *>(g_ptr)), layout);
    auto s_tensor = make_tensor(make_smem_ptr(s_ptr), layout);

    // 使用 CUTLASS/CUTE 的 SM80 cp.async 原子（16B）
    using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>;
    CopyAtom copy_atom;
    copy_atom.call(g_tensor, s_tensor);
}

template <typename op, /* either GM2SM_async or SM2GM */
          TensorLDSTConfig CFG,
          typename value_t,
          typename index_t = int64_t>
__forceinline__ __device__ constexpr void copy_block_GSM(
    value_t *gmem,
    value_t *smem,
    index_t gmem_seq_stride,
    const int lane_id) {
    // 一个 warp 在行方向分多轮搬运：每轮覆盖 4 行（GSM_LDST_ROWS_PER_ITER）
    constexpr int n_row_iters =
        CFG.GSM.row_fragments * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;

    // CUTLASS 风格分工：每轮 32 线程分摊 4 行 -> 每行 8 个线程处理列分块
    constexpr int col_fragments_per_iter = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    constexpr int col_fragments_per_row = CFG.smem_cols / COLS_PER_FRAGMENT;

    // lane_id 映射到本轮负责的 (row, col_fragment)
    const int thread_row = lane_id / col_fragments_per_iter;
    const int thread_col_fragment = lane_id % col_fragments_per_iter;

    #pragma unroll
    for (int r = 0; r < n_row_iters; ++r) {
        const int cur_row = r * GSM_LDST_ROWS_PER_ITER + thread_row;

        #pragma unroll
        for (int c = 0; c < col_fragments_per_row; c += col_fragments_per_iter) {
            const int col_fragment = c + thread_col_fragment;

            // 通过 op 抽象搬运方向：GMEM->SMEM 或 SMEM->GMEM
            op()(&gmem[cur_row * gmem_seq_stride +
                       col_fragment * COLS_PER_FRAGMENT],
                 &smem[cur_row * CFG.smem_cols +
                       col_fragment * COLS_PER_FRAGMENT]);
        }
    }
}

template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id,
    const int col_fragment_offset = 0) {
    // 与参考实现一致：每次用 ldmatrix.x4（CUTLASS 的 ldsm 封装）搬 2 个行分块
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;
    (void)col_fragments;

    // warp 内线程坐标映射：thread_row 负责行，thread_col_fragment 负责列分块
    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

    #pragma unroll
    for (int r = 0; r < CFG.RF.row_fragments; r += row_fragments_per_iter) {
        const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;

        #pragma unroll
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = thread_col_fragment + c + col_fragment_offset;
            value_t *src_ptr = &smem[cur_row * CFG.smem_cols +
                                     smem_col_fragment * ELEMS_PER_VEC4_ACCESS];

            // CUTLASS: ldmatrix.sync.aligned.x4.m8n8.shared.b16（非转置）
            cutlass::Array<unsigned, 4> frag;
            cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(frag, src_ptr);

            regs[r][c] = frag[0];
            regs[r + 1][c] = frag[1];
            regs[r][c + 1] = frag[2];
            regs[r + 1][c + 1] = frag[3];
        }
    }
}

template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id,
    const int row_fragment_offset = 0) {
    // 与参考实现一致：每次搬 2 个行分块（16 行）
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;
    (void)col_fragments;

    // warp 内线程坐标：行内偏移 + 列分块偏移
    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

    #pragma unroll
    for (int r = 0; r < CFG.RF.col_fragments; r += row_fragments_per_iter) {
        const int cur_row = thread_row + (r + row_fragment_offset) * ROWS_PER_FRAGMENT;

        #pragma unroll
        for (int c = 0; c < CFG.RF.row_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = thread_col_fragment + c;
            value_t *src_ptr = &smem[cur_row * CFG.smem_cols +
                                     smem_col_fragment * ELEMS_PER_VEC4_ACCESS];

            // CUTLASS: ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16（转置）
            cutlass::Array<unsigned, 4> frag;
            cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 4>(frag, src_ptr);

            regs[c][r] = frag[0];
            regs[c][r + 1] = frag[1];
            regs[c + 1][r] = frag[2];
            regs[c + 1][r + 1] = frag[3];
        }
    }
}

template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_RF2SM(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id) {
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT;
    constexpr int col_fragments_per_iter = 1;
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    (void)col_fragments;

    // 每个线程一次写回 2 个 fp16（即 1 个 uint32）
    constexpr int elems_per_store = 2;
    const int thread_row = lane_id / 4;
    const int thread_inner_col = (lane_id % 4) * elems_per_store;

    #pragma unroll
    for (int r = 0; r < CFG.RF.row_fragments; ++r) {
        const int cur_row = thread_row + r * rows_per_iter;

        #pragma unroll
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = c;

            reinterpret_cast<uint32_t *>(
                &smem[cur_row * CFG.smem_cols +
                      (smem_col_fragment * ELEMS_PER_VEC4_ACCESS +
                       thread_inner_col)])[0] = regs[r][c];
        }
    }
}
} 


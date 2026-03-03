#pragma once

#include <cstdint>
#include "common.h"
#include "load_store.cuh"

template <typename value_t, int N>
struct RFVector {
    static constexpr int size = N;
    value_t regs[N];

    __forceinline__ __device__ constexpr value_t &operator[](int idx) {
        return regs[idx];
    }
};

template <typename value_t, int stages, int row_fragments, int col_fragments>
struct RFMatrix {
    // ldmatrix 输出以 32-bit 寄存器表示
    using storage_t = uint32_t;
    static constexpr int rows = row_fragments;
    static constexpr int cols = col_fragments;
    static constexpr int n_stages = stages;

    storage_t regs[n_stages][rows][cols];

    __forceinline__ __device__ constexpr storage_t (&data(
        const int stage = 0))[rows][cols] {
        return regs[stage];
    }

    __forceinline__ __device__ constexpr void zero() {
        #pragma unroll
        for (int s = 0; s < n_stages; ++s) {
            #pragma unroll
            for (int r = 0; r < rows; ++r) {
                #pragma unroll
                for (int c = 0; c < cols; ++c) {
                    regs[s][r][c] = 0u;
                }
            }
        }
    }
};

template <TensorLDSTConfig ldst, typename value_t, typename index_t = int64_t>
struct MatrixLDST {
    // 该项目当前默认单 stage；后续可按需放开为多 stage
    static constexpr int mma_load_stages = 1;
    static constexpr bool transposed = ldst.transposed;

    using matrix_storage_t =
        RFMatrix<value_t, mma_load_stages, ldst.RF.row_fragments, ldst.RF.col_fragments>;
    using GM2SM_op = fa_warp::GM2SM_async<value_t>;
    using SM2GM_op = fa_warp::SM2GM<value_t>;

    // Runtime properties
    value_t *gmem_ptr;
    index_t gmem_seq_stride;
    // SMEM -> RF 读取位置
    value_t *smem_srm_ptr;
    // GMEM <-> SMEM 搬运位置
    value_t *smem_gsm_ptr;

    const int lane_id;
    matrix_storage_t storage;

    __forceinline__ __device__ MatrixLDST(value_t *gmem_block_ptr,
                                          index_t _gmem_seq_stride,
                                          value_t *_smem_ptr)
        : lane_id(threadIdx.x % WARP_SIZE) {
        const int warp_rank = threadIdx.x / WARP_SIZE;
        const index_t warp_seq = static_cast<index_t>(ldst.warp_ldst_rows) *
                                 static_cast<index_t>(warp_rank);

        gmem_seq_stride = _gmem_seq_stride;
        gmem_ptr = gmem_block_ptr + warp_seq * gmem_seq_stride;
        smem_gsm_ptr = _smem_ptr + warp_seq * static_cast<index_t>(ldst.smem_cols);
        smem_srm_ptr = ldst.compute_over_entire_block ? _smem_ptr : smem_gsm_ptr;
    }

    __forceinline__ __device__ constexpr void zero() { storage.zero(); }

    __forceinline__ __device__ constexpr typename matrix_storage_t::storage_t (&data(
        const int stage = 0))[matrix_storage_t::rows][matrix_storage_t::cols] {
        return storage.data(stage);
    }

    __forceinline__ __device__ constexpr void advance_gmem_block() {
        gmem_ptr += static_cast<index_t>(ldst.block_size) * gmem_seq_stride;
    }

    __forceinline__ __device__ constexpr void copy_GM2SM() {
        fa_warp::copy_block_GSM<GM2SM_op, ldst>(
            gmem_ptr, smem_gsm_ptr, gmem_seq_stride, lane_id);
    }

    __forceinline__ __device__ constexpr void copy_SM2GM() {
        fa_warp::copy_block_GSM<SM2GM_op, ldst>(
            gmem_ptr, smem_gsm_ptr, gmem_seq_stride, lane_id);
    }

    __forceinline__ __device__ constexpr void copy_SM2RF(int stage = 0,
                                                         int tile_offset = 0) {
        if constexpr (!transposed) {
            fa_warp::copy_warp_fragment_SM2RF<ldst, value_t>(
                storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        } else {
            fa_warp::copy_warp_fragment_transposed_SM2RF<ldst, value_t>(
                storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        }
    }

    __forceinline__ __device__ constexpr void copy_RF2SM() {
        fa_warp::copy_warp_fragment_RF2SM<ldst, value_t>(
            data(), smem_srm_ptr, lane_id);
    }
};
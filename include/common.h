#pragma once

// --------------------------------------------
// 通用常量
// --------------------------------------------
constexpr int WARP_SIZE = 32;
constexpr int ROWS_PER_FRAGMENT = 8;
constexpr int COLS_PER_FRAGMENT = 8;
constexpr int GSM_LDST_ROWS_PER_ITER = 4;
constexpr int BYTES_PER_VEC4_ACCESS = 16;  // 16B 向量访问（128bit）
constexpr int ELEMS_PER_VEC4_ACCESS = 8;   // fp16/bf16 下 16B = 8 个元素

// --------------------------------------------
// 搬运配置结构（作为 NTTP 使用）
// --------------------------------------------
struct TileLayout {
    int row_fragments;
    int col_fragments;
};

struct TensorLDSTConfig {
    // GMEM <-> SMEM 的分块形状
    TileLayout GSM;
    // SMEM <-> RF 的分块形状
    TileLayout RF;

    // 是否使用转置读（通常用于 K/V）
    bool transposed;
    // 一个 block 在序列维度覆盖的行数
    int block_size;
    // smem 中每行元素数（列跨度）
    int smem_cols;

    // 每个 warp 独立负责的行数（通常 = GSM.row_fragments * 8）
    int warp_ldst_rows;
    // 是否让该 warp 覆盖整个 block（K/V 常为 true）
    bool compute_over_entire_block;
};
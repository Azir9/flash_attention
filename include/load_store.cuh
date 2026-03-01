// 负责在 naive 版本中做一些索引计算（不显式用 shared memory）

#pragma once

namespace fa_naive {

// 按照 [B, N, H, d] 布局，计算 (b, i, h, 0) 的起始下标
inline __device__ int index_qkvd(int b, int i, int h,
                                 int B, int N, int H, int d) {
    // ((b * N + i) * H + h) * d
    return ((b * N + i) * H + h) * d;
}

// 按照 [B, N, H, d] 布局，计算 (b, j, h, 0) 的起始下标 (用于 K, V)
inline __device__ int index_kv(int b, int j, int h,
                               int B, int N, int H, int d) {
    return ((b * N + j) * H + h) * d;
}

// 按照 [B, N, H, d] 布局，计算输出 O[b, i, h, 0] 的起始下标
inline __device__ int index_o(int b, int i, int h,
                              int B, int N, int H, int d) {
    return ((b * N + i) * H + h) * d;
}

} // namespace fa_naive

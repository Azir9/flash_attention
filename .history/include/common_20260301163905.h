#pragma once
//定义常量

//编译：循环展开
#define FA_UNROLL _Pragma("unroll")
// 函数修饰，内联，放在device上面
#define FA_DEVICE __forceinline__ __device__
// 常量表达式 用于修饰函数（局限于device端）
#define FA_DEVICE_CONSTEXPR __forceinline__ __device__ constexpr
// gpu中的一个warp大小

//定义常量


// 基础常量（naive 实现里 Br/Bc 暂时没用，预留给后续 block 版本）
constexpr int Br = 32;       // Block Row (Q 的分块大小)
constexpr int Bc = 32;       // Block Col (K, V 的分块大小)
constexpr int d  = 64;       // Head Dimension
constexpr int WARP_SIZE = 32;
constexpr int SHFL_ENTIRE_WARP_MASK=0xffffffff; // warp  规约的掩码

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif
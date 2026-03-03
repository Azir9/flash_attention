# Flash Attention 实现

用于学习目的的 **Flash Attention** CUDA 实现，包含朴素版本与基于 CUTLASS 的 load/store 优化版本。

---

## 1. 项目结构

```
flash_attention/
├── include/           # 头文件
│   ├── common.h       # 常量、TileLayout、TensorLDSTConfig
│   ├── forward_kernel.cuh   # Kernel 参数、地址计算、forward 入口
│   ├── load_store.cuh       # GMEM↔SMEM↔RF 搬运（CUTLASS/CUTE）
│   ├── online_softmax.cuh   # Online Softmax（scale_S_accum、calc_row_max 等）
│   ├── tensor.cuh     # MatrixLDST、RFMatrix 抽象
│   ├── flash_attention.cuh  # 外部调用接口
│   └── gemm.cuh       # 矩阵乘 / 点积辅助
├── src/
│   ├── main.cu        # Demo 入口
│   └── flash_attention_naive.cu  # Naive 版 Kernel 实现
├── CMakeLists.txt
└── README.md
```

---

## 2. 核心算法

### Naive 版本

每个线程负责一个 `(batch, head, query_token)` 的整行输出：

1. **打分**：计算 Q 行与所有 K 行的点积，缩放 `1/√d`
2. **Softmax**：在 Shared Memory 中做稳定 Softmax
3. **加权求和**：用概率对 V 加权得到输出行 O

### 基础优化版（Online Softmax）

- 分块遍历 K/V，使用 **Online Softmax** 动态更新 m、l
- 支持 `scale_S_accum`、`calc_row_max`、`scale_l_O`、`exponentiate_tensor`、`update_row_exp_sum`、`final_softmax_normalization` 等函数

---

## 3. 编译与运行

### 环境要求

- CMake 3.18+
- CUDA Toolkit 11.0+
- C++17 编译器

### 构建

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80   # 按 GPU 架构调整（70/75/80/86）
cmake --build .
```

### 运行 Demo

```bash
./flash_attention_demo
```

程序会打印输出张量前若干元素，用于验证逻辑正确性。

---

## 4. 后续优化方向

- **分块与 Shared Memory**：将 Q/K/V 分块加载到 SMEM，减少 GMEM 访问
- **Warp 级并行**：利用 Warp 内协作做点积累加与矩阵乘
- **Tensor Core**：在 Ampere/Hopper 上使用 `mma` 加速
- **向量化访问**：`float4`/`uint4` 等向量化读写

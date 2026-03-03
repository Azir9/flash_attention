这份 README 文档旨在为你的 `flash_attention_naive` 项目提供清晰的结构说明、安装指南以及后续的性能优化方向。

---

# Flash Attention Naive Implementation

这是一个用于学习目的的 **Flash Attention** 极简/朴素（Naive）版本的 CUDA 实现。它通过将注意力机制的三个步骤（QK^T 乘法、Softmax、与 V 乘法）融合进一个单一的 CUDA Kernel，展示了 Flash Attention 的核心融合思想，尽管该版本尚未包含分块（Tiling）和共享内存（Shared Memory）的高级优化。

## 1. 项目结构

* **`include/`**: 包含头文件。
* `flash_attention.cuh`: 外部调用的接口定义。
* `forward_kernel.cuh`: CUDA Kernel 的声明。
* `online_softmax.cuh`, `gemm.cuh`, `load_store.cuh`: 包含 Softmax 计算、点积运算及索引计算等辅助函数。


* **`src/`**: 源代码。
* `flash_attention_naive.cu`: 核心逻辑实现，包含单线程处理一整行 Query 的逻辑。
* `main.cu`: 演示程序，构造测试数据并验证 Kernel 调用。


* **`CMakeLists.txt`**: 构建脚本，支持 CUDA 编译配置。

## 2. 核心算法逻辑

在 `flash_attention_forward_kernel_naive` 中，我们针对每一个 Batch、每一头、每一个 Query Token ($b, h, i$) 分配一个线程：

1. **计算打分 (Logits)**: 计算 $Q_i$ 与所有 $K_j$ 的点积，并缩放 $1/\sqrt{d}$。
2. **稳定 Softmax**: 在 Shared Memory 中存储 Logits，执行减去最大值的 Softmax 运算。
3. **加权求和**: 使用计算出的概率对 $V$ 矩阵进行加权，得到最终的输出行 $O_i$。

## 3. 编译与运行

### 环境要求

* CMake 3.18+
* CUDA Toolkit 11.0+
* 支持 C++17 的编译器

### 构建步骤

```bash
mkdir build && cd build
# 默认 GPU 架构为 sm_70，可通过 CMAKE_CUDA_ARCHITECTURES 指定
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 
make

```

### 运行 Demo

```bash
./flash_attention_demo

```

该程序会打印形如 `[1, 4, 1, 4]` 形状的张量计算结果，用于快速验证逻辑。

---

## 4. 性能优化建议

当前版本虽然实现了“算子融合”，但在高性能计算（HPC）场景下存在明显的瓶颈。以下是针对该项目的优化路径：

### A. 存储层级优化 (Tiling & Shared Memory)

* **分块计算 (Tiling)**: 目前线程需要遍历完整的 $N$ (Sequence Length)，当 $N$ 较大时，无法完全放入 Shared Memory。应引入分块策略，将 $Q, K, V$ 切分为小的 Block，分批次加载到 Shared Memory 中。
* **减少全局内存访问**: 当前实现在计算点积和加权和时多次从 Global Memory 读取 $K$ 和 $V$。优化后应确保每个数据块只从 Global Memory 加载一次。

### B. 计算并行度优化

* **线程粒度**: 当前“一个线程负责一行”的配置导致 GPU 硬件利用率极低（Warp 内只有一个线程工作）。应改用 **Thread Block** 负责一行或一个分块，利用 Warp 级别的并行执行点积累加（Reduce）和矩阵乘法。
* **Online Softmax**: 引入 FlashAttention 论文中提到的 Online Softmax 算法（Milakov et al.），在分块加载 $K, V$ 的过程中动态更新最大值和累加和，从而避免存储整个 $N$ 维的 Logits 向量。

### C. 指令级与硬件优化

* **Tensor Core 加速**: 在支持的架构（如 Ampere/Hopper）上，使用 `mma` 指令或半精度（FP16/BF16）来加速核心的矩阵乘法运算。
* **向量化读写**: 使用 `float4` 等向量化指令进行数据加载和存储，以填满内存带宽。
* **合并访问 (Coalesced Access)**: 调整内存布局，确保线程束内的线程在访问 $Q, K, V$ 时能够满足合并访问条件。

### D. 静态常量优化

<<<<<<< HEAD
* **模板化**: 将 $d$ (Head Dim) 作为模板参数传入，利用编译器优化循环展开（Loop Unrolling），减少分支预测开销。


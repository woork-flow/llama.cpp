# 第 6 章:走进 ggml

阶段目标:理解"一张计算图怎么变成 GPU 上的 kernel"。这一章只有在你需要**改性能**、**加硬件后端**、**加新的量化类型**或**调试数值精度**时才需要。

如果你只是用 llama.cpp 做应用,可以跳过。

## ggml 是什么

一个极简的张量库。它和 PyTorch 的根本区别:

| | PyTorch | ggml |
|---|---|---|
| 执行方式 | eager,逐条执行 | 先建整张图,再一次性执行 |
| 内存 | 动态分配 | 图建好后一次性规划复用(arena) |
| 算子 | 几百个 | 几十个基础算子,复杂算子靠组合 |
| 依赖 | 巨大 | 纯 C,无依赖 |
| 目标 | 训练 | 推理 |

这就是为什么 llama.cpp 一个仓库能跑在从树莓派到 8 卡 H100 的所有设备上。

## 三个核心头文件

`ggml/include/` 下:

- **`ggml.h`** - 张量(`ggml_tensor`)和图(`ggml_cgraph`),以及所有算子声明。**先读这个**
- **`ggml-backend.h`** - 后端接口。CPU、CUDA、Metal 都是它的实现
- **`ggml-alloc.h`** - 内存规划器

## 阅读顺序

### 1. `ggml/include/ggml.h` 的 `ggml_tensor`

```c
struct ggml_tensor {
    enum ggml_type type;     // F32, F16, Q4_0, Q4_K, ...
    int64_t ne[4];           // 各维长度,4 维够用
    size_t  nb[4];           // 各维字节跨度
    void *  data;
    struct ggml_tensor * src[10];  // 上游节点,建图连接的体现
    ...
};
```

理解两件事:

**维度是"反的"。** 一个 `{n_embd, n_ff}` 的张量,`ne[0] = n_embd` 是最内层、连续存放的那一维。这和数学书上的行优先不一样,和 PyTorch 的 `[out, in]` 也相反。看 `src/models/*.cpp` 时最容易在这里出错。

**`src[]` 就是图。** ggml 的图不是独立结构,而是通过这些指针反向遍历出来的拓扑。

### 2. `ggml/src/ggml.c` 的 op 定义(8115 行)

按这个模式跳着读:

```c
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    // 1. 检查维度合法性
    // 2. ggml_new_tensor 创建结果张量
    // 3. 填充 op 字段和 src 指针 <<< 关键
    // 4. 返回
}
```

注意第 3 步:**只建节点,不做计算**。所有 `ggml_*` 函数都是这个模式。真正执行在 `ggml_graph_compute`。

这一步还能学到不少隐形规则,比如哪些形状能广播、哪些必须是连续的。

### 3. `ggml/src/ggml-backend.cpp`(2991 行)- 后端抽象

关键概念:

- `ggml_backend` - 一个执行设备(CPU/CUDA0/CUDA1...)
- `ggml_backend_buffer` - 一个设备上的内存池
- `ggml_backend_sched` - **调度器**,最值得理解的部分

调度器干什么:给它一张图,它按每个张量"住在哪个设备"把图切成若干段,插入跨设备拷贝,然后依次派发给对应后端。`src/llama-context.cpp` 里的 `graph_compute` 调的就是它。

这就是"张量存 CPU 但部分层在 GPU"这种混合卸载的实现基础。

### 4. `ggml/src/ggml-alloc.c`(1170 行)- 内存规划

原理很简单:按图的拓扑顺序算每个张量什么时候被创建、什么时候最后一次被使用,然后让"死掉"的区间重叠复用同一块内存。

理解了这个,你就明白为什么 llama.cpp 会"建图失败 / 显存不够"——因为它是按**峰值**一次性分配的,不是按需。降低 `-ub` 能降峰值就是这个原因。

### 5. 具体后端

| 目录 | 说明 |
|---|---|
| `ggml/src/ggml-cpu/` | CPU 实现,重点看 `quants.c`(量化)和 `repack.cpp`(布局重排) |
| `ggml/src/ggml-cuda/` | CUDA,73 个 `.cu` 文件,一个算子一个 |
| `ggml/src/ggml-metal/` | Apple Metal |
| `ggml/src/ggml-vulkan/` | Vulkan,跨平台 GPU |
| `ggml/src/ggml-rpc/` | 把远程机器当成一个后端,`tools/rpc` 是配套服务端 |

新后端从 `ggml-backend-impl.h` 的接口开始看:要实现 buffer 类型、device 接口、以及一批 `compute_forward_*`。

### 6. 量化

这是 llama.cpp 能跑起来的关键。相关文件:

- `ggml/src/ggml-quants.c`(5665 行)- 量化/反量化的参考实现,纯 C,可读性最好
- `ggml/src/ggml-common.h` - 各种 block 结构体的定义(比如 `block_q4_K`)
- `ggml/src/ggml-cpu/quants.c` - 带 SIMD 的优化版
- `tools/quantize/` - 量化工具

想搞懂 Q4_K_M 里那个 "K" 是什么,从 `ggml-common.h` 的结构体定义入手,比读文档快。对应测试:`tests/test-quantize-fns.cpp`。

## 怎么验证你读懂了

**改一个后端 op 然后跑对照测试。** 这是这个仓库最靠谱的验证方式:

```sh
build/bin/test-backend-ops            # 遍历所有 op,对照 CPU 参考实现
build/bin/test-backend-ops -o MUL_MAT # 只测矩阵乘
```

这个工具会拿每个 op 在所有后端上跑一遍,和历史参考值比。加新后端或改 kernel 之后必须跑它。

## 性能分析

- `docs/build-profiling.md` - 官方的 profiling 说明
- `docs/ops.md` + `docs/ops/*.csv` - 每个后端支持哪些 op,一张现成的能力表
- `build/bin/llama-bench` - 端到端基线,改完必跑
- `ggml/src/ggml-et/` - 内置的轻量追踪后端,可以看每个 op 的耗时

## 学习建议

这一章不要按顺序通读,建议按"一个问题"驱动:

- 想搞懂量化 -> 只读 `ggml-common.h` + `ggml-quants.c` + `test-quantize-fns`
- 想加后端 -> 只读 `ggml-backend-impl.h` + 一个最简单的后端(比如 `ggml-blas/`)+ `test-backend-ops`
- 想优化某个算子 -> 只读 `ggml-cpu/ops.cpp` 里那一个 op,用 `llama-bench` 和 `test-backend-ops` 验证
- 想理解显存 -> 只读 `ggml-alloc.c` + `ggml-backend.cpp` 的 scheduler 部分

大规模通读 ggml 的收益很低,它是一层为了移植性牺牲了抽象美的代码。
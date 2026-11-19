# 第 7 章:过时概念对照表和常见坑

这一章建议在动手读代码**之前**先扫一遍。llama.cpp 重构过很多次,网上(包括很多 AI 的"记忆")的资料大量过时,照着老文章找代码会白费时间。

## 你所在的版本

- tag:`v1.0.0`,本地分支 `v1.0.0-learn`
- 从 v1.0.0 开始项目采用语义化版本,并且明确规定:**破坏 `include/llama.h` 公开 C API 的改动才升 MAJOR**(见 `docs/release.md`)
- 这意味着公开 API 学习成本最低、最稳定;**内部实现则不保证**(`src/` 下的类随时可能重构)

所以学习策略应该是:公开 API 花力气记牢,内部实现只记结构不记细节。

## 过时概念对照

### 函数和类

| 老资料里的写法 | v1.0.0 里实际是 |
|---|---|
| `llama_load_model_from_file` | `llama_model_load_from_file` |
| `llama_new_context_with_model` | `llama_init_from_model` |
| `llama_build_graph()` 一个巨型函数 | `llm_graph_context` 子类 + `build_arch_graph()` |
| `llama_decode_internal()` | `llama_context::decode()` |
| `llama_model_load_internal()` | `llm_graph_context` 的 `load_arch_hparams` / `load_arch_tensors` |
| `struct llama_kv_cache` 直接访问 | `llama_memory_i` 接口 + 多个实现 |
| `kv_self` 字段 | 已不存在,通过 `ctx->memory` 访问 |
| `llama_kv_cache_unified` | 已拆成 `llama-kv-cache.cpp` / `-iswa` / `-dsa` / `-recurrent` / `-hybrid` |
| `llama_sample_*` 系列函数 | `llama_sampler_init_*` + 采样器链 |
| 整个前向写在一个 `llama.cpp` 里 | `src/models/` 下 176 个模型文件 |
| `ggml_tensor->n_dims` / `ne[]` 语义 | `ne[]` 仍在,但维度顺序务必确认(最内层是 `ne[0]`) |

### 概念

| 老概念 | 现在 |
|---|---|
| prompt 处理和生成是两套代码 | 都是 `llama_decode`,区别只是 batch 大小 |
| KV cache 就是一个数组 | 抽象成"记忆模块",有滑窗、稀疏、循环状态等多种 |
| 分词器只有 SPM 和 BPE | 8 种(`LLAMA_VOCAB_TYPE_*`) |
| 采样器是一个函数 | 可以串成链,还能搬进计算图 |
| 聊天模板是硬编码的 C++ 字符串 | 优先用 GGUF 里存的 Jinja 模板,自研 Jinja 引擎 |
| 输出解析用正则 | 主用 PEG 语法 `common/chat-peg-parser.cpp` |
| llama.cpp 用 minja | **不是**。`common/jinja/` 是自己实现的 Jinja 解释器 |

## 常见坑

### 1. 维度顺序搞反

`create_tensor(..., {n_embd, n_ff}, 0)` 里,`n_embd` 是最内层。PyTorch 里线性层权重是 `[out_features, in_features]`,而 GGUF 里存的是 `[in, out]`(按 GGML 的 `ne[0]` 优先)。从 PyTorch 代码对照过来时,这一条错得最多。

### 2. 以为 `ggml_add()` 会做加法

ggml 的所有算子函数只是**往图里加节点**。真正的计算在 `ggml_backend_sched_graph_compute`。所以你在 `build_arch_graph` 里加 `printf` 是看不到任何数值的。

### 3. `logits_all = true` 导致速度暴跌

默认只算最后一个位置的 logits。一旦打开 `embeddings`、`logits_all` 或者设置 `n_outputs`,计算量会成倍增长。这是新手最常见的"为什么比别人的慢"的原因。

### 4. 忘了 `llama_sampler_accept`

生成循环里如果只调 `sample` 不调 `accept`,重复惩罚、DRY 这些依赖历史的采样器就永远不生效。而且不会报错,只是效果不对,非常难查。

### 5. KV cache 满了返回 1 不是错误

`llama_decode` 返回 1 表示"KV cache 没位置了",不是崩溃。正常做法是 `llama_memory_seq_rm` 腾位置再重试。返回负数才是真错误。

### 6. `-b` 和 `-ub` 搞混

`-b` 是一次 `llama_decode` 的上限,`-ub` 是实际一次进计算图的上限。`-ub` 太大显存会爆,太小会反复建图变慢。

### 7. 直接改 master 上的代码去提 PR

这个仓库对改动极其挑剔,大改动不先开 issue 讨论基本会被关。学习阶段建议只在自己的 fork 上练习。仓库根目录的 `AGENTS.md` 和 `skills/code-review/SKILL.md` 是提交前的必读。

### 8. 从网上抄 `llama.h` 的代码

v1.0.0 之后公开 API 才稳定下来。2024 年那些博客里的代码大部分编译不过,因为函数名全改了。以 `include/llama.h` 和 `examples/` 为准。

## 遇到问题时的排查顺序

1. **看现象**:命令行工具能不能复现?加 `--verbose`。
2. **看返回码**:`llama_decode` 的返回值区分 1(没 slot)、负数(真错误)。
3. **看测试**:`tests/` 里有没有对应的 test 文件?直接跑一遍。
4. **看日志**:LLAMA_LOG_DEBUG 级别的日志里通常已经写明了原因。
5. **再读代码**:到这一步才需要钻源码。

## 有用的命令速查

```sh
# 编译(带调试符号,便于 gdb)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=OFF
cmake --build build -j

# 跑测试
ctest --test-dir build --output-on-failure
build/bin/test-backend-ops

# 看模型信息
build/bin/llama-gguf model.gguf
build/bin/llama-template-analysis -m model.gguf

# 性能
build/bin/llama-bench -m model.gguf
build/bin/llama-bench -m model.gguf -ngl 0,10,20,99   # 一次跑多个配置
```

## 官方资源索引

| 想了解 | 看哪里 |
|---|---|
| 怎么编译 | `docs/build.md` |
| 怎么加模型 | `docs/development/HOWTO-add-model.md`、`skills/add-new-model/` |
| 怎么解析输出 | `docs/development/parsing.md`、`docs/autoparser.md` |
| Jinja 支持范围 | `common/jinja/README.md` |
| 调试测试 | `docs/development/debugging-tests.md` |
| 性能分析 | `docs/build-profiling.md` |
| 后端支持哪些 op | `docs/ops.md`、`docs/ops/*.csv` |
| server 的能力边界 | `tools/server/README.md`、`tools/server/README-dev.md` |
| 提交前自查 | `skills/code-review/SKILL.md` |
| 切一个小模型来测 | `skills/slice-model/SKILL.md` |
# 第 3 章:跟一条 decode 路径

阶段目标:搞懂一次 `llama_decode` 内部到底发生了什么。这是整个引擎的中枢,读懂它,其他模块都只是它的配件。

## 不要线性读

`src/llama-context.cpp` 有 5185 行。这个文件里塞了 context 的全部职责:显存管理、batch 拆分、图调度、状态保存、embedding 输出、优化器接口……从头读到尾一定会迷路。

正确做法:手里拿一个调用栈,跳着读。下面是完整的一条路。

## 调用栈

```
llama_decode                           src/llama-context.cpp:5090   纯 C 接口,转发
  llama_context::decode                src/llama-context.cpp:2140   中枢
    1. balloc->init()                  src/llama-batch.cpp          把 batch 规范化
    2. memory->init_batch()                                          问 KV cache 要 slot
    3. mctx->get_ubatch() 循环          src/llama-memory*.cpp        切成 ubatch
       llama_context::process_ubatch   src/llama-context.cpp:1817   每个 ubatch 一次
         model.build_graph(gparams)    src/models/*.cpp             拼计算图
         res->set_inputs(&ubatch)                                   灌输入数据
         llama_context::graph_compute  src/llama-context.cpp:3123   跑图
           ggml_backend_sched_graph_compute_async
    4. output_reserve() / 拷 logits                                  取出结果
```

## 逐段解释

### 第一步:`balloc->init()` - batch 规范化

你在外面构造的是一个平平无奇的 `llama_batch`:一串 token id,加上每个 token 属于哪个序列、在什么位置。

`llama_batch_allocr`(`src/llama-batch.cpp`,1601 行)把它整理成内部格式,同时做校验:序列 id 合法吗?位置连续吗?要输出多少个 logits?

这一层最重要的是分清两个概念:

- **batch**:你一次调用 `llama_decode` 传进来的全部 token,上限是 `n_batch`
- **ubatch**:引擎真正一次送进计算图的 token,上限是 `n_ubatch`

一个 batch 会被切成多个 ubatch。为什么?因为显存有限。如果你设 `-b 2048 -ub 128`,一个 2048 token 的 prompt 会被切成 16 次计算。每次都要重新走一遍图构建和调度,所以速度明显下降。这就是第 1 章那个练习的原因。

切分策略在 `src/llama-batch.h` 里:

```cpp
llama_ubatch split_simple(uint32_t n_ubatch);   // 简单顺序切,最常见的路径
llama_ubatch split_equal(uint32_t n_ubatch, ...); // 多序列均分,不按顺序
llama_ubatch split_seq(uint32_t n_ubatch);      // 按序列切
```

### 第二步:`memory->init_batch()` - 找位置

这是 v1.0.0 里最值得注意的抽象。老版本里 KV cache 是写死的,新版本抽象成了 `llama_memory_i`,因为现在的模型有各种奇怪的记忆结构:

| 文件 | 对应什么模型 |
|---|---|
| `llama-memory.cpp` | 基类 + 工厂 |
| `llama-kv-cache.cpp` | 普通 Transformer 的全量 KV cache,最基础,先看这个 |
| `llama-kv-cache-iswa.cpp` | 滑窗注意力(局部窗口) |
| `llama-kv-cache-dsa.cpp` | DeepSeek 稀疏注意力 |
| `llama-memory-recurrent.cpp` | Mamba/RWKV 这类循环状态,没有 KV |
| `llama-memory-hybrid.cpp` | 上面几种混合(比如一层注意力一层 Mamba) |

`init_batch` 返回一个 `llama_memory_context_i`,它做两件事:为这次 batch 分配 KV cache 的槽位,以及把 batch 切成 ubatch。

返回值里有 `LLAMA_MEMORY_STATUS_FAILED_PREPARE`,意思是"cache 满了,腾不出位置"。此时 `decode` 会先试着做一次 fragment 整理(`memory_update(true)`)再重试,还不行就返回 1。这就是上下文超长时你看到的那个返回码。

### 第三步:`process_ubatch()` - 建图 + 跑图

每个 ubatch 走一次,分三小步:

**3.1 建图** `model.build_graph(gparams)`

这里有一层很实用的优化:**图复用**。如果当前 ubatch 的形状和上次完全一样,而且拓扑没变(`res->can_reuse(gparams)`),就直接复用上次的 `ggml_cgraph`,跳过建图。所以连续生成 token 时(每次都是 1 个 token),图只建一次,后面全是复用。这是为什么 tg 速度能远高于 pp。

图是由 `llm_graph_context` 的子类拼出来的,具体见第 4 章。

**3.2 分配 + 灌数据** `ggml_backend_sched_alloc_graph` + `res->set_inputs(&ubatch)`

`set_inputs` 把 token id、位置、序列 id 填进图的输入张量。注意 `src/llama-graph.cpp` 里那一堆 `llm_graph_input_*` 类,每个负责一类输入(attn 的 KV、position、rope 等)。

**3.3 跑图** `graph_compute()`

```cpp
ggml_backend_sched_graph_compute_async(sched.get(), gf);
```

`sched` 是 `ggml_backend_sched`,它把一张图按张量所在的设备切成若干段,分别派给 CPU/GPU 后端执行,并处理跨设备的拷贝。这是 ggml 和 llama 的接缝处。

### 第四步:取结果

`process_ubatch` 返回后,`decode` 累积 `n_outputs`,最后一个 ubatch 结束后把 logits 拷到 `ctx->output`。你之后调 `llama_get_logits_ith` 拿到的就是这里的东西。

## 三个最需要理解的概念

**1. 序列(seq_id)。** 一个 context 里可以同时跑多条独立的对话,靠 `seq_id` 区分。这就是 server 里多个 slot 并发生成的原理。KV cache 里的每个位置都标着属于哪个序列。

**2. 位置(pos)。** 不是数组下标,是逻辑位置。多序列共享上下文时,不同序列的位置可能重叠。这也解释了为什么会有 `llama_memory_seq_rm/seq_cp/seq_keep` 这些 API。

**3. 因果性(causal)。** 普通 LLM 是因果注意力,位置 p 只能看到 p 之前。所以 ubatch 可以随便切,先算前面的 token 不影响结果。但 embedding 和 encoder 类模型需要整段一次算完(代码里 `embd_all` 那个判断就是干这个的)。

## 动手练习

1. 在 `llama_context::process_ubatch` 里打开那两行被注释掉的 `//LLAMA_LOG_DEBUG("reusing previous graph\n")` 和 `graph build time`,用 debug 编译跑一次,观察 prefill 阶段和生成阶段建图次数的区别。
2. 加一行日志打印每个 ubatch 的 `n_tokens`,用 `-b 512 -ub 128` 跑长 prompt,验证切分。
3. 把 `-c` 设成很小(比如 64)跑长对话,观察 `LLAMA_MEMORY_STATUS_FAILED_PREPARE` 那条路径。
4. 用 `llama_get_logits_ith` 打印 logits 的前 5 个值,对照 `llama_sampler_sample` 的结果,理解采样器怎么用的。

## 相关文件速查

| 文件 | 行数 | 内容 |
|---|---|---|
| `src/llama-context.cpp` | 5185 | context 全部职责,decode 在 2140 |
| `src/llama-batch.cpp` | 1601 | batch 规范化 + ubatch 切分 |
| `src/llama-memory.cpp` | 144 | 记忆模块基类,很短,建议先读完 |
| `src/llama-kv-cache.cpp` | 3130 | 标准 KV cache 实现 |
| `src/llama-graph.cpp` | 4639 | 图的输入输出和一些公共构件 |
| `src/llama-graph.h` | - | `llm_graph_context` 定义,信息量很大 |

`src/llama-memory.cpp` 只有 144 行,是理解整个记忆抽象的最快入口。
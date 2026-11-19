# 第 2 章:读懂公开 API

阶段目标:搞清楚这个库对外提供什么,以及一次完整推理循环长什么样。

## 先读 `include/llama.h`

1737 行,是项目对外的全部契约。不用记住每个函数,重点是认出这几个"对象":

| 对象 | 是什么 | 关键函数 |
|---|---|---|
| `llama_model` | 权重,只读,可被多个 context 共享 | `llama_model_load_from_file`、`llama_model_free` |
| `llama_context` | 一次推理会话的状态,独占 GPU 显存 | `llama_init_from_model`、`llama_decode`、`llama_free` |
| `llama_sampler` | 采样器,可以串成链 | `llama_sampler_chain_init`、`llama_sampler_sample` |
| `llama_batch` | 一次要处理的 token | `llama_batch_init`、`llama_batch_get_one` |
| `llama_memory` | KV cache 的抽象(老版本没有这层) | 一般通过 context 间接用 |
| `llama_vocab` | 分词器 | `llama_tokenize`、`llama_token_to_piece` |

### 记住生命周期

```
llama_backend_init()                    // 全局初始化,进程级
  llama_model_load_from_file()           // 加载 GGUF
    llama_init_from_model()              // 建 context,分配 KV cache
      loop {
        llama_decode(ctx, batch)         // 前向一次
        llama_sampler_sample(smpl, ...)  // 采样出下一个 token
        llama_sampler_accept(smpl, tok)  // 让采样器记住这个 token
      }
    llama_free(ctx)
  llama_model_free(model)
llama_backend_free()
```

一个 model 可以配多个 context,这在 server 里就是这样用的(多个 slot 共享一份权重,但各自有 KV cache)。

## 再读 `examples/simple/simple.cpp`

238 行,最小的完整推理程序。按这个顺序读:

1. `main()` 开头解析参数,注意 `llama_model_params` 和 `llama_context_params` 两个结构体的字段,它们对应命令行参数。
2. 第 98 行 `llama_model_load_from_file`,加载 GGUF。
3. 第 128 行 `llama_init_from_model`,建 context。注意 `n_ctx` 和 `n_batch` 的关系。
4. 第 178 行开始的 main loop,这是核心:
   - 第一轮用整段 prompt 调 `llama_decode`(prefill)
   - 之后每轮只喂 1 个 token(自回归生成)
   - 每轮 `llama_sampler_sample(smpl, ctx, -1)`,这里的 `-1` 表示"取最后一个位置的 logits"
5. 最后 `llama_batch_free`、`llama_sampler_free`、`llama_free`、`llama_model_free`,注意释放顺序和申请顺序相反。

## 三个容易忽略但很重要的点

**prefill 和 decode 是同一条路。** 很多人以为 prompt 处理和逐 token 生成是两套代码。实际上都是 `llama_decode`,区别只是 batch 里塞了多少 token。引擎内部会根据 token 数走不同优化路径。

**`llama_decode` 是同步的但内部有异步。** 它提交计算图到后端调度器,然后等结果。CPU 上看起来是同步执行,GPU 上内部是异步提交。

**logits 不是免费的。** `llama_context_params.logits_all = false`(默认)时,只有最后一个 token 的位置算 logits。一旦你打开 embeddings 或 logits_all,计算量和输出拷贝量都会暴涨。这是新手最容易踩的性能坑。

## 动手练习

1. 把 `examples/simple/simple.cpp` 改成:读标准输入的多轮对话,每轮保留历史。体会 `llama_decode` 只喂新 token 的意义。
2. 用 `llama_tokenize` 手动把 prompt 转成 token 数组,再自己构造 `llama_batch`,替代 `llama_batch_get_one`。
3. 把采样器链改成 `top_k + top_p + temp + dist`,对比不同参数下输出的稳定性。
4. 试试 `llama_model_params.n_gpu_layers = 0` 和 `99`,看 `llama_model_load_from_file` 的耗时变化。

## 延伸阅读

- `common/common.h` - 把上面这堆样板代码包好了,`tools/` 下的所有工具都基于它
- `common/arg.cpp` - 所有命令行参数的真实定义,查某个参数作用时最准
- `include/llama-cpp.h` - 给 C++ 用的 RAII 包装,建议自己的项目照这个写
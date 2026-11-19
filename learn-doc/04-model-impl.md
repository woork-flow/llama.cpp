# 第 4 章:看一个模型怎么实现

阶段目标:从最经典的 Llama 入手,搞懂一个模型在 llama.cpp 里需要写什么、这些东西分别对应 GGUF 里的哪些数据。读完你就能看懂任意一个新模型文件在干什么。

## `src/models/` 有多少东西

176 个 `.cpp` 文件,每个对应一个模型家族。不要被这个数字吓到,它们全都是同一套骨架的不同填法。

挑 `src/models/llama.cpp` 读,只有 236 行,而且最干净。

## 一个模型类要实现三件事

```cpp
// src/models/models.h 里定义,每个模型一个类
struct llama_model_llama : public llama_model_base {
    void load_arch_hparams(llama_model_loader & ml) override;  // 1. 读超参
    void load_arch_tensors(llama_model_loader & ml) override;  // 2. 建张量
    std::unique_ptr<llm_graph_context> build_arch_graph(...) const override; // 3. 拼前向图
};
```

这三件事对应了"模型"这个概念的三层含义:配置文件、权重矩阵、计算流程。

### 1. `load_arch_hparams` - GGUF 里的 key 决定超参

`src/models/llama.cpp` 第 3 行开始:

```cpp
ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
```

`LLM_KV_*` 这些枚举在 `src/llama-arch.h` 里,值是 GGUF 里的字符串名字(比如 `llama.attention.layer_norm_rms_epsilon`)。这层间接是有意设计的:GGUF 里的标准名字 -> 内部枚举 -> 模型代码,这样换 HuggingFace 的元数据命名时只改映射表,不动模型代码。

同一个文件里还有一堆按 `n_layer` 猜型号的 switch(1B/3B/8B/13B...)。这只影响显示和默认参数,不影响计算,可以跳过。

### 2. `load_arch_tensors` - 权重怎么命名、张量多大

这部分信息量最大,但逻辑很直白:

```cpp
tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
```

要点:

- `tn()` 生成 GGUF 里的张量名,`LLM_TENSOR_*` 同样是 `src/llama-arch.h` 里的枚举。
- 维度顺序是 `{n_embd, n_ff}` 这种"列优先"写法,和 GGML 的内存布局一致,和 PyTorch 的 `[out, in]` 是反的。看代码时经常在这里搞混。
- `TENSOR_NOT_REQUIRED` 表示这个张量可以缺失(比如 `lm_head` 和词嵌入共享权重时)。缺失后代码会 fallback,见上面那段 `if (output == NULL)`。
- `TENSOR_DUPLICATED` 表示这个张量被多层共用,只在第 0 层真实分配。
- MoE 的分支用 `n_expert == 0` 判断,普通 FFN 和专家 FFN 是两套张量。

**关键理解**:这一步只是登记"我要哪些张量、多大、可不可以缺",真正的数据是 `src/llama-model-loader.cpp` 按名字从 GGUF 文件里 mmap 进来的。所以这里写错一个名字,加载时就会报 "missing tensor"。

### 3. `build_arch_graph` - 前向逻辑

```cpp
std::unique_ptr<llm_graph_context> llama_model_llama::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<false>>(params);
}
```

注意这个写法:它不返回一张图,而是返回一个 `llm_graph_context` 子类的实例,**构造这个对象的过程就是拼图的过程**。构造函数跑完,图就拼好了。

`graph<embed>` 那个 `embed` 模板参数区分"生成模型"和"embedding 模型":embedding 模型不需要 lm_head,也不需要 KV cache(整段一次算完)。同一个文件复用了两套。

### 前向代码逐行看

构造函数里就是标准的 Transformer:

```
inpL = build_inp_embd(model.tok_embd)      // 词嵌入查表
inp_pos = build_inp_pos()                   // 位置输入
inp_attn = build_attn_inp_kv()              // KV cache 输入句柄

for il in 0..n_layer:
    cur = build_norm(inpL, attn_norm, RMS)     // pre-norm
    Q,K,V = build_qkv(layer, cur, ...)         // QKV 投影
    Q = build_rope(Q, inp_pos, il)             // 旋转位置编码
    K = build_rope(K, inp_pos, il)
    cur = build_attn(inp_attn, wo, Q, K, V, ...)  // 注意力 + 输出投影
    ffn_inp = cur + inpL                          // 残差
    cur = build_norm(ffn_inp, ffn_norm, RMS)
    cur = build_ffn(...) 或 build_moe_ffn(...)    // FFN 或 MoE
    cur = cur + ffn_inp                           // 残差
    inpL = cur

cur = build_norm(cur, output_norm, RMS)       // final norm
res->t_logits = build_lora_mm(model.output, cur)  // lm_head,输出 logits
```

几个要点:

**这些都是"构造图"不是"算数"。** `ggml_add`、`ggml_mul_mat` 这类调用只是往计算图里加一个节点,不产生任何计算。真正的计算发生在第 3 章的 `graph_compute`。

**`build_*` 是基类提供的助手。** `build_norm`、`build_ffn`、`build_attn` 这些都在 `src/llama-graph.cpp` / `llama-graph.h` 的 `llm_graph_context` 里。它们封装了大量细节:量化、LoRA、量化 scale、KV cache 读写、Flash Attention 分支。这也是为什么一个 236 行的文件能表达一个完整模型。

**`cb(cur, "name", il)` 是调试回调。** 它给张量打上可读名字,只在 `llama-eval-callback` 之类的工具里生效。读代码时可以忽略,但它对调试非常有用。

**`inp_out_ids` 那个 `ggml_get_rows` 是性能优化。** 最后一层如果只需要少数几个位置的 logits,提前把不需要的行砍掉,省掉大半个 lm_head 的计算。这就是 `logits_all=false` 在模型层的落地。

## 从 GGUF 到一个模型的完整链路

```
文件 model.gguf
  ggml/src/gguf.cpp            解析容器格式(key-value + 张量表)
  src/llama-model-loader.cpp   mmap 权重,按名字取张量
  src/llama-arch.cpp           arch 字符串 -> LLM_ARCH_* 枚举,key/张量名映射表
  src/llama-model.cpp:3683     switch(arch) 创建对应的模型类
  src/models/*.cpp             上面那三个虚函数
```

`src/llama-model.cpp` 第 3683 行那个大 switch 就是"架构名 -> C++ 类"的注册表。想知道某个模型有没有支持,先查 `src/llama-arch.cpp` 的那个 `LLM_ARCH_NAMES` 表。

## 推荐对比阅读

读完 `models/llama.cpp` 后,挑这两个对比,能看清"同一套骨架怎么变形":

| 文件 | 看点 |
|---|---|
| `src/models/deepseek2.cpp` 或 `deepseek3.cpp` | MLA(低秩压缩的注意力)、MoE、YaRN rope |
| `src/models/gemma3.cpp` | 多层注意力窗口交替、各种归一化的变体 |
| `src/models/qwen3moe.cpp` | MoE 的另一种写法,和上面两个对比 |
| `src/models/mamba.cpp` | 完全没有注意力,走 `llama-memory-recurrent`,和 Transformer 是两个物种 |

## 相关文档和工具

- `docs/development/HOWTO-add-model.md` - 官方加模型指南,加了模型必读
- `skills/add-new-model/` - 仓库内置的 skill,比文档更贴近当前代码
- `docs/development/llama-star/` - 训练相关,和推理无关
- `build/bin/llama-gguf` - 直接 dump 一个 GGUF 的所有 key 和张量表,对照代码看最有效
- `build/bin/llama-eval-callback` - 把每个张量的实际数值打出来,验证前向是否按预期

## 动手练习

1. `build/bin/llama-gguf model.gguf` 导出元数据,对照 `models/llama.cpp` 里每个 `get_key` 和 `create_tensor`,看是不是一一对应。
2. 在 `build_arch_graph` 的循环里加一行 `fprintf(stderr, "layer %d\n", il)`,跑一次,确认循环次数等于 `n_layer`。
3. 找一个 llama.cpp 不支持的模型架构,看 HOWTO,试着加个最简支持(只加 hparams 和张量声明,先不管图)。
4. 用 `llama-eval-callback` 打印某一层 attention 的输出,和 PyTorch 推理同层输出对比,验证数值正确性。
# 第 5 章:分词、采样、聊天模板

阶段目标:这三块和 `decode` 主干相对独立,可以按需读。它们在"模型输出一段像样的文字"这件事上各占一环。

```
用户文本 --分词--> token id --decode--> logits --采样--> 下一个 token --反分词--> 文本
                                    ^
                        聊天模板负责把多轮对话拼成模型认识的格式
```

## 5.1 分词:`src/llama-vocab.cpp`

4853 行,不建议通读。要知道的是"分词器有好几种互不兼容的算法",`include/llama.h` 里有枚举:

```c
LLAMA_VOCAB_TYPE_SPM    = 1, // SentencePiece,LLaMA 系
LLAMA_VOCAB_TYPE_BPE    = 2, // GPT-2 系 byte-level BPE
LLAMA_VOCAB_TYPE_WPM    = 3, // WordPiece,BERT 系
LLAMA_VOCAB_TYPE_UGM    = 4, // Unigram,T5 系
LLAMA_VOCAB_TYPE_RWKV   = 5, // 贪心
LLAMA_VOCAB_TYPE_PLAMO2 = 6, // Aho-Corasick + 动态规划
...
```

所以文件里的代码是几套实现并列的,读的时候只挑你关心的那一套(一般先看 BPE,最经典)。

### 一个容易踩的坑:`pre_type`

除了分词算法本身,还有一个"预处理类型"(`src/llama-vocab.h` 里 `llama_vocab_pre_type`),几十个取值。它处理的是"空格在哪、标点算不算边界、数字要不要拆"这类模型特有的怪癖,比如 Llama3 的分词和 GPT2 的分词算法相同但预处理不同。

这解释了为什么同一个算法在不同模型上切出来的 token 不一样。调试分词问题时,先确认 `pre_type` 对不对。

### 值得关注的几个函数

| 函数 | 作用 |
|---|---|
| `llama_tokenize` | 文本 -> token,有个 `add_special` 参数控制要不要加 BOS |
| `llama_token_to_piece` | 单个 token -> 文本片段,注意一个汉字可能跨多个 token |
| `llama_tokenize` 的 `special` 参数 | 是否把 `<|im_start|>` 这类特殊 token 当作单个 token 而不是拆成字符 |
| `llama_vocab_is_eog` | 判断是不是结束符(EOS/EOT 等),生成循环靠它停下 |

### 实践建议

用 `build/bin/llama-tokenize` 直接验证,别猜:

```sh
build/bin/llama-tokenize -m model.gguf -p "你好世界"
build/bin/llama-tokenize -m model.gguf -p "你好世界" --ids   # 只看 id
```

对应测试:`tests/test-tokenizer-0.cpp`、`test-tokenizer-1-bpe.cpp`、`test-tokenizer-1-spm.cpp`。测试里的用例是最好的行为文档。

BPE 到底怎么合并、中文为什么不是一个字一个 token、分词表存在 GGUF 的哪个字段,见[附录第 10 章](10-tokenizer-bpe.md)。

## 5.2 采样:`src/llama-sampler.cpp`

4381 行,但结构非常规整:每个采样器是一个实现了 `llama_sampler_i` 接口的结构体。

### 采样器链

核心设计是**链式**:每个采样器接收 logits,可能修改它,也可能直接选出 token 并结束这条链。

```cpp
auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));  // 必须放最后
```

顺序很重要:`top_k` 先砍掉大部分候选,`top_p` 再按累积概率砍,`temp` 缩放,`dist` 按概率抽。把 `dist` 放前面,后面的就完全不生效了。

### 现有的采样器

`include/llama.h` 里的 `llama_sampler_init_*` 就是全部清单:

| 类别 | 采样器 |
|---|---|
| 选择 | `greedy`、`dist`、`top_k`、`top_p`、`min_p`、`typical`、`top_n_sigma`、`xtc`、`adaptive_p` |
| 缩放 | `temp`、`temp_ext` |
| 惩罚 | `penalties`(重复/频率/存在惩罚)、`dry` |
| 约束 | `grammar`、`grammar_lazy_patterns`、`logit_bias` |
| 自适应 | `mirostat`、`mirostat_v2` |

### 三个接口函数

```c
sample(smpl, ctx, idx)   // 选出 token
accept(smpl, token)      // 告诉采样器"这个 token 被接受了"
reset(smpl)              // 清空调状态
```

`accept` 很关键:`penalties` 采样器要记住历史 token 才能算重复惩罚。所以生成循环里每轮都要调 `llama_sampler_accept`。这是新手常见的 bug 来源(忘了 accept,惩罚就永远不生效)。

### 值得注意的新东西:后端采样

代码里有 `llama_sampler_backend_begin` 和 `build_sampling()`(在第 4 章那个 `build_graph` 里),说明现在有"把采样搬进计算图"的能力。这样 logits 不用从 GPU 拷回 CPU,省掉大量传输。这是大词表模型的性能优化,读的时候可以跳过,但要知道它存在。

### 相关测试

`tests/test-sampling.cpp` 和 `tests/test-backend-sampler.cpp`。

## 5.3 聊天模板:`common/jinja/` + `common/chat*.cpp`

这是最容易把人劝退的一块,因为它的复杂度来自"兼容所有模型的怪癖",不是算法难。

### 分三层

**第一层:Jinja 引擎** `common/jinja/`

注意:这是 llama.cpp **自己实现的** Jinja 解释器,不是 minja,也不是第三方库。HuggingFace 上的模型都用 Jinja 模板描述 chat 格式,所以必须能渲染它们。

- `lexer.cpp` / `parser.cpp` / `runtime.cpp` - 经典的解释器三段式
- `README.md` - 说明了支持的语法范围
- 测试:`tests/test-jinja.cpp`

如果你只是想用,不用读这层。

**第二层:模板渲染** `common/chat.cpp`

`llama_chat_apply_template` 的实现就在这里。它做两件事:优先用 GGUF 里存的 Jinja 模板渲染;如果没有模板,就退回一组内置的 C++ 模板(按模型家族硬编码的那种老式做法)。

**第三层:输出解析** `common/chat-peg-parser.cpp` + `docs/autoparser.md`

模型生成完文字后,要从中解析出"思考内容 / 正文 / 工具调用"三部分。老做法是用正则,新做法用 PEG 语法(`docs/development/parsing.md`)。

`docs/autoparser.md` 讲的是那套"自动检测模型特征"的机制:给一个模型,引擎自己判断它是 Qwen 系还是 Llama 系,该用哪个解析器。这块代码在 `common/chat-auto-parser*.cpp`。

### 实用建议

先别读代码,先用工具把现象看清:

```sh
build/bin/llama-template-analysis -m model.gguf    # 看这个模型用什么模板
build/bin/llama-chat 或者 llama-cli 的 -sys/-p     # 看模板渲染结果
build/bin/test-chat-template                       # 跑内置的所有模板测试
```

聊天模板出问题时,99% 是"模型以为的格式"和"你喂的格式"不一致。用 `--verbose` 看实际送给模型的 prompt,比读代码快得多。

## 这三块的共同学习方式

1. 先用命令行工具观察现象
2. 找到对应的 `tests/test-*.cpp` 看期望行为
3. 只在出问题或者要改动时,才去读实现

它们的实现都很长,但没有哪一行是"必须理解才能继续"的。相反,第 3 章和第 4 章的主干是绕不过去的。
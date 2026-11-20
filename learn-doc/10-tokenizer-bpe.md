# 附录:分词为什么不是一个字一个 token

第 5 章只讲了"分词器有好几种算法",这一章以本地 `models/Qwen3.5-4B-Q4_K_M.gguf` 为例,把中文分词这个最常见的困惑彻底拆开:BPE 到底在做什么、词表里到底存了什么、分词表是不是跟着模型文件一起下下来的。

## 一、现象

```sh
build/bin/llama-tokenize -m models/Qwen3.5-4B-Q4_K_M.gguf -p "你好呀中国人"
109266 -> '你好'
102894 -> '呀'
104693 -> '中国人'
```

5 个汉字切成 3 个 token,`你好` 和 `中国人` 各自是一个 token。这不是 bug,也不是模型的词表坏了。

## 二、BPE 是什么

**Byte Pair Encoding**,本来是 1994 年的数据压缩算法,2016 年被 Sennrich 拿去切词。核心只有两步:从最小单位开始,反复把最高频的相邻对合并成一个新符号。

### 训练:建词表

假设语料只有这几句:

```
中文   中文字   分词   中国人   中国人   中国人
```

初始词表是单字 `{中, 国, 人, 文, 字, 分, 词}`,统计相邻对:

| 相邻对 | 次数 |
|---|---|
| (中,国) | 3 |
| (国,人) | 3 |
| (中,文) | 2 |
| (文,字) | 1 |
| (分,词) | 1 |

第 1 轮合并 `(中,国)`,新 token `中国` 入表;第 2 轮句子变成 `中国 人`,`(中国,人)` 出现 3 次,再合并出 `中国人`。最终得到一张有序的表:

```
merges: [ ("中","国"), ("中国","人"), ... ]
tokens: [ 中, 国, 人, ..., 中国, 中国人 ]
```

一直合并到目标词表大小为止。这就是"哪些字能合成一个 token"的**唯一依据:语料频率**,没有任何语言学知识参与。所以 `的分` 这种在"目的分析""别的分词器"里高频出现的跨词组合也会被合成一个 token。

### 编码:新文本 -> token id

编码不是查词典,而是按 `merges` 的顺序(rank 越小越优先)反复套用可用的合并:

1. `你好` 的 rank 很小,先合上;
2. `呀` 找不到可以合并的伙伴,保持单字;
3. `中国人` 的 rank 也小,合上。

结果是 `[你好, 呀, 中国人]`。BPE 的编码结果是确定的,只看这张优先级表,不做概率搜索。

## 三、字节级 BPE 和预切分

Qwen 用的是 GPT-2 那套**字节级 BPE**:先把 256 个字节值映射成可打印字符(空格变成 `Ġ`,两个空格变成 `ĠĠ`,所以 dump 出来的 merges 长这样):

```
tokenizer.ggml.merges = ['Ġ Ġ', 'ĠĠ ĠĠ', 'i n', 'Ġ t', ...]
```

好处是没有任何 OOV:任何 UTF-8 字符串都能编码,生僻字退化成几个字节 token。

另一个容易忽略的环节是**预切分**(pretokenizer regex),它在 BPE 之前先按正则把文本切成 chunk,BPE 只在 chunk 内部合并。也就是说,算法相同的两个模型,预切分不同,切出来的 token 也完全不同。Qwen 系的正则在 `src/llama-vocab.cpp:400`:

```
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|
\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

`你好呀中国人` 全是 `\p{L}`,所以是**一个整 chunk**,BPE 可以在这 6 个字内部自由合并。而数字 `\p{N}` 被单独切出来,这也解释了为什么这类模型常常数不清长数字里的小数位。

## 四、这个模型的词表实测

把 `tokenizer.ggml.tokens` 里 248320 条 token 用 GPT-2 的字节映射还原成 UTF-8 后统计:

| 指标 | 值 |
|---|---|
| 词表大小 | 248320 |
| merge 规则条数 | 247587 |
| 单字汉字 token | 5425 |
| 由 >=2 个汉字组成的 token | **48001** |
| 多字 token 最长 | 15 个汉字 |

多字 token 的长度分布:

| 汉字数 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| 个数 | 28200 | 11824 | 6408 | 931 | 367 | 105 | 85 | 30 |

最长的几个,全部是语料里高频出现的固定串:

```
承担一切因您的行为而直接或间接
并不意味着赞同其观点或证实其
加倍支付迟延履行期间的债务利息
新时代中国特色社会主义思想   (这一类最长的有 15 字)
```

### 几个典型切分

```sh
-p "中华人民共和国"      -> [99584]                       1 token
-p "中国"                -> [96328]                       1 token
-p "中文的分词不是一个字一个token呀"
                         -> ['中文','的分','词','不是一个','字','一个','token','呀']
-p "魑魅魍魉"            -> [96667,239,99736,96667,235,96667,231]
```

最后一行是字节回退(`byte fallback`):`魅` 在词表里,`魑`、`魍`、`魉` 不在,于是退化成原始 UTF-8 字节,4 个字花掉 7 个 token。生僻字、罕见符号、某些 emoji 都会走这条路。

想验证任意字符串,直接跑工具,不要靠猜:

```sh
build/bin/llama-tokenize -m models/Qwen3.5-4B-Q4_K_M.gguf -p "测试文本"
build/bin/llama-tokenize -m models/Qwen3.5-4B-Q4_K_M.gguf -p "测试文本" --ids --show-count
```

## 五、其他分词方式

`include/llama.h:72` 的枚举就是 llama.cpp 支持的全部:

| 方式 | 原理 | 代表模型 | 枚举 |
|---|---|---|---|
| **BPE**(字节级) | 高频相邻对合并,按 merge rank 贪心编码 | GPT-2/4, Llama 3, Qwen, DeepSeek | `LLAMA_VOCAB_TYPE_BPE` |
| **SPM-BPE** | 同为 BPE,但用 SentencePiece 框架 + `▁` 表空格 + byte fallback | Llama 1/2, Mistral, Baichuan | `LLAMA_VOCAB_TYPE_SPM` |
| **WordPiece** | 仍是合并,但选让语料似然最大的对,不是频率最高的对;编码用最长匹配,续接子词加 `##` | BERT, BGE | `LLAMA_VOCAB_TYPE_WPM` |
| **Unigram LM** | 反过来:先给巨大词表,用 EM 估每个 subword 概率,再按"删掉它损失最小"逐步裁剪;编码用 Viterbi 求最大概率切分 | T5, ALBERT, mBART | `LLAMA_VOCAB_TYPE_UGM` |
| **贪心最长匹配** | 不合并,直接从词表里找最长命中 | RWKV | `LLAMA_VOCAB_TYPE_RWKV` |
| **Aho-Corasick + 动态规划** | 多模式串匹配出候选,再 DP 选最优切分 | PLaMo-2/3 | `LLAMA_VOCAB_TYPE_PLAMO2/3` |
| 字符级 | 一个字符一个 token,序列极长 | 早期 char-RNN | 不支持 |
| 词级 | 按空格切词,天然 OOV | word2vec 时代 | 不支持 |
| tokenizer-free | 直接吃原始字节,靠模型内部做 patch | ByT5, MegaByte, BLT | 不支持 |

几个容易混的点:

- **SentencePiece 不等于 Unigram**。SentencePiece 是个库,里面既能训 BPE 也能训 Unigram。Llama 用的是它训出来的 BPE,所以叫 SPM-BPE。
- **BPE 和 Unigram 的区别**在训练方向:BPE 自底向上合并(离散、确定),Unigram 自顶向下裁剪(概率、能给多种切分打分)。Unigram 训练更慢,但对形态丰富的语言(土耳其语、芬兰语)更好。
- **想做到"一字一 token"** 只能靠字符级切分,或在预切分正则里把每个汉字单独切开。代价是同样的句子要多几倍 token,上下文利用率和推理速度都变差。大词表 BPE 就是为了压这个比值。

## 六、分词表就在 GGUF 里

`.gguf` 是自包含的,分词相关的全部 metadata 都在里面,不需要再把 `tokenizer.json` / `vocab.json` / `merges.txt` 一起下载。用 gguf-py 的脚本看:

```sh
PYTHONPATH=gguf-py python3 gguf-py/gguf/scripts/gguf_dump.py --no-tensors models/Qwen3.5-4B-Q4_K_M.gguf
```

```
34: STRING   |        1 | tokenizer.ggml.model = 'gpt2'                # 分词器类型
35: STRING   |        1 | tokenizer.ggml.pre = 'qwen35'                # 预切分规则
36: [STRING] |   248320 | tokenizer.ggml.tokens = ['!', '"', ...]      # 词表本体:id -> token
37: [INT32]  |   248320 | tokenizer.ggml.token_type = [1, 1, ...]      # normal/unknown/control/user-defined
38: [STRING] |   247587 | tokenizer.ggml.merges = ['Ġ Ġ', 'i n', ...]  # BPE 合并规则,顺序即优先级
39: UINT32   |        1 | tokenizer.ggml.eos_token_id = 248046
40: UINT32   |        1 | tokenizer.ggml.padding_token_id = 248044
41: BOOL     |        1 | tokenizer.ggml.add_bos_token = False
42: BOOL     |        1 | tokenizer.ggml.add_eos_token = False
43: STRING   |        1 | tokenizer.chat_template = '{%- set image_count = ... %}'
44: BOOL     |        1 | tokenizer.ggml.normalizer.nfc = True
45: BOOL     |        1 | tokenizer.ggml.clean_spaces = False
```

几点说明:

- 转换时一次性打包进去:`convert_hf_to_gguf.py` 把 HF 仓库里的 `tokenizer.json`(或 `vocab.json` + `merges.txt`)以及 `tokenizer_config.json` 里的 special tokens、chat template 全部转成上面这些 key。
- `merges` 247587 条,`tokens` 248320 条,两者数量接近:BPE 里每个 merge 产出一个新 token。
- 词表这部分大约占 6 MB。`models/` 下那些 `ggml-vocab-*.gguf` 就是从模型里抽出来的纯词表文件,给 `tests/test-tokenizer-0` 之类的单元测试用,比如 `ggml-vocab-qwen35.gguf` 是 5.9 MB。
- 只有想用 Python 侧的 HF `AutoTokenizer` 做对照时,才需要额外去 HF 下分词文件。

## 七、顺带:`n_ctx_train = 0` 的告警是误报

跑 `llama-tokenize` 时会出现:

```
W llama_context: n_ctx_seq (512) > n_ctx_train (0) -- possible training context overflow
```

原因:`llama-tokenize` 用 `vocab_only = true` 加载模型(`tools/tokenize/tokenize.cpp:146`),而 `llm_load_hparams` 在 `src/llama-model.cpp:1483` 遇到 `vocab_only` 就提前返回,`n_ctx_train` 一直是 0,于是 `src/llama-context.cpp:447` 的检查必然触发。

文件里 `qwen35.context_length = 262144` 是有的,所以这条告警在 vocab-only 场景下没有意义,忽略即可。

## 八、速查

| 问题 | 答案 | 从哪看 |
|---|---|---|
| 为什么一个词是一个 token | 语料里高频,被 BPE 合并过 | `tokenizer.ggml.merges` |
| 词表多大 | 248320 | `tokenizer.ggml.tokens` 条数 |
| 中文单字都是 token 吗 | 常用字是,生僻字退化成字节 | `魑魅魍魉` 实测 7 token |
| 分词算法哪种 | 字节级 BPE + qwen35 预切分 | `tokenizer.ggml.model` / `.pre` |
| 分词表要另外下吗 | 不用,在 GGUF metadata 里 | `gguf_dump.py --no-tensors` |
| 别的高频组合为什么合并 | BPE 只看相邻频率,不懂词边界 | 本文第二节 |
| 怎么快速验证 | `llama-tokenize --ids`,server 用 `POST /tokenize` | - |
| 想看实现 | `src/llama-vocab.cpp` 的 `llm_tokenizer_bpe`,预切分正则在 380-450 行 | - |
| 对应的测试 | `tests/test-tokenizer-0.cpp`、`test-tokenizer-1-bpe.cpp` | - |
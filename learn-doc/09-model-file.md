# 附录:读懂一个 GGUF 文件的名字和超参

以本地 `models/Qwen3.5-4B-Q4_K_M.gguf` 为例,回答两个最常被问的问题:名字里的字母数字是什么意思,以及这个模型能吃多长的上下文、一次能吐多少。所有数字都是从文件里读出来的,不是查模型卡抄的。

## 一、文件名拆开看

| 片段 | 含义 |
|---|---|
| `Qwen3.5` | 模型系列 |
| `4B` | 参数量约 4 billion = 40 亿。这个文件实测 4,205,751,296 个参数,文件头里的 `general.size_label` 写的是 `4.2B` |
| `Q4` | 权重量化位宽,名义上平均 4 bit / 权重 |
| `K` | 用 k-quant 方案,不是老的 `Q4_0` / `Q4_1` |
| `M` | 混合精度的档位,Medium |

### 参数量 != 文件大小

参数量是模型结构决定的,文件大小是量化决定的。同一个 4.2B 模型:

| 量化 | 大小 |
|---|---|
| F16 | ~8.4 GiB |
| Q8_0 | ~4.5 GiB |
| **Q4_K_M** | **2.81 GiB(实测)** |

所以看文件大小判断"这模型多大"很容易被骗:一个 2.8 GiB 的 Q4 文件和 8.4 GiB 的 F16 文件是同一个模型。

### Q4_K 是什么

老的 `Q4_0` / `Q4_1` 是"每 32 个权重一组,共享一个 16 bit 的缩放因子",理论 4.5 bit/权重。k-quant 改成了两层,以 `Q4_K` 为例:

- 一个大 block 装 256 个权重,分 8 个 sub-block,每个 sub-block 32 个权重;
- sub-block 有自己的 scale(和 min),这些 scale 本身被压成 6 bit 再存一遍;
- 大 block 还有一个总的 scale 兜底。

好处是把"缩放因子"的开销摊薄了,同样 4 bit 的情况下精度明显好于 `Q4_0`,并且能按张量挑不同的位宽。

### M 是"哪个张量多给几位"

`Q4_K_M` 不是一个统一位宽,它是一张"按张量分配位宽"的配方。核心逻辑在 `src/llama-quant.cpp`:

- `src/llama-quant.cpp:973` - 默认所有张量 `GGML_TYPE_Q4_K`
- `src/llama-quant.cpp:715` - ffn_down 里"重要"的层升到 `Q6_K`
- `src/llama-quant.cpp:762` - attention 的输出投影升到 `Q5_K`
- `src/llama-quant.cpp:513` 的 `use_more_bits()` 决定哪些层算"重要":开头 1/8 层、结尾 1/8 层,中间每隔 3 层挑 1 层

所以 `_S` / `_M` / `_L` 的区别只是这张配方给多给少:`_S` 几乎全按名义位宽走,`_L` 给更多张量加位宽。

这个文件里的实际分布(用 gguf-py 数出来的):

| 张量类型 | 张量个数 | 位宽 |
|---|---|---|
| `Q4_K` | 97 | 4.5 bit/权重 |
| `Q8_0` | 152 | 8.5 bit/权重 |
| `F32` | 177 | 32 bit/权重 |

`F32` 那 177 个是 norm、`ssm_a`、`ssm_conv1d`、`ssm_dt.bias` 这类小张量,本来就该保持原精度。`Q8_0` 大量出现是因为 SSM 部分和部分 projection 没走 k-quant 配方。

合起来平均位宽 = 3021123584 字节 x 8 / 4205751296 参数 = **5.75 bit/权重**,远高于名字里的 "4"。名字里的数字是主体权重的名义值,不是实际平均值。

### 常见档位怎么选

| 量化 | 4.2B 模型大小 | 说明 |
|---|---|---|
| F16 | ~8.4 GiB | 基准,几乎没人跑 |
| Q8_0 | ~4.5 GiB | 接近无损,显存够就选它 |
| Q6_K | ~3.4 GiB | 很难看出差别 |
| Q5_K_M | ~3.0 GiB | 质量很好 |
| **Q4_K_M** | **2.81 GiB(实测)** | 社区默认,性价比最高 |
| Q3_K_M | ~2.0 GiB | 小模型上掉点明显 |
| Q2_K | ~1.6 GiB | 只在 70B 以上的模型上勉强能用 |

表里大小是估算,`Q4_K_M` 那行是实测。别用名字里的位宽去推算大小:这个文件名义 Q4,实际平均 5.75 bit,因为 SSM 那部分张量是 `Q8_0` / `F32`。比大小直接看文件。

一张 8 GB 的卡跑 4B 模型,`Q4_K_M` 或 `Q5_K_M` 都合适;`Q8_0` 就要靠上下文短一点或者部分层放 CPU 了。

## 二、上下文窗口:262144

文件头里的值:

```
qwen35.context_length = 262144
```

也就是 **256K token**。四个读法,从轻到重:

```sh
# 1. 只读元数据,不加载权重,最快
python3 gguf-py/gguf/scripts/gguf_dump.py --no-tensors models/Qwen3.5-4B-Q4_K_M.gguf | grep context_length

# 2. 用 gguf-py 当库读
python3 -c "import sys; sys.path.insert(0,'gguf-py'); from gguf import GGUFReader; \
print(GGUFReader('models/Qwen3.5-4B-Q4_K_M.gguf').fields['qwen35.context_length'].contents())"

# 3. 加载模型时打印,会真的把权重读一遍
build/bin/llama-cli -m models/Qwen3.5-4B-Q4_K_M.gguf -n 1 -p hi -v 2>&1 | grep print_info
# print_info: n_ctx_train           = 262144

# 4. 服务里查
curl -s localhost:8080/v1/models | jq '.data[0].meta.n_ctx_train'
curl -s localhost:8080/props    | jq '.default_generation_settings.n_ctx'
```

注意 `n_ctx_train`(模型训练时的长度)和 `n_ctx`(这次运行真正分配的 KV cache 长度)是两回事。`-c 0` 是默认值,意思是"直接等于训练长度",见 `common/common.h:471`。实测确实会分配满:

```
llama_context: n_ctx     = 262144
```

### 256K 要花多少内存

实测(`-ngl 0`,全 CPU):

| 项 | 大小 | 随上下文增长 |
|---|---|---|
| 模型权重(mmap) | 2881 MiB | 否 |
| KV cache | 8192 MiB | 是,32 KiB / token |
| SSM 递归状态 | 50 MiB | 否 |
| 输出 buffer | 0.95 MiB | 否 |

这个模型是混合结构,不是纯 attention:

```
qwen35.block_count             = 32
qwen35.full_attention_interval = 4      # 每 4 层才是真 attention
qwen35.attention.head_count_kv = 4
qwen35.attention.key_length    = 256
qwen35.attention.value_length  = 256
```

32 层里只有 8 层存 KV(`attention.recurrent_layers` 里标成 `False` 的是第 3、7、11、15、19、23、27、31 层,其余 24 层是 SSM)。每 token 的 KV:

```
2(K 和 V) * 8 层 * 4 head * 256 dim * 2 字节(f16) = 32 KiB
262144 token * 32 KiB = 8 GiB
```

剩下 24 层是 SSM,状态大小固定,和上下文长度无关,只要 50 MiB。这是它 256K 还能跑的原因:纯 attention 的 4B 模型按同样 head 配置算 KV,要 4 倍以上。

### 实操建议

`-c` 设多大就按多大分配 KV,默认的 262144 会直接吃掉 8 GiB,不值得。按需给:

```sh
# 日常用 32K,够读代码、跑 agent
build/bin/llama-cli -m models/Qwen3.5-4B-Q4_K_M.gguf -c 32768 -n 8192 -p "你好"

# 真要长上下文再把 KV 压一下,大约减半
build/bin/llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf -c 131072 -ctk q8_0 -ctv q8_0
```

## 三、单次最多输出多少

**GGUF 里没有这个字段。** 模型本身没有"最大输出 token 数"的概念,`tokenizer.chat_template`、超参里都没有。能输出多少只由两个东西卡住:

```
可输出 token = min(n_predict, n_ctx - prompt 已占用的 token)
```

- `n_predict` 是运行参数,`common.h:470` 默认 `-1`,意思是"不限",一直生成到出 EOS 或者撑满上下文;
- `n_ctx - prompt` 是硬上限,上下文满了就只能停下。

所以这个模型的理论上限是 **262144 减去 prompt 长度**,但实际没人这么用:生成 25 万 token 要跑几个小时,质量也早就崩了。

```sh
# CLI: -n 是上限
build/bin/llama-cli -m models/Qwen3.5-4B-Q4_K_M.gguf -c 32768 -n 4096 -p "写个快排"
```

```sh
# server: max_tokens 和 n_predict 等价,不传就用启动参数 / 默认 -1
curl -s localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "写个快排"}],
  "max_tokens": 4096
}'
```

被截断时怎么发现:响应里的 `finish_reason` 是 `length` 而不是 `stop`(`tools/server/server-task.cpp:374`)。CLI 里就是 `-n` 用满后直接停,没有任何提示。

实践上按用途给:聊天 2K 到 4K,写代码 8K,这类 thinking 模型(模板里默认带 ` thinking`)要给到 8K 以上才不会被思考过程吃掉全部预算。

## 四、速查

| 问题 | 答案 | 从哪看 |
|---|---|---|
| 这个模型多大 | 4.2B 参数 | `general.size_label` |
| 量化到什么程度 | Q4_K_M,平均 5.75 bit | 文件名 + 统计张量类型 |
| 文件占多少磁盘 | 2.81 GiB | `ls -l` |
| 上下文窗口 | 262144 token | `qwen35.context_length` |
| 跑满上下文要多少内存 | 8 GiB KV + 2.8 GiB 权重 | 加载日志里的 `KV buffer size` |
| 单次最多输出 | 没有固定值 | `n_predict` 和剩余上下文 |
| 输出被截断了吗 | `finish_reason == "length"` | API 响应 |
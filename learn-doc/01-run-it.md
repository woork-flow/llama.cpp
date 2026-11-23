# 第 1 章:先跑起来

阶段目标:建立"手感"。在没读一行代码之前,先知道这个项目能干什么、参数改了什么。

## 准备一个模型

需要一个 GGUF 文件。在 HuggingFace 上搜带 `-GGUF` 后缀的仓库,挑一个 3B 到 7B 的模型就够学习了。太小(比如 0.5B)看不出性能差别,太大加载慢、占显存。

放到任意目录,后面用 `-m 路径` 指定。

## 编译

如果需要从头编:

```sh
CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)
```

`CUDACXX` 别省。删掉 `build/` 重新配置时,CMakeCache 里的编译器路径一起没了,而 `/usr/local/cuda/bin` 默认不在 PATH 里,cmake 会直接报 `No CMAKE_CUDA_COMPILER could be found`。长期用的话把 `/usr/local/cuda/bin` 加进 PATH,这一步就可以省掉。

纯 CPU 学习完全够用,去掉 `-DGGML_CUDA=ON` 编译更快。产物都在 `build/bin/`。

编完确认产物是当前分支的:

```sh
build/bin/llama-cli --version   # 看 commit 是否等于 git rev-parse --short HEAD
```

跑模型前先瞄一眼显存。其余进程占着显存时,报错是 `unable to allocate CUDA0 buffer`,不是编译坏了:

```sh
nvidia-smi --query-gpu=memory.free --format=csv
```

## 两个入口:llama-cli 和 llama-completion

这两个最容易搞混,行为差别不小。

| | `llama-cli` | `llama-completion` |
|---|---|---|
| 源码 | `tools/cli/`,约 1300 行 | `tools/completion/completion.cpp`,955 行 |
| 结构 | 内部起一个 `llama-server` 线程,自己用 HTTP 客户端跟它对话 | 单进程,直接调 `llama.h` 循环 decode 加采样 |
| 能力 | 跟 server 一致:jinja 模板、自动 parser、思考段提取、工具调用 | 老式实现,自己拼模板字符串 |
| 关掉对话模式 | 不行,没有 `-no-cnv` | `-no-cnv` |
| 适合 | 日常聊天 | 测基线、纯文本补全、对照第 2 章的最小循环 |

`llama-completion` 就是老版本的 `llama-cli`,被搬到了 `tools/completion/`;现在占着 `llama-cli` 这个名字的是 server 的客户端。

### llama-cli - 聊天 UI

```sh
build/bin/llama-cli -m model.gguf -p "用一句话解释什么是 KV cache" -n 128 -st
```

`-st`(`--single-turn`)是必须的,不加就进交互模式等输入。

注意 `-n` 是**生成 token 总数**的上限,思考内容也算在里面。现在的模型多是思考模型,Qwen3.5、gemma-4 这些会把 128 个 token 全花在思考段里,正文一个 token 都出不来,看起来就像“输出不全”。三种改法:

```sh
build/bin/llama-cli -m model.gguf -p "..." -n -1 -st                      # 放开到无限,吐 EOS 为止
build/bin/llama-cli -m model.gguf -p "..." -n 256 -st -rea off            # 关掉思考,直接给答案
build/bin/llama-cli -m model.gguf -p "..." -n 512 -st --reasoning-budget 0 # 思考立刻结束,正文照常
```

`-rea off` 改的是模板里的 `enable_thinking`,`--reasoning-budget` 只截思考段,后者更细。

### llama-completion - 测基线用这个

```sh
build/bin/llama-completion -m model.gguf -p "用一句话解释什么是 KV cache" -n 200 -no-cnv
```

不加 `-no-cnv` 它会检测到 chat template 就自动进对话模式(`llama-completion: chat template is available, enabling conversation mode`),`-n` 同样被思考段吃掉。测速度、看 KV cache 复用,一律加 `-no-cnv`。

它还有几个 `llama-cli` 没有的老参数:`-r`/`--reverse-prompt` 撞到指定串就停下、`--in-prefix`/`--in-suffix`、`-i` 交互模式。做纯补全实验时用得上。

## 另外三个常用工具

### llama-server - OpenAI 兼容的 HTTP 服务

```sh
build/bin/llama-server -m model.gguf -c 4096 --port 8080
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}]}'
```

服务端是 `tools/server/`,是整个仓库里最复杂的上层应用。它负责多请求并发、slot 调度、流式输出、聊天模板渲染。等你读懂 decode 那一层之后再来读它。

### llama-bench - 性能基线

```sh
build/bin/llama-bench -m model.gguf
```

输出 pp(prompt processing)和 tg(token generation)两个速度。以后改任何代码,先跑这个看有没有变慢。

```
linux@linux-System-Product-Name:~/code/native/llama.cpp$ build/bin/llama-bench -m ./models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7782 MiB):
  Device 0: NVIDIA GeForce RTX 4060, compute capability 8.9, VMM: yes, VRAM: 7782 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.81 GiB |     4.21 B | CUDA       |  -1 |           pp512 |     3841.34 ± 134.92 |
| qwen35 4B Q4_K - Medium        |   2.81 GiB |     4.21 B | CUDA       |  -1 |           tg128 |         64.08 ± 2.50 |

build: af0416f7b (12360)
```

### llama-tokenize - 看分词结果

```sh
build/bin/llama-tokenize -m model.gguf -p "hello world" 
```

```
linux@linux-System-Product-Name:~/code/native/llama.cpp$ build/bin/llama-tokenize -m ./models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf -p 你好呀中国人
0.00.421.783 W llama_context: n_ctx_seq (512) > n_ctx_train (0) -- possible training context overflow
109266 -> '你好'
102894 -> '呀'
104693 -> '中国人'
```

### 另外两个值得知道的

```sh
build/bin/llama-simple -m model.gguf -n 16 "hello"   # examples/simple,代码最少;prompt 是位置参数,没有 -p
build/bin/llama-perplexity -m model.gguf -f wiki.test.raw   # 判断量化/改动有没有伤到质量
```

## 调参数看现象

这一步是这个阶段的核心。改一个参数,观察行为变化:

| 参数 | 含义 | 试着改小看什么 |
|---|---|---|
| `-c` | 上下文长度,最多能记多少 token | 改小到 128,长对话会被截断 |
| `-b` | batch size,一次 decode 最多塞多少 token | 主要影响 prompt 处理速度 |
| `-ub` | ubatch size,实际一次送进计算图的 token 数 | 影响显存峰值 |
| `-ngl` | 放到 GPU 上的层数,0 表示纯 CPU | 观察速度和显存的线性变化 |
| `-t` | CPU 线程数 | 超过物理核数通常反而变慢 |
| `--temp` `--top-k` `--top-p` | 采样参数 | temp 设 0 输出变确定 |
| `-n` | 生成 token 上限,思考段也算 | 设 64,思考模型基本出不来正文 |
| `-fa` | flash attention,on/off/auto | 关掉看显存和速度变化 |

特别值得玩:`-b 2048 -ub 128` 这种组合,你会发现 pp 速度明显下降。为什么?读第 3 章就明白了(一个 batch 会被切成多个 ubatch)。

一组参考基线(RTX 4060 8G,Qwen3.5-4B Q4_K_M 全部 offload,显存空闲 5.6G):

```sh
build/bin/llama-bench -m model.gguf
```

```
pp512   3985 t/s
tg128     64 t/s
```

同一台机器 `llama-completion -n 128 -c 4096 -no-cnv` 报 65 t/s,两边对得上。

比数字不一致更坑的是 **`-fit`**。它默认是 on,`n_ctx` 和 `-ngl` 没显式指定时,引擎会按空闲显存自动缩放:`-fitc` 是最小 ctx(默认 4096),`-fitt` 是每张卡留多少余量(默认 1024 MiB)。同一台机器,显存空闲时自动给 `n_ctx = 32768`,被别的推理服务占了几个 G 之后会掉到 `n_ctx = 4096`。

所以定基线时:

- 显式写 `-c`,不要靠自动
- 跑之前先 `nvidia-smi --query-gpu=memory.free --format=csv` 看一眼,边上挂着别的服务会把数拉得很难看
- 想完全自己控制,加 `-fit off`

测速用 `llama-bench`,它把 ctx 固定在 `n_prompt + n_gen`,不受 `-fit` 影响。数字怎么判读(跑满了没有)见[第 8 章](08-bench-metrics.md)。

`llama-completion` 结尾的 `graphs reused` 和 `graph reallocs` 还是值得看:reused 等于生成步数、reallocs 为 0,说明 KV cache 位置连续、图尺寸稳定。这个数掉下来,通常是 `-b`/`-ub` 设错了,或 position 不连续。

## 一个 5 分钟的验证练习

1. 用 `llama-bench` 记录一次基线。
2. 把 `-ngl` 从 0 试到 99,画出速度曲线,找到"刚好全放显存"的临界点。
3. 用 `llama-tokenize` 看看中文和英文分别被切成几个 token,理解为什么中文更费上下文。
4. 用 `llama-perplexity` 跑同一个模型的两个量化版本(比如 Q4_K_M 和 Q8_0),看困惑度差多少。

### 练习 2 的实测结果

2026-11 在这台机器上扫的(Qwen3.5-4B Q4_K_M,32 层,显存空闲 6.2G):

```sh
build/bin/llama-bench -m models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf -ngl 0,6,12,18,24,28,32,33,99
```

| ngl | pp512 (t/s) | tg128 (t/s) | 备注 |
|---:|---:|---:|---|
| 0 | 1187 | 19.8 | 纯 CPU |
| 6 | 1354 | 24.9 | |
| 12 | 1589 | 27.4 | |
| 18 | 1905 | 33.6 | |
| 24 | 2438 | 42.0 | |
| 28 | 2915 | 49.8 | |
| 32 | 3787 | 64.2 | 层数放满,但还差一层 |
| 33 | 4149 | 70.5 | 临界点:全部上卡 |
| 99 | 4181 | 71.0 | 饱和,和 33 持平 |

要点:

- 临界点是 **ngl = n_layer + 1 = 33**,不是 32。offload 按 `n_layer + 1` 个层(32 个 block 加 1 个输出层)从后往前数,见 `src/llama-model.cpp` 的 `i_gpu_start = n_layer_all + 1 - n_gpu_layers`。`-ngl 32` 时第 0 个 block 还在 CPU 上,每个 token 多一次 GPU-CPU 折返,这就是 32 到 33 那一跳(tg +10%)的来源。
- `-ngl 99` 和 33 完全持平,99 只是"保证超过 n_layer + 1"的习惯写法。默认的 `-ngl -1` 实测 tg 68.9,也是全放档。
- 全放时显存峰值约 4.9G(含桌面占用 1.5G),模型侧约 3.4G = 权重 2.8G + KV + 计算 buffer。8G 卡放这个模型绰绰有余,所以只看到性能拐点,没看到 OOM 一侧;换 13B 以上的模型才会撞 `unable to allocate CUDA0 buffer`。
- 换 0.8B(24 层)扫出来形状一样:拐点在 25,tg 从 64 线性涨到 232 t/s。
- 上面"参考基线"记的 tg 64 比这几轮全放档(69 到 71)低一截,同机同 build,差异来自测量环境。定基线前先看 `nvidia-smi`,跑完也别只信一次。

### 练习 4 的实测结果

2026-11 在这台机器上跑的(RTX 4060 8G,语料用 `scripts/get-wikitext-2.sh` 下载的 wikitext-2-raw,`-c 512`,全部 580 个 chunk 约 29.7 万 token):

```sh
sh scripts/get-wikitext-2.sh
build/bin/llama-perplexity -m models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf -f wikitext-2-raw/wiki.test.raw -c 512
build/bin/llama-perplexity -m models/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf   -f wikitext-2-raw/wiki.test.raw -c 512
build/bin/llama-perplexity -m models/Qwen3.5-0.8B/Qwen3.5-0.8B-Base-Q8_0.gguf -f wikitext-2-raw/wiki.test.raw -c 512
```

| 模型 | 量化 | PPL | 耗时 |
|---|---|---:|---:|
| Qwen3.5-4B | Q8_0 | 9.956 +/- 0.072 | 127 s |
| Qwen3.5-4B | Q4_K_M | 10.160 +/- 0.073 | 101 s |
| Qwen3.5-0.8B-Base | Q8_0 | 13.961 +/- 0.103 | 43 s |

两个 4B 文件都来自 ggml-org/Qwen3.5-4B-GGUF,文件大小与仓库逐字节一致,同源可比。

要点:

- **同模型对比才有结论**:Q4_K_M 对 Q8_0 是 10.160 对 9.956,+0.204 即 +2.1%,落在 Q4_K_M 的典型区间(0.5% 到 2% 略多一点)。结论:这个量化没有异常损伤,可以用。
- **跨尺寸对比说明不了量化问题**:0.8B Q8_0 对 4B Q8_0 差 40%,容量差距远远压过任何量化档的影响。只跑 Q4_K_M 和 0.8B Q8_0 两个文件是得不出"量化伤没伤"的结论的。
- ±0.07 是 580 个 chunk 间方差的标准误,不是量化噪声。0.204 的差约为合成误差的 2 倍,边缘显著但方向稳定;差值远超这个误差才算铁证。
- Q8_0 比 Q4_K_M 贵 1.4GB 权重(4.48GB 对 3.03GB),pp 也慢了约 25%(127s 对 101s),换回来的只是 2% 的 PPL。日常用 Q4_K_M 是合理选择。
- 想更直接地度量"伤了多少":`llama-perplexity --kl-divergence` 先用高精度模型存基准 logits,再算量化模型的 KL 散度和 top-1 翻转率,用法见 `tools/perplexity/README.md`。

做完这些,你对这个项目的"外围"就有概念了,可以开始读代码。
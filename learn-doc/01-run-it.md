# 第 1 章:先跑起来

阶段目标:建立"手感"。在没读一行代码之前,先知道这个项目能干什么、参数改了什么。

## 准备一个模型

需要一个 GGUF 文件。在 HuggingFace 上搜带 `-GGUF` 后缀的仓库,挑一个 3B 到 7B 的模型就够学习了。太小(比如 0.5B)看不出性能差别,太大加载慢、占显存。

放到任意目录,后面用 `-m 路径` 指定。

## 编译

仓库里已经有 `build/` 和 CUDA 版本了。如果要从头编:

```sh
cmake -B build -DGGML_CUDA=ON
cmake --build build -j
```

纯 CPU 学习完全够用,去掉 `-DGGML_CUDA=ON` 编译更快。产物都在 `build/bin/`。

## 四个必玩的工具

### llama-cli - 最基础的对话

```sh
build/bin/llama-cli -m model.gguf -p "用一句话解释什么是 KV cache" -n 128
```

这是最直接的接口。它会加载模型、建上下文、把 prompt 编码、循环 decode 加采样。

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

### llama-tokenize - 看分词结果

```sh
build/bin/llama-tokenize -m model.gguf -p "hello world" 
```

### 另外两个值得知道的

```sh
build/bin/llama-simple -m model.gguf -p "hello"   # 就是 examples/simple,代码最少
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

特别值得玩:`-b 2048 -ub 128` 这种组合,你会发现 pp 速度明显下降。为什么?读第 3 章就明白了(一个 batch 会被切成多个 ubatch)。

## 一个 5 分钟的验证练习

1. 用 `llama-bench` 记录一次基线。
2. 把 `-ngl` 从 0 试到 99,画出速度曲线,找到"刚好全放显存"的临界点。
3. 用 `llama-tokenize` 看看中文和英文分别被切成几个 token,理解为什么中文更费上下文。
4. 用 `llama-perplexity` 跑同一个模型的两个量化版本(比如 Q4_K_M 和 Q8_0),看困惑度差多少。

做完这些,你对这个项目的"外围"就有概念了,可以开始读代码。
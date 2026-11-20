# 第 8 章:读懂 llama-bench 的 pp / tg 数字

阶段目标:看到一行 bench 结果,能立刻判断"这个速度是跑满了还是有问题"。

## pp 和 tg 分别是什么

一次推理分两个阶段,llama-bench 分开测:

| 缩写 | 全称 | 干什么 | 对应参数 |
|---|---|---|---|
| pp | prompt processing / prefill | 一次性读入整段 prompt 的所有 token,并行计算 | `-p 512` |
| tg | text generation / decode | 逐 token 自回归,算完一个接回输入再算下一个 | `-n 128` |

两个阶段的瓶颈完全不同,所以不能用同一把尺子:

- pp 一次算 512 个 token,矩阵乘规模大,GPU 计算单元吃满,瓶颈是**算力(FLOPS)**
- tg 一次只算 1 个 token,矩阵乘退化成 GEMV,计算单元近乎空闲,瓶颈是**显存带宽**。每生成 1 个 token 必须把全部权重从显存完整读一遍

结果就是数值差两个数量级:pp 几千 tok/s,tg 几十 tok/s。

**体感速度只看 tg**(出字快不快),pp 决定的是"按下回车后等多久才开始出字"。

## tg:用带宽判断有没有跑满

稠密模型每 token 需要读的权重字节数,约等于模型文件体积。所以:

```
每 token 读权重 = 模型体积
tg t/s x 模型体积 = 实际显存带宽消耗
实际带宽 / 带宽峰值 = 达成率
```

实测例子(RTX 4060 8GB,峰值带宽 272 GB/s):

```
模型 2.81 GiB ≈ 3.02 GB
tg 64 t/s -> 64 x 3.02 = 193 GB/s
193 / 272 = 71%
```

判读经验:

| 达成率 | 结论 |
|---|---|
| 70% 以上 | 正常偏好,基本到顶 |
| 50% - 70% | 查是否有层 offload 到 CPU、是否降频、kernel 是否慢 |
| 50% 以下 | 大概率有层跑在 CPU 上,或者显存带宽被别的进程占了 |

注意 272 GB/s 是标称峰值,实际可用约 85% - 90%。所以 71% 换算成"实际可达带宽"约 80%,已经很接近上限了。

## pp:用算力判断

```
512 token / pp 速度 = 耗时
参数量 x 2 FLOP x 512 token = 计算量
计算量 / 耗时 = 实际 FLOPS
```

实测例子:

```
512 / 3841 = 0.133 s
4.21e9 x 2 x 512 = 4.3 TFLOP
4.3 / 0.133 ≈ 32 TFLOPS
```

RTX 4060 的 FP16 tensor 峰值约 60 TFLOPS,达成 50% 属于正常。Q4_K 这类量化矩阵乘要边反量化边算,加上 pp512 的 batch 还不够大,很难超过 55% - 60%。

## 实例:Qwen3.5-4B-Q4_K_M(4060 8GB)

原始输出和基线记录方式见[第 1 章](01-run-it.md)的 `llama-bench` 一节,这里只做判读。

```
| model                   |       size | params | backend | ngl |  test |    t/s |
| qwen35 4B Q4_K - Medium |   2.81 GiB | 4.21 B | CUDA    |  -1 | pp512 | 3841.34 |
| qwen35 4B Q4_K - Medium |   2.81 GiB | 4.21 B | CUDA    |  -1 | tg128 |   64.08 |
```

| 测试 | 实测 | 理论上限 | 达成率 |
|---|---|---|---|
| pp512 | 3841 +- 135 t/s | FP16 tensor 约 60 TFLOPS | 约 50% |
| pp512 `-fa 1` | 3957 +- 54 t/s | 同上 | 约 52% |
| tg128 | 64.1 +- 2.5 t/s (193 GB/s) | 272 GB/s 峰值,实际约 240 GB/s | 峰值 71%,实际约 80% |

结论:这个构建在 sm_89 上工作正常,没有 CPU 回退或慢 kernel 的迹象。

### 为什么 tg 比同尺寸纯 dense 模型低

同样 4B Q4_K_M 的纯 attention 模型在这张卡上能到 70 - 80 t/s。Qwen3.5 是混合架构:32 层里每 4 层才有 1 层 full attention(共 8 层),其余 24 层是 gated delta net 线性注意力。

delta net 层每 token 除了 matmul,还有 conv1d、门控、delta rule 的递归状态更新等一堆 elementwise / scan 小算子。这类算子 kernel 启动开销大、访存效率低,拉高了每 token 的固定成本,所以 tg 低 10% - 20%。

换个角度看,这个架构的好处是:只有 8 层用 KV cache,24 层是固定尺寸的 recurrent state。所以**长上下文时 tg 几乎不衰减**,而 dense 模型在 32k 上下文能掉一半。8GB 卡跑长上下文,这是主要卖点。

## 调优清单

1. 换更小量化:`IQ4_XS`(2.4GB)或 `Q3_K_M`(2.2GB),tg 约 78 - 85 t/s
2. 查功耗墙:`nvidia-smi -q -d POWER | grep -i "Power Limit"`,115W 桌面版正常,被限到 80W 以下就是瓶颈
3. 查是否降频:跑 bench 时 `nvidia-smi dmon`,看 mem 列是否停在 8501 MHz
4. `-fa 1` 对 pp 有小幅提升(约 3%),对 tg 基本无影响
5. 确认全量 offload:`numa` / 线程数在 `-ngl 99` 下无关紧要,若 tg 远低于带宽估算,先确认没有层留在 CPU

## 常用命令

```sh
# 一次测完 pp 和 tg
build/bin/llama-bench -m model.gguf -p 512 -n 128 -r 3

# 对比 flash attention 开关
build/bin/llama-bench -m model.gguf -p 512 -n 128 -r 3 -fa 0,1

# 只测 tg,多跑几轮看方差
build/bin/llama-bench -m model.gguf -p 0 -n 128 -r 5

# 看不同量化 / 不同 ngl 的对比表
build/bin/llama-bench -m a.gguf -m b.gguf -ngl 99 -p 512 -n 128
```

`-r` 是重复次数,结果里的 `+-` 是标准差。tg 的标准差应该很小(几个百分点),偏大说明有其他进程抢 GPU 或者降频。
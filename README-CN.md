# 🕯️ Candle-vLLM-GCU

> **面向燧原GCU平台的大语言模型推理与聊天服务框架**，其基于 `Candle-GCU` 和 开源项目[`Candle-vLLM`](https://github.com/EricLBuehler/candle-vllm) 实现，兼容OpenAI API接口。

---

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README-CN.md">简体中文</a> |
</p>

## 🚀 快速开始

### 📦 环境依赖

```bash
# 安装燧原驱动与运行时环境
sudo ./TopsPlatform_1.7.*.run
dpkg -i eccl_3.6.*.deb
dpkg -i topsaten_3.6.*_amd64.deb
```

### 📦 方式一 — 安装预编译包（推荐）

从 [GitHub Releases](https://github.com/EnflameTechnology/candle-vllm-gcu/releases) 下载 GCU 预编译 `.deb`（资源名类似 `candle-vllm_*_amd64.deb`），然后安装：

```bash
# 示例（请替换为最新 Release 中的资源版本）：
wget https://github.com/EnflameTechnology/candle-vllm-gcu/releases/download/v0.8.8/candle-vllm_0.8.8-1_amd64.deb
sudo dpkg -i candle-vllm_0.8.8-1_amd64.deb

# 二进制安装到 /usr/local/bin/candle-vllm
candle-vllm --p 2000 --w /path/to/model --ui-server
```

### 🔧 方式二 — 从源码安装

```bash
# 安装 Rust（需版本 1.88.0 或更高）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install bindgen
cargo install bindgen-cli

# 更新子项目
git submodule update --init --recursive
cd candle-vllm

# 安装（之后可直接使用 `candle-vllm`）, aten特性（可选）与 graph 特性不兼容
cargo install --features gcu,eccl,aten --path .
```
---

## ✅ 支持的特性

- ✅ **多卡与多机并行推理**（Multi-GPUs, Multi-Nodes）
- ✅ **量化支持**（GPTQ、AWQ）
- ✅ **连续批处理**
- ✅ **分页注意力机制**
- ✅ **分块预填充**
- ✅ **KV 缓存支持**
  - ✅ BF16
  - ✅ FP16
  - ❌ 暂不支持 INT8
- ✅ **KVCache / GDN State Offloading**（显存不足时将 KV 缓存与 GDN/Mamba 状态换出到 CPU）
- ✅ **兼容 OpenAI 接口的服务**
- ❌ **多模态模型**
- 🛠️ **CUDA Graph** _(开发中)_


## ⚙️ 启动参数说明

- [`ENV_PARAM`] candle-vllm [`PROGRAM_PARAM`] [`MODEL_ID/MODEL_WEIGHT_PATH`] [`MODEL_TYPE`] [`MODEL_PARAM`]  
  <details open>
    <summary>显示详情</summary>

    **示例:**

    ```shell
    [RUST_LOG=warn] candle-vllm [--log --dtype bf16 --p 2000 --d 0,1 --kv-fraction 0.6 --ui-server] [--w /home/weights/QwQ-32B]
    ```

    `ENV_PARAM`: RUST_LOG=warn

    `PROGRAM_PARAM`：--log --dtype bf16 --p 2000 --d 0,1 --ui-server --kv-fraction 0.6

    `MODEL_WEIGHT_PATH`: --w /home/weights/QwQ-32B

    其中， `--p`: 服务端口; `--d`: 设备序列号; `--w`: 权重路径 (safetensors路径); `--f`: 权重文件 (GGUF模型使用); `--m`: Huggingface model-id; `--kv-fraction`：模型加载后按剩余显存比例自动分配 KV Cache（默认约 `0.6`，长文本或大批量可适当增大，如 `0.8`）; `--prefill-chunk-size`指定分块prefill时的块大小（默认8K，`0`为禁用）; `--ui-server`同时启动ChatGPT风格前端网站。
  </details>

---

## 🎥 聊天演示视频

**🔷 Qwen3-30B-MoE BF16（Enflame S60）** <img src="resources/Qwen3-30B-MoE-S60-Candle-vLLM-GCU.gif" width="100%">

---

## 📊 模型支持与性能

当前支持在 **Enflame S60 (48GB)** 上运行的模型如下（非Graph模式）：

**1k tokens 推理长度输出统计：**

| Model ID | Model Type | Supported | Speed (BF16, bs=1)| Thoughput (BF16, bs=16) |
|--|--|--|--|--|
| #1 | **LLAMA** |✅|30 tks/s (7B), 27 tks/s (LLaMa3.1 8B)| 375 tks/s (LLaMa3.1 8B) |
| #2 | **Mistral** |✅|29 tks/s (7B)|330 tks/s (7B)|
| #3 | **Phi (v1, v1.5, v2)** |✅|TBD|TBD|
| #4 | **Phi-3** |✅|38 tks/s (3.8B)|320 tks/s (BF16+F32, 7B)|
| #5 | **Yi** |✅|28 tks/s (6B)|305 tks/s (6B)|
| #6 | **StableLM** |✅|48 tks/s (3B)|425 tks/s (BF16, 3B)|
| #7 | BigCode/StarCode |TBD|TBD|TBD|
| #8 | ChatGLM |TBD|TBD|TBD|
| #9 | **QWen2** |✅|25 tks/s (14B, **tp=2**)|322 tks/s (14B, **tp=2, bs=32**)|
| #9 | **Qwen3** |✅|28 tks/s (8B, **bs=1**)|607 tks/s (14B, **bs=48**)|
| #10 | **Qwen3-MoE** |✅|48 tks/s (30B, **tp=2**)| 175 tks/s (30B, **tp=2**, **bs=8**)|
| #11 | **Qwen3.5/3.6 35B** |✅|35 tks/s (35B, **tp=2**)|TBD|
| #12 | **Google Gemma** |✅|73 tks/s (2B)| 920 tks/s (2B) |
| #13 | GLM4 |✅|TBD|TBD|
| #14 | Moondream-2 (Multimodal LLM) |TBD|TBD|TBD|
| #15 | **DeepSeek-V3/R1** (awq 671/685B, offloading) |✅|~8tks/s (**tp=8**)|155tks/s (**tp=8, bs=48**)|
| #16 | **QwQ-32B** |✅|15 tokens (**tp=2**)|230 tokens (**tp=2, bs=32**)|
---

## 💡 使用示例

<details open>
<summary><strong>运行未压缩模型</strong></summary>

```bash
candle-vllm --p 2000 \
  --w /home/DeepSeek-R1-Distill-Llama-8B/ --ui-server
```

```bash
candle-vllm --w /home/Qwen3-30B-A3B-Instruct-2507/ --d 0,1 --ui-server
```

Qwen3.5/3.6

```bash
candle-vllm --m Qwen/Qwen3.5-35B-A3B --d 0,1 --ui-server
```

</details>

<details open>
<summary><strong>运行 GPTQ 量化模型</strong></summary>

```bash
# 格式转换 （8bit gptq -> Enflame format）
python3 transform_safetensors.py --src /path/to/gptq \
--dst /path/to/gptq-enflame --bits 8 --method gptq --group 128 --nk True

#运行格式转换后的模型
candle-vllm --dtype bf16 --p 2000 --w /path/to/gptq-enflame --ui-server
```

</details>

<details open>
<summary><strong>运行 AWQ 量化模型</strong></summary>

```bash
# 格式转换 （4bit awq -> Enflame format）
python3 transform_safetensors.py --src /path/to/awq \
--dst /path/to/awq-enflame --bits 4 --method awq --group 64 --nk True

#运行格式转换后的模型
candle-vllm --dtype f16 --p 2000 --w /path/to/awq-enflame --ui-server
```

</details>

## 🖥️ 多卡与多节点推理支持

<details open>
<summary><strong>多进程多卡推理</strong></summary>

```bash
# 指定卡0和卡1
candle-vllm --p 2000 --d 0,1 --w /path/to/model --ui-server
```

</details>

<details open>
<summary><strong>多节点多卡推理（TCP，无需 MPI）</strong></summary>

跨多台机器做张量并行时，节点间通过 **TCP** 交换 ECCL UniqueId（用于建立集合通信），再由本地多进程完成各卡上的 ECCL 通信。

```bash
# 主节点 (例如 192.168.1.100)，在 candle-vllm 目录下：
candle-vllm --d 0,1,2,3,4,5,6,7 --w /data/deepseek-enflame \
  --num-nodes 2 --node-rank 0 --master-addr 192.168.1.100 --master-port 29500

# 工作节点 (例如 192.168.1.101)：
candle-vllm --d 0,1,2,3,4,5,6,7 --w /data/deepseek-enflame \
  --num-nodes 2 --node-rank 1 --master-addr 192.168.1.100 --master-port 29500
```

| 参数 | 说明 |
|------|------|
| `--num-nodes N` | 集群节点总数 |
| `--node-rank R` | 本机节点编号（`0` = 主节点） |
| `--master-addr ADDR` | 主节点 IP；工作节点必须填可连通的主节点地址。主节点若省略则绑定 `0.0.0.0` |
| `--master-port PORT` | ECCL ID 交换端口（默认 `29500`）；前向协调还会使用 `PORT+1`，请一并放行防火墙 |

**要求：**

- 各节点本地已有相同模型权重，并使用同一套 `candle-vllm` 二进制；各节点 `--d` 中的本地卡数应一致。
- 工作节点能 TCP 访问主节点的 `--master-port` 与 `--master-port + 1`。
- 全局世界大小 = `num_nodes × 每节点本地设备数`（例如 2 节点 × 8 卡 = tp=16）。

</details>

---

## 💬 客户端、Web UI 与 Agent

服务启动后提供 OpenAI 兼容 API（默认 `http://localhost:2000`）。可用交互聊天、内置 Web UI、并发压测或 Agent 自动化。

### 选项 1：交互式聊天（`chat.py`）

```bash
pip install openai rich click
python3 examples/chat.py
python3 examples/chat.py --live   # Markdown 实时渲染
```

### 选项 2：内置 Chat Web UI（`--ui-server`）

```bash
# 启动 API 的同时打开 ChatGPT 风格 Web UI（支持历史记录）
candle-vllm --p 2000 --w /path/to/model --ui-server
# 浏览器打开 UI：http://localhost:1999 （UI 端口 = API 端口 - 1）
```

### 选项 3：并发请求基准测试（`benchmark.py`）

```bash
# --batch 为请求总数别名；也可用 --num-prompts / --concurrency 等参数
python3 examples/benchmark.py --batch 16 --max_tokens 1024
python3 examples/benchmark.py --num-prompts 64 --concurrency 8 \
  --input-lens 128,512,2048 --output-lens 128,512
```

详见 [`benchmark.py`](candle-vllm/examples/benchmark.py)。

### 选项 4：xbot Agent（Vibe Coding / 后端自动化）

[`xbot`](candle-vllm/docs/xbot.md) 是 Rust AI Agent，可直连 candle-vllm 的 OpenAI 兼容接口做项目扫描、交互 REPL 或自动化任务。

```bash
# 1) 启动服务（示例端口 2000）
candle-vllm --p 2000 --w /path/to/model

# 2) 安装并配置 xbot
cargo install xbot   # 或: npm install -g @trusted-ai/xbot
xbot onboard
xbot config --provider
# Provider: custom → 名称 candle-vllm → API Base: http://localhost:2000/v1/ → 无需 API Key

# 3) 在项目中运行
cd YOUR_PROJECT
xbot chat /init
xbot chat "find bugs in this project."
xbot repl    # 交互式 TUI
```

完整说明见 [`docs/xbot.md`](candle-vllm/docs/xbot.md)。

---

## 📦 转换为 Enflame 量化格式

1. 使用 `transform_safetensors.py` 工具转换模型。
2. 示例：

```bash
# 8bit gptq 量化格式转换
python3 transform_safetensors.py --src /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit --dst /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit-Enflame --bits 8 --method gptq --group 128 --nk True

# 4bit awq 量化格式转换
python3 transform_safetensors.py --src /data/DeepSeek-R1-AWQ --dst /data/DeepSeek-R1-AWQ-Enflame/ --bits 4 --method awq --group 64 --nk True

# 运行格式转换后模型
candle-vllm --p 2000 --w /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit-Enflame
```
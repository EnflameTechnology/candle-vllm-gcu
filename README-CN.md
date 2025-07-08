# 🕯️ Candle-vLLM-GCU

> **面向燧原GCU平台的大语言模型推理与聊天服务框架**，其基于 `Candle-GCU` 和 开源项目[`Candle-vLLM`](https://github.com/EricLBuehler/candle-vllm) 实现，兼容OpenAI API接口。

---

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README-CN.md">简体中文</a> |
</p>

## 🚀 快速开始

### 🔧 构建 Candle-vLLM-GCU

```bash
# 安装 Rust（需版本 1.83.0 或更高）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装系统依赖项
sudo apt install libssl-dev pkg-config -y

# 安装燧原驱动与运行时环境
sudo ./TopsPlatform_1.4.xxxx.run
dpkg -i eccl_3.4.xxx_amd64.deb

# 克隆并构建项目
git clone git@git.enflame.cn:era/candle-vllm-gcu.git
cd candle-vllm-gcu
git submodule update --init --recursive
cd candle-vllm

# 构建适用于单节点（单卡或单机多卡）
cargo build --release --features gcu,eccl

# 构建适用于多节点（MPI，多机多卡）支持版本
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin clang libclang-dev -y
cargo build --release --features gcu,eccl,mpi
```

---

## ⚙️ 启动参数说明

**命令格式：**

```bash
[ENV_PARAM] cargo run [BUILD_PARAM] -- [PROGRAM_PARAM] [MODEL_WEIGHT_PATH] [MODEL_TYPE] [MODEL_PARAM]
```

**示例：**

```bash
RUST_LOG=warn cargo run --release --features gcu,eccl -- \
--multi-process --log --dtype bf16 --port 2000 --device-ids "0,1" --kvcache-mem-gpu 8192 \
--weight-path /home/weights/QwQ32B-GPTQ-4Bit \
qwen2 --quant gptq --temperature 0.7 --penalty 1.0 --top-k 40 --top-p 0.95
```

支持的 `MODEL_TYPE` 类型：

```
["llama", "llama3", "mistral", "phi2", "phi3", "qwen2", "qwen3", "gemma", "yi", "stable-lm", "deep-seek"]
```

---

## 🎥 聊天演示视频

**🔷 DeepSeek-R1 685B（AWQ，约 8 tokens/s，8 张 Enflame S60，约 120GB 权重卸载到 CPU内存）** <img src="resources/DeepSeek-R1-685B-S60-Candle-vLLM-GCU.gif" width="1024pt" height="695pt" >

**🔷 LLaMa3.1 8B（AWQ，约 40 tokens/s，1 张 Enflame S60）** <img src="resources/LLaMa3.1-8B-S60-Quant.gif" width="85%" height="85%" >

---

## 📊 模型支持与性能

当前支持在 **Enflame S60 (48GB)** 上运行的模型如下：

**1k tokens 推理长度输出统计：**

| Model ID | Model Type | Supported | Speed (BF16, bs=1)| Thoughput (BF16, bs=16) | Thoughput (W4A16)
|--|--|--|--|--|--|
| #1 | **LLAMA** |✅|30 tks/s (7B), 27 tks/s (LLaMa3.1 8B)| 375 tks/s (LLaMa3.1 8B) | 41 tks/s (**bs=1**), 1185 tks/s (**bs=48**)|
| #2 | **Mistral** |✅|29 tks/s (7B)|330 tks/s (7B)|TBD|
| #3 | **Phi (v1, v1.5, v2)** |✅|TBD|TBD|TBD|
| #4 | **Phi-3** |✅|38 tks/s (3.8B)|320 tks/s (BF16+F32, 7B)|TBD|
| #5 | **Yi** |✅|28 tks/s (6B)|305 tks/s (6B)|TBD|
| #6 | **StableLM** |✅|48 tks/s (3B)|425 tks/s (BF16, 3B)|TBD|
| #7 | BigCode/StarCode |TBD|TBD|TBD|
| #8 | ChatGLM |TBD|TBD|TBD|
| #9 | **QWen2** |✅|22 tks/s (14B, **tp=2**)|322 tks/s (14B, **tp=2, bs=32**)|TBD|
| #9 | **Qwen3** |✅|23 tks/s (8B, **bs=1**)|607 tks/s (14B, **bs=48**)|TBD|
| #10 | **Google Gemma** |✅|51 tks/s (2B)| 577 tks/s (2B) |TBD|
| #11 | GLM4 |✅|TBD|TBD|
| #12 | Moondream-2 (Multimodal LLM) |TBD|TBD|TBD|
| #13 | **DeepSeek-V3/R1** (awq 671/685B, offloading) |✅|~8tks/s (**tp=8**)|155tks/s (**tp=8, bs=48**)|TBD|
| #14 | **QwQ-32B** |✅|10.6 tokens (**tp=2**)|214 tokens (**tp=2, bs=32**)|TBD|
---

## 💡 使用示例

<details>
<summary><strong>运行未压缩模型</strong></summary>

```bash
target/release/candle-vllm --port 2000 \
--weight-path /home/DeepSeek-R1-Distill-Llama-8B/ \
llama3 --temperature 0. --penalty 1.0
```

</details>

<details>
<summary><strong>运行 GPTQ 量化模型</strong></summary>

```bash
python3 transform_safetensors.py --src /path/to/gptq \
--dst /path/to/gptq-enflame --bits 8 --method gptq --group 128 --nk True

target/release/candle-vllm --dtype bf16 --port 2000 \
--weight-path /path/to/gptq-enflame qwen2 --quant gptq \
--temperature 0. --penalty 1.0
```

</details>

<details>
<summary><strong>运行 AWQ 量化模型</strong></summary>

```bash
python3 transform_safetensors.py --src /path/to/awq \
--dst /path/to/awq-enflame --bits 4 --method awq --group 64 --nk True

target/release/candle-vllm --multi-process --dtype f16 --port 2000 \
--device-ids "0" --weight-path /path/to/awq-enflame llama3 \
--quant awq --temperature 0. --penalty 1.0
```

</details>

---

## 🧠 原位量化（实验功能）

将原始权重直接转换为 Enflame 格式并加载运行：

```bash
cargo run --release --features gcu -- --port 2000 \
--weight-path /home/Meta-Llama-3.1-8B-Instruct/ \
llama3 --quant q8_0
```

> ⚠️ *提示：`q4_k` 是更理想的量化格式，目前批处理性能仍在优化中。*

---

## 🖥️ 多卡与多节点推理支持

<details>
<summary><strong>多进程多卡推理</strong></summary>

```bash
target/release/candle-vllm --multi-process --port 2000 \
--device-ids "0,1" --weight-path /path/to/model llama3 \
--temperature 0. --penalty 1.0
```

</details>

<details>
<summary><strong>多节点（MPI）配置</strong></summary>

```bash
# 安装 MPI
sudo apt install libopenmpi-dev openmpi-bin clang libclang-dev -y

# 构建支持 MPI 的版本
cargo build --release --features gcu,eccl,mpi

# 使用 mpirun 启动
sudo mpirun -np 16 -x RUST_LOG=info -hostfile ./hostfile \
--allow-run-as-root -bind-to none -map-by slot \
--mca btl_tcp_if_include %NET_INTERFACE% \
target/release/candle-vllm --multi-process --dtype bf16 --port 2000 \
--device-ids "0,1,2,3,4,5,6,7" --weight-path /data/deepseek-enflame \
deep-seek --quant awq --temperature 0. --penalty 1.0
```

</details>

---

## 💬 聊天前端支持

### 选项 1：使用 `chat.py` 快速测试

```bash
pip install openai rich click
python3 examples/chat.py
python3 examples/chat.py --live # Markdown 实时渲染支持
```

### 选项 2：使用 Chat UI

```bash
git clone https://github.com/guoqingbao/candle-vllm-demo.git
cd candle-vllm-demo
apt install npm
npm install -g n && n stable
npm install -g pnpm
pnpm install
pnpm run dev
```

---

## 📈 性能基准测试

运行批处理基准测试：

```bash
python3 examples/benchmark.py --batch 16 --max_tokens 1024
```

详见脚本 [`benchmark.py`](candle-vllm/examples/benchmark.py)。

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
cargo run --release --features gcu -- --port 2000 \
--weight-path /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit-Enflame llama3 --quant gptq
```

---

## 🛠️ 开发计划（TODO）

* [x] 优化生成速度。
* [ ] 增加更多量化格式支持（如 `q4_k`、`w4a16`）。
* [x] 支持多用户同时对话。
* [ ] 支持多模态模型。

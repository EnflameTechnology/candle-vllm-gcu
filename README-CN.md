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
# 安装 Rust（需版本 1.88.0 或更高）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装系统依赖项
sudo apt install libssl-dev pkg-config -y

# 安装燧原驱动与运行时环境
sudo ./TopsPlatform_1.7.*.run
dpkg -i eccl_3.6.*.deb

# Install bindgen
cargo install bindgen-cli

# 更新子项目
git submodule update --init --recursive
cd candle-vllm

# 基础构建
cargo build --release --features gcu,eccl

# +CUDA Graph特性
cargo build --release --features gcu,eccl,graph
```

#### 构建使用TopsAten + Flash Attention（可选）
[下载](https://github.com/EnflameTechnology/candle-vllm-gcu/releases/download/v0.8.6/flash-attn-gcu_0.1.0-1_amd64.deb)并安装GCU Flash Attention

```bash
dpkg -i topsaten_3.6.*_amd64.deb # 安装topsaten算子库
dpkg -i flash-attn-gcu_0.1.0-1_amd64.deb # 安装 flash attn算子库
# 启用 falshattn 特性以加速Prefill速度 (与graph特性不兼容)
cargo build --release --features gcu,eccl,aten,flashattn
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
- ✅ **兼容 OpenAI 接口的服务**
- ❌ **多模态模型**
- 🛠️ **CUDA Graph** _(开发中)_


## ⚙️ 构建及启动参数说明

- [`ENV_PARAM`] cargo run [`BUILD_PARAM`] -- [`PROGRAM_PARAM`] [`MODEL_ID/MODEL_WEIGHT_PATH`] [`MODEL_TYPE`] [`MODEL_PARAM`]  
  <details open>
    <summary>显示详情</summary>

    **示例:**

    ```shell
    [RUST_LOG=warn] cargo run [--release --features gcu,eccl,flashattn] -- [--log --dtype bf16 --p 2000 --d 0,1 --mem 8192 --ui-server] [--weight-path /home/weights/QwQ-32B]
    ```

    `ENV_PARAM`: RUST_LOG=warn

    `BUILD_PARAM`: --release --features gcu,eccl,flashattn

    `PROGRAM_PARAM`：--log --dtype bf16 --p 2000 --d 0,1 --ui-server --mem 8192

    `MODEL_WEIGHT_PATH`: --w /home/weights/QwQ-32B

    其中， `--p`: 服务端口; `--d`: 设备序列号; `--w`: 权重路径 (safetensors路径); `--f`: 权重文件 (GGUF模型使用); `--m`: Huggingface model-id; `--mem`参数控制KV Cache缓存，长文本或批量推理量请增大缓存; `--prefill-chunk-size`指定分块prefill时的块大小（默认8K，`0`为禁用）; `--ui-server`同时启动ChatGPT风格前端网站。
  </details>

---

## 🎥 聊天演示视频

**🔷 DeepSeek-R1 685B（AWQ，约 8 tokens/s，8 张 Enflame S60-48G，约 120GB 权重卸载到 CPU内存）** <img src="resources/DeepSeek-R1-685B-S60-Candle-vLLM-GCU.gif" width="85%">

**🔷 LLaMa3.1 8B（AWQ，约 40 tokens/s，1 张 Enflame S60-48G）** <img src="resources/LLaMa3.1-8B-S60-Quant.gif" width="85%">

---

## 📊 模型支持与性能

当前支持在 **Enflame S60 (48GB)** 上运行的模型如下（非Graph模式）：

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
| #10 | **Qwen3-MoE** |✅|41 tks/s (30B, **tp=2**)|TBD|TBD|
| #11 | **Qwen3.5/3.6 35B** |✅|30 tks/s (35B, **tp=2**)|TBD|TBD|
| #12 | **Google Gemma** |✅|51 tks/s (2B)| 577 tks/s (2B) |TBD|
| #13 | GLM4 |✅|TBD|TBD|
| #14 | Moondream-2 (Multimodal LLM) |TBD|TBD|TBD|
| #15 | **DeepSeek-V3/R1** (awq 671/685B, offloading) |✅|~8tks/s (**tp=8**)|155tks/s (**tp=8, bs=48**)|TBD|
| #16 | **QwQ-32B** |✅|10.6 tokens (**tp=2**)|214 tokens (**tp=2, bs=32**)|TBD|
---

## 💡 使用示例

<details open>
<summary><strong>运行未压缩模型</strong></summary>

```bash
target/release/candle-vllm --p 2000 \
  --w /home/DeepSeek-R1-Distill-Llama-8B/ --ui-server
```

```bash
target/release/candle-vllm --w /home/Qwen3-30B-A3B-Instruct-2507/ --d 0,1 --ui-server
```

Qwen3.5/3.6

```bash
target/release/candle-vllm --m Qwen/Qwen3.5-35B-A3B --d 0,1 --ui-server
```

</details>

<details open>
<summary><strong>运行 GPTQ 量化模型</strong></summary>

```bash
# 格式转换 （8bit gptq -> Enflame format）
python3 transform_safetensors.py --src /path/to/gptq \
--dst /path/to/gptq-enflame --bits 8 --method gptq --group 128 --nk True

#运行格式转换后的模型
target/release/candle-vllm --dtype bf16 --p 2000 --w /path/to/gptq-enflame --ui-server
```

</details>

<details open>
<summary><strong>运行 AWQ 量化模型</strong></summary>

```bash
# 格式转换 （4bit awq -> Enflame format）
python3 transform_safetensors.py --src /path/to/awq \
--dst /path/to/awq-enflame --bits 4 --method awq --group 64 --nk True

#运行格式转换后的模型
target/release/candle-vllm --dtype f16 --p 2000 --w /path/to/awq-enflame --ui-server
```

</details>

## 🖥️ 多卡与多节点推理支持

<details open>
<summary><strong>多进程多卡推理</strong></summary>

```bash
# 指定卡0和卡1
target/release/candle-vllm --p 2000 --d 0,1 --w /path/to/model --ui-server
```

</details>

<details>
<summary><strong>多节点多卡推理（MPI）配置</strong></summary>

```bash
# 安装 MPI
sudo apt install libopenmpi-dev openmpi-bin clang libclang-dev -y

# 构建支持 MPI 的版本
cargo build --release --features gcu,eccl,mpi

# 使用 mpirun 启动（确保双机中权重与candle-vllm binary在相同目录）
sudo mpirun -np 16 -x RUST_LOG=info -hostfile ./hostfile \
--allow-run-as-root -bind-to none -map-by slot \
--mca btl_tcp_if_include %NET_INTERFACE% \
target/release/candle-vllm --dtype bf16 --p 2000 \
--d 0,1,2,3,4,5,6,7 --w /data/deepseek-enflame deep-seek
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

### 选项 2：使用 Chat UI (带聊天记录功能)

```bash
# 安装Rust aichat
cargo install aichat

aichat --serve
# 选择 `openai-compatible`模式, provide name 填写`candle-vllm`
# 将 candle-vllm API 地址（如 http://0.0.0.0:2000/v1/）填入， (API Key为空, `LLMs to include`选择default)
# 点击 aichat生成的 "LLM Playground" URL地址即可
```

https://github.com/user-attachments/assets/6fbad80b-e4d8-453f-b50d-50f61fa8c4f3

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
cargo run --release --features gcu -- --p 2000 \
--w /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit-Enflame
```

---

## 🛠️ 开发计划（TODO）

* [ ] 增加GGUF模型支持（如 `q4_k`量化格式）。
* [ ] 支持多模态模型。

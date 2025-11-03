# 🕯️ Candle-vLLM-GCU

> **A large language model inference and chat service framework designed for Enflame GCU**, built on top of `Candle-GCU` and the open-source project [`Candle-vLLM`](https://github.com/EricLBuehler/candle-vllm), and fully compatible with the OpenAI API.

---

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README-CN.md">简体中文</a> |
</p>

## 🚀 Getting Started

### 🔧 Build Candle-VLLM-GCU

```bash
# Install Rust (version 1.83.0 or higher)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install required system dependencies
sudo apt install libssl-dev pkg-config -y

# Install Enflame's drivers and runtime
sudo ./TopsPlatform_1.4.5.xxxx.run
dpkg -i eccl_3.5.xxx_amd64.deb

# Install bindgen
cargo install bindgen-cli

# Clone and build the project
git clone git@git.enflame.cn:era/candle-vllm-gcu.git
cd candle-vllm-gcu
git submodule update --init --recursive
cd candle-vllm

# Build for single-node setup
cargo build --release --features gcu,eccl

# Build for multi-node support (MPI)
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin clang libclang-dev -y
cargo build --release --features gcu,eccl,mpi
```

---

## ✅ Supported Features

- ✅ **Multi-rank** (Multi-GPUs, Multi-Nodes)
- ✅ **Quantization** (GPTQ, AWQ)
- ✅ **Continuous Batching**
- ✅ **Paged Attention**
- ✅ **Chunked Prefill**
- ✅ **KV Cache**
  - ✅ BF16
  - ✅ FP16
  - ❌ INT8
- ✅ **OpenAI-Compatible Server**
- ❌ **Multimodal Models** 
- 🛠️ **CUDA Graph** _(Under Development)_

## ⚙️ Build and Running Parameters
- [`ENV_PARAM`] cargo run [`BUILD_PARAM`] -- [`PROGRAM_PARAM`] [`MODEL_ID/MODEL_WEIGHT_PATH`]
  <details open>
    <summary>Show details</summary>

    **Example:**

    ```shell
    [RUST_LOG=warn] cargo run [--release --features gcu,eccl] -- [--log --dtype bf16 --p 2000 --d 0,1 --mem 8192] [--w /home/weights/QwQ-32B/]
    ```

    `ENV_PARAM`: RUST_LOG=warn

    `BUILD_PARAM`: --release --features gcu,eccl

    `PROGRAM_PARAM`：--log --dtype bf16 --p 2000 --d 0,1 --mem 8192

    `MODEL_WEIGHT_PATH`: --w /home/weights/QwQ-32B

    where, `--p`: server port; `--d`: device ids; `--w`: weight path (safetensors folder); `--f`: weight file (for gguf); `--m`: huggingface model-id; `--mem` is the key parameter to control KV cache usage (increase this for large batch); `--prefill-chunk-size` chunk the prefill into size defined in this flag (default 8K, `0` for disable).
  </details>

---

## 🎥 Demo Chat Videos

**🔷 DeepSeek-R1 685B (AWQ, \~8 tokens/s, 8 x Enflame S60, offloaded \~120GB to CPU)** <img src="resources/DeepSeek-R1-685B-S60-Candle-vLLM-GCU.gif" width="1024pt" height="695pt" >

**🔷 LLaMa3.1 8B (AWQ, \~40 tokens/s, 1 x Enflame S60)** <img src="resources/LLaMa3.1-8B-S60-Quant.gif" width="85%" height="85%" >

---

## 📊 Model Support & Performance

Currently supported models on **Enflame S60 (48GB)**:

__List of 1k decoding results:__
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

## 💡 Usage Examples

<details open>
<summary><strong>Run Uncompressed Models</strong></summary>

```bash
target/release/candle-vllm --p 2000 --w /home/DeepSeek-R1-Distill-Llama-8B/
```

</details>

<details open>
<summary><strong>Run GPTQ Quantized Models</strong></summary>

```bash
# convert (8bit gptq) model to Enflame format
python3 transform_safetensors.py --src /path/to/gptq \
--dst /path/to/gptq-enflame --bits 8 --method gptq --group 128 --nk True

# run the converted model
target/release/candle-vllm --dtype bf16 --p 2000 --w /path/to/gptq-enflame
```

</details>

<details open>
<summary><strong>Run AWQ Quantized Models</strong></summary>

```bash
# convert (4bit awq) model to Enflame format
python3 transform_safetensors.py --src /path/to/awq \
--dst /path/to/awq-enflame --bits 4 --method awq --group 64 --nk True

# run the converted model
target/release/candle-vllm --dtype f16 --p 2000 --w /path/to/awq-enflame
```

</details>

## 🖥️ Multi-GPU & Multi-Node Inference

<details open>
<summary><strong>Multi-Process, Multi-GPU</strong></summary>

```bash
# Use card 0 and card 1
target/release/candle-vllm --p 2000 --d 0,1 --weight-path /path/to/model
```

</details>

<details>
<summary><strong>Multi-Node (MPI) Setup</strong></summary>

```bash
# Install MPI
sudo apt install libopenmpi-dev openmpi-bin clang libclang-dev -y

# Build
cargo build --release --features gcu,eccl,mpi

# Launch via mpirun (make sure that model weights and candle-vllm binary located in the same folder in different machines)
sudo mpirun -np 16 -x RUST_LOG=info -hostfile ./hostfile \
--allow-run-as-root -bind-to none -map-by slot \
--mca btl_tcp_if_include %NET_INTERFACE% \
target/release/candle-vllm --dtype bf16 --p 2000 \
--d 0,1,2,3,4,5,6,7 --w /data/deepseek-enflame
```

</details>

---

## 💬 Chat Frontends

### Option 1: Quick Test via `chat.py`

```bash
pip install openai rich click
python3 examples/chat.py
python3 examples/chat.py --live # with markdown support
```

### Option 2: Chat UI

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

## 📈 Benchmarking

Run batched benchmark tests:

```bash
python3 examples/benchmark.py --batch 16 --max_tokens 1024
```

Refer to the [`benchmark.py`](candle-vllm/examples/benchmark.py) script for async chat example.

---

## 📦 Quantization to Enflame Format

1. Use `transform_safetensors.py` to convert models.
2. Samples:

```bash
# 8bit gptq conversion
python3 transform_safetensors.py --src /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit --dst /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit-Enflame --bits 8 --method gptq --group 128 --nk True

# 4bit awq conversion
python3 transform_safetensors.py --src /data/DeepSeek-R1-AWQ --dst /data/DeepSeek-R1-AWQ-Enflame/ --bits 4 --method awq --group 64 --nk True

# run the converted model
cargo run --release --features gcu -- --p 2000 \
--w /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit-Enflame
```

---

## 🛠️ TODO

* [ ] Add GGUF model support (e.g., `q4_k` quantization).
* [ ] Extend support to multimodal models.

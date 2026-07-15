# 🕯️ Candle-vLLM-GCU

> **A large language model inference and chat service framework designed for Enflame GCU**, built on top of `Candle-GCU` and the open-source project [`Candle-vLLM`](https://github.com/EricLBuehler/candle-vllm), and fully compatible with the OpenAI API.

---

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README-CN.md">简体中文</a> |
</p>

## 🚀 Getting Started

### 📦 Prerequisites

```bash
# Install Enflame drivers and runtime
sudo ./TopsPlatform_1.7.*.run
dpkg -i eccl_3.6.*.deb
dpkg -i topsaten_3.6.*_amd64.deb
```

### 📦 Option 1 — Install prebuilt package (recommended)

Download the GCU `.deb` from the [GitHub Releases](https://github.com/EnflameTechnology/candle-vllm-gcu/releases) page (asset name like `candle-vllm_*_amd64.deb`), then install:

```bash
# Example (replace with the asset version from the latest release):
wget https://github.com/EnflameTechnology/candle-vllm-gcu/releases/download/v0.8.8/candle-vllm_0.8.8-1_amd64.deb
sudo dpkg -i candle-vllm_0.8.8-1_amd64.deb

# Binary installs to /usr/local/bin/candle-vllm
candle-vllm --p 2000 --w /path/to/model --ui-server
```

### 🔧 Option 2 — Install from source

```bash
# Install Rust (version 1.88.0 or higher)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install bindgen
cargo install bindgen-cli

# Update sub-project
git submodule update --init --recursive
cd candle-vllm

# Install (then use `candle-vllm` directly); aten is optional and incompatible with graph
cargo install --features gcu,eccl,aten --path .
```

---

## ✅ Supported Features

- ✅ **Multi-rank** (Multi-GPUs, Multi-Nodes)
- ✅ **Quantization** (GPTQ, AWQ)
- ✅ **Continuous Batching**
- ✅ **Paged Attention**
- ✅ **Flash Attention**
- ✅ **Chunked Prefill**
- ✅ **Prefix Caching**
- ✅ **KV Cache**
  - ✅ BF16
  - ✅ FP16
  - ❌ INT8
- ✅ **KVCache / GDN State Offloading** (swap KV cache and GDN/Mamba state to CPU under GPU/GCU memory pressure)
- ✅ **OpenAI-Compatible Server**
- ❌ **Multimodal Models** 
- 🛠️ **CUDA Graph** _(Under Development)_

## ⚙️ Running Parameters
- [`ENV_PARAM`] candle-vllm [`PROGRAM_PARAM`] [`MODEL_ID/MODEL_WEIGHT_PATH`]
  <details open>
    <summary>Show details</summary>

    **Example:**

    ```shell
    [RUST_LOG=warn] candle-vllm [--log --dtype bf16 --p 2000 --d 0,1 --kv-fraction 0.6 --ui-server] [--w /home/weights/QwQ-32B/]
    ```

    `ENV_PARAM`: RUST_LOG=warn

    `PROGRAM_PARAM`：--log --dtype bf16 --p 2000 --d 0,1 --ui-server --kv-fraction 0.6

    `MODEL_WEIGHT_PATH`: --w /home/weights/QwQ-32B

    where, `--p`: server port; `--d`: device ids; `--w`: weight path (safetensors folder); `--f`: weight file (for gguf); `--m`: huggingface model-id; `--kv-fraction` auto-sizes the KV cache as a fraction of remaining device memory after model load (default about `0.6`; raise it, e.g. `0.8`, for longer context or larger batches); `--prefill-chunk-size` chunk the prefill into size defined in this flag (default 8K, `0` for disable); `--ui-server` start with a ChatGPT-like WebUI.
  </details>

---

## 🎥 Demo Chat Videos

**🔷 Qwen3-30B-MoE BF16 (Enflame S60)** <img src="resources/Qwen3-30B-MoE-S60-Candle-vLLM-GCU.gif" width="100%">

---

## 📊 Model Support & Performance

Currently supported models on **Enflame S60 (48GB)** (under graph mode disabled):

__List of 1k decoding results:__
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
| #9 | **QWen2** |✅|22 tks/s (14B, **tp=2**)|322 tks/s (14B, **tp=2, bs=32**)|
| #9 | **Qwen3** |✅|23 tks/s (8B)|607 tks/s (14B, **bs=48**)|
| #10 | **Qwen3-MoE** |✅|41 tks/s (30B, **tp=2**)|TBD|
| #11 | **Qwen3.5/3.6 35B** |✅|30 tks/s (35B, **tp=2**)|TBD|
| #12 | **Google Gemma** |✅|51 tks/s (2B)| 577 tks/s (2B) |
| #13 | GLM4 |✅|TBD|TBD|
| #14 | Moondream-2 (Multimodal LLM) |TBD|TBD|TBD|
| #15 | **DeepSeek-V3/R1** (awq 671/685B, offloading) |✅|~8tks/s (**tp=8**)|155tks/s (**tp=8, bs=48**)|
| #16 | **QwQ-32B** |✅|15 tokens (**tp=2**)|230 tokens (**tp=2, bs=32**)|
---

## 💡 Usage Examples

<details open>
<summary><strong>Run Uncompressed Models</strong></summary>

```bash
candle-vllm --p 2000 --w /home/DeepSeek-R1-Distill-Llama-8B/ --ui-server
```

```bash
candle-vllm --w /home/Qwen3-30B-A3B-Instruct-2507/ --d 0,1
```

Qwen3.5/3.6

```bash
candle-vllm --m Qwen/Qwen3.5-35B-A3B --d 0,1 --ui-server
```

</details>

<details open>
<summary><strong>Run GPTQ Quantized Models</strong></summary>

```bash
# convert (8bit gptq) model to Enflame format
python3 transform_safetensors.py --src /path/to/gptq \
--dst /path/to/gptq-enflame --bits 8 --method gptq --group 128 --nk True

# run the converted model
candle-vllm --p 2000 --w /path/to/gptq-enflame --ui-server
```

</details>

<details open>
<summary><strong>Run AWQ Quantized Models</strong></summary>

```bash
# convert (4bit awq) model to Enflame format
python3 transform_safetensors.py --src /path/to/awq \
--dst /path/to/awq-enflame --bits 4 --method awq --group 64 --nk True

# run the converted model
candle-vllm --p 2000 --w /path/to/awq-enflame --ui-server
```

</details>

## 🖥️ Multi-GPU & Multi-Node Inference

<details open>
<summary><strong>Multi-Process, Multi-GPU</strong></summary>

```bash
# Use card 0 and card 1
candle-vllm --p 2000 --d 0,1 --w /path/to/model --ui-server
```

</details>

<details open>
<summary><strong>Multi-Node (TCP, no MPI)</strong></summary>

For cross-machine tensor parallelism, nodes exchange the **ECCL UniqueId over TCP** (to bootstrap collective communication); local multi-process ECCL then handles per-device traffic. **MPI / `mpirun` is not required.** Build with `--features gcu,eccl` as usual.

```bash
# Master node (e.g. 192.168.1.100), from the candle-vllm directory:
candle-vllm --d 0,1,2,3,4,5,6,7 --w /data/deepseek-enflame \
  --num-nodes 2 --node-rank 0 --master-addr 192.168.1.100 --master-port 29500

# Worker node (e.g. 192.168.1.101):
candle-vllm --d 0,1,2,3,4,5,6,7 --w /data/deepseek-enflame \
  --num-nodes 2 --node-rank 1 --master-addr 192.168.1.100 --master-port 29500
```

| Flag | Description |
|------|-------------|
| `--num-nodes N` | Total nodes in the cluster |
| `--node-rank R` | Rank of this node (`0` = master) |
| `--master-addr ADDR` | Master IP; workers must set a reachable master address. If omitted on the master, it binds `0.0.0.0` |
| `--master-port PORT` | ECCL ID exchange port (default `29500`); forward coordination also uses `PORT+1` — open both in the firewall |

**Requirements:**

- Every node has the same model weights locally and the same `candle-vllm` binary; each node's local device count in `--d` should match.
- Workers can reach the master on `--master-port` and `--master-port + 1` over TCP.
- Global world size = `num_nodes × local devices per node` (e.g. 2 nodes × 8 cards = tp=16).

</details>

---

## 💬 Clients, Web UI & Agents

After the server starts, an OpenAI-compatible API is available (default `http://localhost:2000`). Use interactive chat, the built-in Web UI, concurrent benchmarks, or agent automation.

### Option 1: Interactive chat (`chat.py`)

```bash
pip install openai rich click
python3 examples/chat.py
python3 examples/chat.py --live   # live Markdown rendering
```

### Option 2: Built-in Chat Web UI (`--ui-server`)

```bash
# Start the API together with a ChatGPT-style Web UI (with chat history)
candle-vllm --p 2000 --w /path/to/model --ui-server
# Open the UI in a browser: http://localhost:1999 (UI port = API port - 1)
```

### Option 3: Concurrent request benchmark (`benchmark.py`)

```bash
# --batch is an alias for total prompts; also supports --num-prompts / --concurrency, etc.
python3 examples/benchmark.py --batch 16 --max_tokens 1024
python3 examples/benchmark.py --num-prompts 64 --concurrency 8 \
  --input-lens 128,512,2048 --output-lens 128,512
```

See [`benchmark.py`](candle-vllm/examples/benchmark.py) for the full option list.

### Option 4: xbot agent (vibe coding / backend automation)

[`xbot`](candle-vllm/docs/xbot.md) is a Rust AI agent that talks to candle-vllm’s OpenAI-compatible API for project scanning, interactive REPL, or automation.

```bash
# 1) Start the server (example port 8000)
candle-vllm --p 8000 --w /path/to/model

# 2) Install and configure xbot
cargo install xbot   # or: npm install -g @trusted-ai/xbot
xbot onboard
xbot config --provider
# Provider: custom → name candle-vllm → API Base: http://localhost:8000/v1/ → no API key

# 3) Run in your project
cd YOUR_PROJECT
xbot chat /init
xbot chat "find bugs in this project."
xbot repl    # interactive TUI
```

Full guide: [`docs/xbot.md`](candle-vllm/docs/xbot.md).

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
candle-vllm --p 2000 --w /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit-Enflame
```
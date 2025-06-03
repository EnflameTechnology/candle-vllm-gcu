# Candle-VLLM-GCU
OpenAI-API compatible chat service for `GCU` platform based on `candle-gcu` and `candle-vllm (paged attention)`


## Getting started

### Build Candle-vLLM-GCU

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh #install rust, 1.83.0+ required
sudo apt install libssl-dev pkg-config -y
sudo ./TopsPlatform_1.4xxxx.run # Install driver, topsruntime and topscc
dpkg -i eccl_3.4xxx_amd64.deb  # Install ECCL in the environment where candle-vllm-gcu resides.

git clone git@git.enflame.cn:era/candle-vllm-gcu.git
git submodule update --init --recursive
cd candle-vllm
cargo build --release --features gcu,eccl #single-node

#multinode
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin -y #install mpi
sudo apt install clang libclang-dev
cargo build --release --features gcu,eccl,mpi #build with mpi feature
```

### Build/Run Parameters

[`ENV_PARAM`] cargo run [`BUILD_PARAM`] -- [`PROGRAM_PARAM`] [`MODEL_ID/MODEL_WEIGHT_PATH`] [`MODEL_TYPE`] [`MODEL_PARAM`]

**Example:**
```shell
[RUST_LOG=warn] cargo run [--release --features gcu,eccl] -- [--multi-process --log --dtype bf16 --port 2000 --device-ids "0,1" --kvcache-mem-gpu 8192] [--weight-path /home/weights/QwQ32B-GPTQ-4Bit] [qwen2] [--quant gptq --temperature 0.7 --penalty 1.0 --top-k 40 --top-p 0.95]
```

`ENV_PARAM`: RUST_LOG=warn

`BUILD_PARAM`: --release --features gcu,eccl

`PROGRAM_PARAM`：--multi-process --log --dtype bf16 --port 2000 --device-ids "0,1" --kvcache-mem-gpu 8192

`MODEL_WEIGHT_PATH`: --weight-path /home/weights/QwQ32B-GPTQ-4Bit

`MODEL_TYPE`: qwen2

`MODEL_PARAM`: --quant gptq --temperature 0.7 --penalty 1.0 --top-k 40 --top-p 0.95

where, `MODEL_TYPE` in ["llama", "llama3", "mistral", "phi2", "phi3", "qwen2", "gemma", "yi", "stable-lm", "deep-seek"]

## Demo chat video 

**DeepSeek-R1 671/685B (AWQ, ~8 tokens/s, 8 x Scorpio S60 (8 x 48GB))** (offloaded ~120GB weights to CPU memory)
<img src="resources/DeepSeek-R1-685B-S60-Candle-vLLM-GCU.gif" width="1024pt" height="695pt" >

**LLaMa3.1 8B (AWQ, ~40 tokens/s, 1 x Scorpio S60)**
<img src="resources/LLaMa3.1-8B-S60-Quant.gif" width="85%" height="85%" >

## Status

Currently, candle-vllm-gcu supports chat serving for the following models on `S60`.

__`List of 1k decoding results:`__
| Model ID | Model Type | Supported | Speed (BF16, `batch size=1`)| Thoughput (BF16, `batch size=16`) | Thoughput (W8A16, `batch size=16`) | Thoughput (W4A16, `batch size=48`)
|--|--|--|--|--|--|--|
| #1 | **LLAMA** |✅|30 tks/s (7B), 27 tks/s (LLaMa3.1 8B)| 305 tks/s (LLaMa3.1 8B) | 375 tks/s (LLaMa3.1 8B) | 1060 tks/s|
| #2 | **Mistral** |✅|29 tks/s (7B)|330 tks/s (7B)|TBD|TBD|
| #3 | **Phi (v1, v1.5, v2)** |✅|TBD|TBD|TBD|
| #4 | **Phi-3** |✅|38 tks/s (3.8B)|320 tks/s (BF16+F32, 7B)|TBD|TBD|
| #5 | **Yi** |✅|28 tks/s (6B)|305 tks/s (6B)|TBD|TBD|
| #6 | **StableLM** |✅|48 tks/s (3B)|425 tks/s (BF16, 3B)|TBD|TBD|
| #7 | BigCode/StarCode |TBD|TBD|TBD|TBD|
| #8 | ChatGLM |TBD|TBD|TBD|TBD|
| #9 | **QWen2/Qwen3** |✅|22 tks/s (14B, **tp=2**)|322 tks/s (14B, **tp=2, bs=32**)|TBD|
| #10 | **Google Gemma** |✅|51 tks/s (2B)| 577 tks/s (2B) |TBD|TBD|
| #11 | GLM4 |✅|TBD|TBD|TBD|
| #12 | Moondream-2 (Multimodal LLM) |TBD|TBD|TBD|TBD|
| #13 | **DeepSeek-V3/R1** (awq 671/685B, `offloading`) |✅|~8tks/s (**tp=8**)|155tks/s (**tp=8, bs=48**)|TBD|TBD|
| #14 | **QwQ-32B** |✅|10 tokens (**tp=2**)|186 tokens (**tp=2, bs=32**)|TBD|TBD|

## Detailed Usage

Run **Uncompressed** models
```shell
target/release/candle-vllm --port 2000 --weight-path /home/DeepSeek-R1-Distill-Llama-8B/ llama3 --temperature 0. --penalty 1.0
```

Run **GPTQ models**
```shell
python3 transform_safetensors.py --src /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_8bit --dst /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_8bit-Enflame --bits 8 --method gptq --group 128 --nk True
target/release/candle-vllm --dtype bf16 --port 2000 --weight-path /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_8bit-Enflame qwen2 --quant gptq --temperature 0. --penalty 1.0
```

Run **AWQ models**,
```shell
python3 transform_safetensors.py --src /home/Llama3.1-8B-AWQ --dst /home/Llama3.1-8B-AWQ-Enflame/ --bits 4 --method awq --group 64 --nk True

target/release/candle-vllm --multi-process --dtype f16 --port 2000 --device-ids "0" --weight-path /home/Llama3.1-8B-AWQ llama3 --quant awq --temperature 0. --penalty 1.0
```

You may also run specific model using **Huggingface model-id**, e.g.,
```shell
target/release/candle-vllm --port 2000 --model-id meta-llama/Llama-2-7b-chat-hf llama
target/release/candle-vllm --port 2000 --model-id avoroshilov/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g --quant gptq --penalty 1.0 --temperature 0.
```

Run **Multi-process Multi-GPU**

```shell
target/release/candle-vllm --multi-process --port 2000 --device-ids "0,1" --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --temperature 0. --penalty 1.0
```

```shell
target/release/candle-vllm --multi-process --dtype bf16 --port 2000 --device-ids "0,1" --weight-path /home/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4/ llama3 --quant gptq --temperature 0. --penalty 1.0
```

Run **Multi-threaded Multi-GPU** (for debug purpose)
```shell
#simply remove the "--multi-process"
target/debug/candle-vllm --port 2000 --device-ids "0,1" --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --temperature 0. --penalty 1.0
```
**Note:** number of GPUs (`--device-ids`) used must be aligned to 2^n (e.g., 2, 4, or 8).

If you encountered problems under Multi-threaded Multi-GPU mode, you may:
```shell
export ECCL_P2P_DISABLE=1 # disable p2p cause this feature can cause illegal memory access in certain environments
```

Run **DeepSeek-R1 (671B/685B) on Lower GCU Memory Setups**, e.g., **single-node** with `8 x S60(48GB)`
```shell
python3 transform_safetensors.py --src /data/DeepSeek-R1-AWQ/ --dst /data/DeepSeek-R1-AWQ-Enflame/ --bits 4 --method awq --group 64 --nk True

RUST_LOG=warn cargo run --release --features gcu,eccl -- --log --multi-process --dtype bf16 --port 2000 --device-ids "0,1,2,3,4,5,6,7" --weight-path /data/DeepSeek-R1-AWQ-Enflame/ deep-seek --quant awq --temperature 0. --penalty 1.0 --num-experts-offload-per-rank 11
```
**Note:** This setup offloads 11 experts per rank (a total of 88 out of 256 experts) to the CPU (around 125GB additional host memory required). During inference, these offloaded experts are swapped back into GCU memory as needed. If you have even less GCU memory, consider increasing the `--num-experts-offload-per-rank` parameter (up to a maximum of 32 experts per rank in this case).

Run **DeepSeek-R1 (671B/685B) on multi-node**, e.g., (`8 x S60(48GB)` x 2 nodes)
```shell
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin -y #install mpi
sudo apt install clang libclang-dev
cargo build --release --features gcu,eccl,mpi #build with mpi feature
python3 transform_safetensors.py --src /data/DeepSeek-R1-AWQ/ --dst /data/DeepSeek-R1-AWQ-Enflame/ --bits 4 --method awq --group 64 --nk True #convert awq deepseek to Enflame format
#running multinode inference with mpi runner
sudo mpirun -np 16 -x RUST_LOG=info -hostfile ./hostfile --allow-run-as-root -bind-to none -map-by slot --mca plm_rsh_args "-p 22" --mca btl_tcp_if_include %NET_INTERFACE% target/release/candle-vllm --log --multi-process --dtype bf16 --port 2000 --device-ids "0,1,2,3,4,5,6,7" --weight-path /data/DeepSeek-R1-AWQ-Enflame/ deep-seek --quant awq --temperature 0. --penalty 1.0
```
**Note**: MPI Runner requires `identical` hardware and software configurations for all nodes, please ensure weights and candle-vllm binaries located in the identical folders in difference nodes. The the nodes need to be ssh (port 22 in this case) passwordless for each other (root user if `--allow-run-as-root`). `%NET_INTERFACE%` is the active network interface obtained through command 'ifconfig -a'. You may disable InfiniBand if it's not available in the nodes by insert env "-x ECCL_IB_DISABLE=1". Where, `hostfile` can be defined as:

Example (two nodes, each with 8 GCUs)
```
192.168.1.100 slots=8
192.168.1.101 slots=8
```

### Chat frontend (any frontend compatible with openai API, simple options available below):
#### Option 1: Chat with Chat.py (for simple tests)
Install API and chatbot dependencies (openai package is only used for local chat with candle-vllm)

```shell
python3 -m pip install openai rich click
```

__Option 2:__ Chat with mini chatbot

```shell
python3 examples/chat.py #plain text
```

```shell
python3 examples/chat.py --live #live update with Markdown, may cause flick
```
__Option 2:__ Chat with ChatUI

Install ChatUI and its dependencies:

```
cd candle-vllm-demo
apt install npm #install npm if needed
npm install n -g #update node js if needed
n stable #update node js if needed
npm i -g pnpm #install pnpm manager
pnpm install #install ChatUI dependencies
```

Launch the Chat UI:
```
pnpm run dev # run the ChatUI
```


## Quantized (GPTQ)
1) Transform GPTQ model (8bit) to Enflame (W8A16) format using `transform_safetensors.py`

2) Change `quant_method` in config.json to `w8a16`

3) Run the transformed quantized model

```shell
cargo run --release --features gcu -- --port 2000 --weight-path /home/Meta-Llama-3.1-8B-Instruct-GPTQ-Enflame/ llama3 --temperature 0. --penalty 1. --quant gptq
```

## Batched requests (Benchmark)

Refer to `examples/benchmark.py`

Install openai API package (python3 -m pip install openai) and run

```shell
python3 examples/benchmark.py --batch 16 --max_tokens 1024
```

``` python
async def benchmark():
    model = "mistral7b"
    max_tokens = 1024
    # 16 requests
    prompts = ["Explain how to best learn Rust.", 
               "Please talk about deep learning in 100 words.", 
               "Do you know the capital city of China? Talk the details of you known.", 
               "Who is the best female actor in the world? Explain why.",
               "How to dealing with depression?",
               "How to make money in short time?",
               "What is the future trend of large language model?",
               "The famous tech companies in the world.",
               "Explain how to best learn Rust.", 
               "Please talk about deep learning in 100 words.", 
               "Do you know the capital city of China? Talk the details of you known.", 
               "Who is the best female actor in the world? Explain why.",
               "How to dealing with depression?",
               "How to make money in short time?",
               "What is the future trend of large language model?",
               "The famous tech companies in the world."]
    
    # send 16 chat requests at the same time
    tasks: List[asyncio.Task] = []
    for i in range(len(prompts)):
        tasks.append(
            asyncio.create_task(
                chat_completion(model, max_tokens, prompts[i]))
        )

    # obtain the corresponding stream object for each request
    outputs: List[Stream[ChatCompletionChunk]] = await asyncio.gather(*tasks)

    # tasks for streaming chat responses
    tasks_stream: List[asyncio.Task] = []
    for i in range(len(outputs)):
        tasks_stream.append(
            asyncio.create_task(
                stream_response(i, outputs[i]))
        )

    # gathering the response texts
    outputs: List[(int, str)] = await asyncio.gather(*tasks_stream)

    # print the results, you may find chat completion statistics in the backend server (i.e., candle-vllm)
    for idx, output in outputs:
        print("\n\n Response {}: \n\n {}".format(idx, output))


asyncio.run(benchmark())
```

## In-situ quantization for GCU

Candle-vllm-gcu now supports in-situ quantization, allowing the transformation of default weights (F32/F16/BF16) into GGML format during model loading.

```
cargo run --release --features gcu -- --port 2000 --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --quant q8_0
```

Options for `quant` parameters on GCU: ["q8_0"]

**Please note**:

1) In-situ quantization is a new feature under experimental development, and `q8_0` is not an ideal solution for quantization. `q4_k` format is prefered.

2) Batched processing still requires further optimizations when operating in quantization mode.

## TODO
1. Optimization of generation speed.
2. Add supports for more quantization formats (`q8_0` and `gptq` supported).
3. Simultaneous chat serving for multiple users.
4. Support multimodal models.

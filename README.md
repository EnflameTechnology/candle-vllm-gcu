# Candle-VLLM-GCU
OpenAI-API compatible chat service for `GCU` platform based on `candle-gcu` and `candle-vllm (paged attention)`


## Getting started

### Install dependencies
#### Step 1: Install Rust & download code
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt install libssl-dev
sudo apt install pkg-config
git clone git@git.enflame.cn:era/candle-vllm-gcu.git
git submodule update --init --recursive
```

#### Step 2: Run candle-vllm service on GCU

Run `Unquantized` models
```shell
cd candle-vllm
cargo run --release --features gcu -- --port 2000 --dtype bf16 --weight-path /home/weights/Meta-Llama-3.1-8B-Instruct/ llama3 --temperature 0. --penalty 1.
```

Run `Quantized` models (transform weight to Enflame format using the given script, `dtype` is used for kv cache and attention)
```shell
cd candle-vllm
cargo run --release --features gcu -- --port 2000 --dtype bf16 --weight-path /home/weights/Meta-Llama-3.1-8B-Instruct-GPTQ-Enflame/ llama3 --quant gptq --temperature 0. --penalty 1.
```

Run **DeepSeek** MoE models
```shell
cargo run --release --features gcu -- --port 2000 --weight-path /home/DeepSeek-V2-Lite-Chat deep-seek --penalty 1.0 --temperature 0.
```

Run `Multi-threaded` `Multi-GCU` inference:

```shell
dpkg -i eccl_3.1xxx_amd64.deb  # Install ECCL in the environment where candle-vllm-gcu resides.
```

```shell
cargo run --release --features gcu,eccl -- --port 2000 --dtype bf16 --device-ids "0,1" --weight-path /home/weights/Meta-Llama-3.1-8B-Instruct/ llama3 --temperature 0. --penalty 1.0
```

For large batch processing, increase the `holding time` (e.g., 1000 ms, holding more requests as a batch) (you may also increase kv cache by setting `kvcache-mem-gpu`)
```shell
cargo run --release --features gcu,eccl -- --dtype bf16 --port 2000 --device-ids "0,1" --holding-time 1000 --weight-path /home/weights/QwQ-32B/ qwen2 --temperature 0.8 --penalty 1.0
```

Run `Multi-process` `Multi-GCU` inference:
```shell
cargo run --features gcu,eccl --dtype bf16 --port 2000 --device-ids "0,1" --multi-process --weight-path /home/weights/Meta-Llama-3.1-8B-Instruct-GPTQ-EnflameT llama3 --quant gptq --temperature 0. --penalty 1.0

cargo run --features gcu,eccl --dtype bf16 --port 2000 --device-ids "0,1" --multi-process --weight-path /home/weights/Meta-Llama-3.1-8B-Instruct llama3 --temperature 0. --penalty 1.0
```

Run `Multi-threaded` `Multi-GCU` inference for `quantized models`
```shell
cargo run --release --features gcu,eccl -- --dtype bf16 --port 2000 --device-ids "0,1" --weight-path /home/weights/DeepSeek-R1-Distill-Qwen-14B-GPTQ-Enflame/ qwen2 --quant gptq --temperature 0.8 --penalty 1.0 --top-k 32 --top-p 0.9
```

**Note:** 
1) `Multi-GCU` inference (with `Multi-threaded` or `Multi-process`) is not stable at the moment.
2) Only unquantized and quantized `GPTQ` models are supported under multi-gcu setting.

#### Step 3: 

__Option 1:__ Chat with Chat.py (recommended)

Install API and chatbot dependencies (openai package is only used for local chat with candle-vllm)
```shell
python3 -m pip install openai
python3 -m pip install rich
python3 -m pip install click
```

Chat with the mini chatbot
```shell
python3 examples/chat.py
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

## Demo chat video (on Scorpio S60, LLaMa3.1 8B, 8bit, ~30 tokens/s)
<img src="resources/LLaMa3.1-8B-S60-Quant.gif" width="85%" height="85%" >

## Status

Currently, candle-vllm-gcu supports chat serving for the following models on `S60`.

__`List of 1k decoding results:`__
| Model ID | Model Type | Supported | Speed (BF16, `batch size=1`)| Thoughput (BF16, `batch size=16`) | Thoughput (W8A16, `batch size=16`)
|--|--|--|--|--|--|
| #1 | **LLAMA** |✅|30 tks/s (7B), 27 tks/s (LLaMa3.1 8B)| 305 tks/s (LLaMa3.1 8B) | 375 tks/s (LLaMa3.1 8B) |
| #2 | **Mistral** |✅|29 tks/s (7B)|330 tks/s (7B)|TBD|
| #3 | **Phi (v1, v1.5, v2)** |✅|TBD|TBD|
| #4 | **Phi-3** |✅|38 tks/s (3.8B)|320 tks/s (BF16+F32, 7B)|TBD|
| #5 | **Yi** |✅|28 tks/s (6B)|305 tks/s (6B)|TBD|
| #6 | **StableLM** |✅|48 tks/s (3B)|425 tks/s (BF16, 3B)|TBD|
| #7 | BigCode/StarCode |TBD|TBD|TBD|
| #8 | ChatGLM |TBD|TBD|TBD|
| #9 | **QWen2** |✅|22 tks/s (14B, **tp==2**)|322 tks/s (14B, **tp==2, bs==32**)|
| #10 | **Google Gemma** |✅|51 tks/s (2B)| 577 tks/s (2B) |TBD|
| #11 | Blip-large (Multimodal) |TBD|TBD|TBD|
| #12 | Moondream-2 (Multimodal LLM) |TBD|TBD|TBD|
| #13 | **DeepSeek-V2** |✅|TBD|TBD|TBD|
| #13 | **QwQ-32B** |✅|10 tokens (**tp=2**)|186 tokens (**tp=2, bs=32**)|TBD|

## General Usage
`MODEL_TYPE` = ["llama", "llama3", "mistral", "phi2", "phi3", "qwen2", "gemma", "yi", "stable-lm", "deep-seek"]

`WEIGHT_FILE_PATH` = Corresponding weight path for the given model type

Parameters **before** `MODEL_TYPE`: used for candle-vllm general config (e.g, port, dtype, kvcache size)

Parameters **after** `MODEL_TYPE`: used for model config (e.g., sampling parameters, quant)  

```shell
cargo run --release --features gcu -- --port 2000 --weight-path <WEIGHT_FILE_PATH> <MODEL_TYPE>
```

Example: 
```shell
cargo run --release --features gcu -- --port 2000 --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --temperature 0.7
```

**or**

`MODEL_ID` = Huggingface model id

```shell
cargo run --release --features gcu -- --port 2000 --model-id <MODEL_ID> <MODEL_TYPE>
```

Example: 

You may supply penalty and temperature to the model to prevent potential repetitions, for example:

```shell
cargo run --release --features gcu -- --port 2000 --weight-path /home/mistral_7b/ mistral --penalty 1.1 --temperature 0.7
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

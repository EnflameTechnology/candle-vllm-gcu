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
```
cd candle-vllm
cargo run --release --features gcu -- --port 2000 --weight-path /home/llama2_7b/ llama --repeat-last-n 64
```

#### Step 3: Chat with ChatUI (recommended)

Install ChatUI and its dependencies:

```
cd candle-vllm-demo
apt install npm #install npm if needed
npm install n -g #update node js if needed
n stable #update node js if needed
npm i -g pnpm #install pnpm manager
pnpm install #install ChatUI dependencies
```

Launching the ChatUI:
```
pnpm run dev # run the ChatUI
```

## Demo chat video (on GCU, LLaMa2 7B, BF16)
<img src="resources/candle-vllm-gcu-demo.gif" width="85%" height="85%" >

## Status

Currently, candle-vllm-gcu supports chat serving for the following models on `S60`.

| Model ID | Model Type | Supported | Speed (BF16, `batch size=1`)| Thoughput (BF16, `batch size=16`)
|--|--|--|--|--|
| #1 | **LLAMA/LLAMA2/LLaMa3/LLaMa3.1** |✅|20 tks/s (7B), 18 tks/s (LLaMa3.1 8B)| 175 tks/s (LLaMa3.1 8B) |
| #2 | **Mistral** |✅|21 tks/s (7B)| 245 tks/s (7B) |
| #3 | **Phi (v1, v1.5, v2)** |✅|TBD|
| #4 | **Phi-3 （3.8B, 7B）** |✅|31 tks/s (3.8B)|
| #5 | **Yi** |✅|22 tks/s (6B)|
| #6 | **StableLM** |✅|35 tks/s (3B)|
| #7 | BigCode/StarCode |TBD|TBD|
| #8 | ChatGLM |TBD|TBD|
| #9 | **QWen2 (1.8B, 7B)** |✅|43 tks/s (1.8B)|
| #10 | **Google Gemma** |✅|43 tks/s (2B)|
| #11 | Blip-large (Multimodal) |TBD|TBD|
| #12 | Moondream-2 (Multimodal LLM) |TBD|TBD|

## General Usage
`MODEL_TYPE` = ["llama", "llama3", "mistral", "phi2", "phi3", "qwen2", "gemma", "yi", "stable-lm"]

`WEIGHT_FILE_PATH` = Corresponding weight path for the given model type

```
cargo run --release --features gcu -- --port 2000 --weight-path <WEIGHT_FILE_PATH> <MODEL_TYPE>
```

Example: 
```
cargo run --release --features gcu -- --port 2000 --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --penalty 1.1 --temperature 0.7
```

**or**

`MODEL_ID` = Huggingface model id

```
cargo run --release --features gcu -- --port 2000 --model-id <MODEL_ID> <MODEL_TYPE>
```

Example: 

You may supply penalty and temperature to the model to prevent potential repetitions, for example:

```
cargo run --release -- --port 2000 --weight-path /home/mistral_7b/ mistral --penalty 1.1 --temperature 0.7
```

## Batched requests

Refer to `examples/benchmark.py`

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
## TODO
1. Optimization of generation speed.
2. Add quantization support.
3. Simultaneous chat serving for multiple users.
4. Support multimodal models.

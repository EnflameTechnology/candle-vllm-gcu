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

Currently, candle-vllm-gcu supports chat serving for the following models.

| Model ID | Model Type | Supported | Speed (S60, BF16)
|--|--|--|--|
| #1 | **LLAMA/LLAMA2/LLaMa3** |✅|20 tks/s (7B)|
| #2 | Mistral |TBD|TBD|
| #3 | Phi (v1, v1.5, v2) |TBD|TBD|
| #4 | **Phi-3 （3.8B, 7B）** |✅|29 tks/s (3.8B)|
| #5 | Yi |TBD|TBD|
| #6 | StableLM |TBD|TBD|
| #7 | BigCode/StarCode |TBD|TBD|
| #8 | ChatGLM |TBD|TBD|
| #9 | **QWen2 (1.8B, 7B)** |✅|43 tks/s (1.8B)|
| #10 | **Google Gemma** |✅|43 tks/s (2B)|
| #11 | Blip-large (Multimodal) |TBD|TBD|
| #12 | Moondream-2 (Multimodal LLM) |TBD|TBD|

## Usage
`MODEL_TYPE` = ["llama", "phi3", "qwen2", "gemma"]

`WEIGHT_FILE_PATH` = Corresponding weight path for the given model type

```
cargo run --release --features gcu -- --port 2000 --weight-path <WEIGHT_FILE_PATH> <MODEL_TYPE> --repeat-last-n 64
```

or

`MODEL_ID` = Huggingface model id

```
cargo run --release --features gcu -- --port 2000 --model-id <MODEL_ID> <MODEL_TYPE> --repeat-last-n 64
```

## Support
TODO

## Roadmap
TODO

## Contributing
TODO

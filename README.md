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
cargo run --release --features gcu -- --port 2000 --weight-path /home/llama2_7b/ llama7b --repeat-last-n 64
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

## Usage
TODO

## Support
TODO

## Roadmap
TODO

## Contributing
TODO

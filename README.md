# Candle-VLLM-GCU
OpenAI-API compatible chat service for `GCU` platform based on `candle-gcu` and `candle-vllm (paged attention)`


## Getting started

### Install dependencies
#### Step 1: Install Rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt install libssl-dev
sudo apt install pkg-config
```

#### Step 2: Run candle-vllm service on GCU
```
cd candle-vllm
cargo run --release --features gcu,scorpio -- --port 2000 --weight-path /home/llama2_7b/ llama7b --repeat-last-n 64
```

#### Step 3: Chat with ChatUI (recommended)

```
cd ChattierGPT-UI
pip install -r requirements.txt
python -m streamlit run src/main.py
```
## Demo chat video (on GPU)
<img src="https://github.com/guoqingbao/candle-vllm/blob/master/res/candle-vllm-demo.gif" width="95%" height="95%" >

## TODO
Porting candle-vllm to GCU
Demo chat video (on GCU)

## Usage
TODO

## Support
TODO

## Roadmap
TODO

## Contributing
TODO

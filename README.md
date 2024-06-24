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
cargo run --release --features gcu -- --port 2000 --weight-path /home/llama2_7b/ llama7b --repeat-last-n 64
```

#### Step 3: Chat with ChatUI (recommended)

```
cd ChattierGPT-UI
pip install -r requirements.txt
python -m streamlit run src/main.py
```

You may fix bugs for streamlit if error prompt 
`OSError: [Errno 28] inotify watch limit reached`

Uncommets for files dist-packages/streamlit/web/bootstrap.py:

    #_install_config_watchers(flag_options)
    
    #_install_pages_watcher(main_script_path)

## Demo chat video (on GCU, 2x video playback speed)
<img src="resources/candle-vllm-gcu-demo.gif" width="95%" height="95%" >

## Usage
TODO

## Support
TODO

## Roadmap
TODO

## Contributing
TODO

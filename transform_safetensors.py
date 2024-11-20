#python3 transform_safetensors.py --src /path/to/source.safetensors --dst /path/to/destination.safetensors
#src: /home/weights/Meta-Llama-3.1-8B-Instruct-GPTQ/gptq_model-8bit-128g.safetensors
#dst: /home/weights/Meta-Llama-3.1-8B-Instruct-GPTQ-Enflame/model.safetensors

import torch
from safetensors.torch import load_file, save_file
import argparse
import os
import shutil

def transform_weight(src_tensor, bits=8):
    """
    Transform 8-bit GPTQ weights from int32 to int8 representation.

    Args:
        src_tensor (torch.Tensor): Source tensor of shape [K/(32/bits), N].
        bits (int): Number of bits to pack. Default is 8.

    Returns:
        torch.Tensor: Transformed int8 tensor.
    """
    pack_num = int(32 / bits)
    uint8_tensor = torch.zeros(src_tensor.shape[0] * pack_num, src_tensor.shape[1], dtype=torch.uint8)
    for i in range(src_tensor.shape[0]):
        cur_weight = src_tensor[i, :]
        for j in range(pack_num):
            shift = j * bits
            unpacked_weight = torch.bitwise_right_shift(cur_weight, shift)
            uint8_tensor[i * pack_num + j, :] = torch.bitwise_and(unpacked_weight, 2**bits - 1)
    int8_tensor = uint8_tensor.to(torch.int32).sub(128).to(torch.int8)
    return int8_tensor

def transform_file(src_file, dst_file):
    """
    Transform and save safetensors file.

    Args:
        src_file (str): Path to the source safetensors file.
        dst_file (str): Path to the target safetensors file.
    """
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Source file not found: {src_file}")
    
    print(f"Loading source file: {src_file}")
    src_dict = load_file(src_file)
    tgt_dict = {}

    for key, tensor in src_dict.items():
        if key.endswith(".g_idx") or key.endswith(".qzeros"):
            continue
        if key.endswith(".qweight"):
            print(f"Transforming tensor: {key}")
            tgt_dict[key] = transform_weight(tensor)
        else:
            tgt_dict[key] = tensor

    print(f"Saving transformed file: {dst_file}")
    save_file(tgt_dict, dst_file)
    print("Transformation complete.")

def main():
    """
    Main function to handle command-line arguments for transforming safetensors files.
    """
    parser = argparse.ArgumentParser(description="Transform GPTQ safetensors weights to enflame format.")
    parser.add_argument(
        "--src", 
        type=str, 
        required=True, 
        help="Path to the source safetensors file."
    )
    parser.add_argument(
        "--dst", 
        type=str, 
        required=True, 
        help="Path to save the transformed safetensors file."
    )

    args = parser.parse_args()

    try:
        dst_directory = os.path.dirname(os.path.abspath(args.dst))
        if not os.path.exists(dst_directory):
            os.makedirs(dst_directory)
        transform_file(args.src, args.dst)
        shutil.copy2(os.path.dirname(os.path.abspath(args.src)) + "/tokenizer.json", dst_directory)
        shutil.copy2(os.path.dirname(os.path.abspath(args.src)) + "/config.json", dst_directory)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

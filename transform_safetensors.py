# example (8-bit gptq to Enflame w8a16): 
# python3 transform_safetensors.py --src /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit --dst /data/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit-Enflame --bits 8 --method gptq --group 128 --nk true


import torch
from safetensors.torch import load_file, save_file
import argparse
import os
import shutil

def awq_rearrange_uint4_int32_uint8(
                                qweight,
                                qzeros,
                                scales,
                                bits = 4,
                                pack_factor = 32 // 4,
                                rearrange_group=128):
    assert rearrange_group % 2 == 0, "rearrange group must be multiple of 2."
    qweight_shape = qweight.shape
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    shifts = torch.arange(0, 32, bits, device=qweight.device)

    # unpacking columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    iweights = iweights.view(iweights.shape[0], -1)
    assert qweight_shape[1] * pack_factor == iweights.shape[1], \
            f"unpacked qweight shape error: {qweight_shape} and {iweights.shape}"
    # unpacking columnwise
    izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    izeros = izeros.view(izeros.shape[0], -1)
    assert qweight_shape[1] * pack_factor == izeros.shape[1], \
            f"unpacked qzeros shape error: {qweight_shape} and {izeros.shape}"
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device=qweight.device
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)
    iweights = iweights[:, reverse_order_tensor]
    izeros = izeros[:, reverse_order_tensor]
    # overflow checks
    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    izeros = izeros * scales

    # weight rearrange
    if iweights.shape[0] % rearrange_group != 0:
        padding = torch.zeros([iweights.shape[0] % rearrange_group, iweights.shape[1]],
                                dtype=iweights.dtype,
                                device=iweights.device)
        iweights = torch.concat([iweights, padding], dim=0)
    rweight_shape = (int(iweights.shape[0] / 2), iweights.shape[1])
    rweight = torch.zeros(rweight_shape, dtype=torch.uint8).to(qweight.device)
    half_group = int(rearrange_group / 2)
    try:
        shifts = torch.arange(
            0, iweights.shape[0], device=qweight.device).reshape(int(iweights.shape[0] / half_group), -1)
        rweight |= torch.bitwise_left_shift(
            iweights[shifts[::2].reshape(-1)], 0)
        rweight |= torch.bitwise_left_shift(
            iweights[shifts[1::2].reshape(-1)], 4)
    except Exception as e:
        raise RuntimeError(f'weight rearrange error: {e}')

    return rweight, izeros

def gptq_rearrange_uint4_int32_uint8(
                                qweight,
                                qzeros,
                                scales,
                                bits = 4,
                                group_size = 128,
                                rearrange_group=128):
    assert bits in [4], "only 4 bit gptq quant is supported."
    wf = torch.tensor(list(range(0, 32, bits)),
                        dtype=torch.int32).unsqueeze(0)
    zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
            wf.unsqueeze(0),
        ).to(torch.int8)
    zeros = zeros + 1
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)
    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
    scales = scales.reshape(-1, 1, scales.shape[-1])

    weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
            wf.unsqueeze(-1),
        ).to(torch.int8)
    weight = torch.bitwise_and(weight, (2**bits) - 1)
    weight = weight.reshape(-1, weight.shape[2])

    zeros = zeros * scales
    zeros = zeros.reshape(-1, zeros.shape[2])

    # weight rearrange
    if weight.shape[0] % rearrange_group != 0:
        padding = torch.zeros([weight.shape[0] % rearrange_group, weight.shape[1]],
                                dtype=weight.dtype,
                                device=weight.device)
        weight = torch.concat([weight, padding], dim=0)
    rweight_shape = (int(weight.shape[0] / 2), weight.shape[1])
    rweight = torch.zeros(rweight_shape, dtype=torch.uint8).to(qweight.device)
    half_group = int(rearrange_group / 2)
    try:
        shifts = torch.arange(
            0, weight.shape[0], device=qweight.device).reshape(int(weight.shape[0] / half_group), -1)
        rweight |= torch.bitwise_left_shift(
            weight[shifts[::2].reshape(-1)], 0)
        rweight |= torch.bitwise_left_shift(
            weight[shifts[1::2].reshape(-1)], 4)
    except Exception as e:
        raise RuntimeError(f'weight rearrange error: {e}')

    return rweight, zeros

def transform_gptq_weight_8bits(src_tensor, nk):
    """
    Transform 8-bit GPTQ weights from int32 to int8 representation.

    Args:
        src_tensor (torch.Tensor): Source tensor of shape [K/(32/bits), N].
        bits (int): Number of bits to pack. Default is 8.

    Returns:
        torch.Tensor: Transformed int8 tensor.
    """
    bits = 8
    pack_num = int(32 / bits)
    uint8_tensor = torch.zeros(src_tensor.shape[0] * pack_num, src_tensor.shape[1], dtype=torch.uint8)
    for i in range(src_tensor.shape[0]):
        cur_weight = src_tensor[i, :]
        for j in range(pack_num):
            shift = j * bits
            unpacked_weight = torch.bitwise_right_shift(cur_weight, shift)
            uint8_tensor[i * pack_num + j, :] = torch.bitwise_and(unpacked_weight, 2**bits - 1)
    int8_tensor = uint8_tensor.to(torch.int32).sub(128).to(torch.int8)
    if nk:
        int8_tensor = int8_tensor.t().contiguous()
    return int8_tensor

def transform_file(src_folder, dst_folder, bits, method, group_size, nk):
    """
    Transform and save safetensors file.

    Args:
        src_folder (str): Path to the source safetensors file.
        dst_folder (str): Path to the target safetensors file.
    """
    if not os.path.exists(src_folder):
        raise FileNotFoundError(f"Source file not found: {src_folder}")
    
    print(f"Loading source file: {src_folder}")
    tgt_dict = {}

    for file in os.listdir(src_folder):
        if not file.endswith(".safetensors") or file.find("model") < 0:
            continue
        f = os.path.join(src_folder, file)
        dst_f = os.path.join(dst_folder, file)

        src_dict = load_file(f)
        tgt_dict = {}

        for key, tensor in src_dict.items():
            if bits == 8:
                if key.endswith(".g_idx") or key.endswith(".qzeros"):
                    continue
                if key.endswith(".qweight"):
                    print(f"Transforming tensor: {key}")
                    tgt_dict[key] = transform_gptq_weight_8bits(tensor, nk)
                else:
                    tgt_dict[key] = tensor
            else:
                if key.endswith(".g_idx") or key.endswith(".qzeros") or key.endswith(".scales"):
                    continue
                if key.endswith(".qweight"):
                    print(f"Transforming tensor: {key}")
                    key_prefix = key.replace(".qweight", "")
                    key_qzeros = key_prefix + ".qzeros"
                    qzeros = src_dict[key_qzeros]
                    key_scales = key_prefix + ".scales"
                    scales = src_dict[key_scales]
                    key_g_idx = key_prefix + ".g_idx"
                    g_idx = src_dict[key_g_idx]
                    if method == "awq":
                        qweight, qzeros = awq_rearrange_uint4_int32_uint8(tensor, qzeros, scales)
                    else:
                        qweight, qzeros = gptq_rearrange_uint4_int32_uint8(tensor, qzeros, scales)
                        assert torch.all(g_idx[1:] - g_idx[:-1] >= 0).item() and \
                            g_idx[-1] - g_idx[0] + 1 == g_idx.shape[0] / group_size, \
                            "gcu only support g_idx is continuous."
                        g_idx = g_idx.reshape([-1, group_size])[:, 0]
                        qzeros = qzeros[g_idx.cpu()].to(qweight.device)
                        scales = scales[g_idx.cpu()].to(qweight.device)

                    tgt_dict[key] = qweight
                    tgt_dict[key_qzeros] = qzeros
                    tgt_dict[key_scales] = scales

        print(f"Saving transformed file: {dst_f}")
        save_file(tgt_dict, dst_f)
    print("Transformation complete.")

import json
def load_json(json_path, fn):
    json_fn = os.path.join(json_path, fn)
    with open(json_fn, "r", encoding="utf-8") as f_json:
        json_dict = json.load(f_json)
    return json_dict

def main():
    """
    Main function to handle command-line arguments for transforming safetensors files.
    """
    parser = argparse.ArgumentParser(description="Transform AWQ and GPTQ weights to Enflame format (w8a16, w4a16(awq, gptq)).")
    parser.add_argument(
        "--src", 
        type=str, 
        required=False, 
        help="Path to the source safetensors single file."
    )
    parser.add_argument(
        "--dst", 
        type=str, 
        required=True, 
        help="Path to save the transformed safetensors file."
    )

    parser.add_argument(
        "--bits", 
        type=int, 
        required=True, 
        default=8,
        help="Weight bits (default 8 bit for w8a16)."
    )

    parser.add_argument(
        "--method", 
        type=str, 
        required=True, 
        default="gptq",
        help="Weight format (gptq or awq)."
    )

    parser.add_argument(
        "--group", 
        type=int, 
        required=False, 
        default=128,
        help="Group size."
    )

    parser.add_argument(
        "--nk", 
        type=bool, 
        required=False, 
        default=True,
        help="Output nk format (default nk format)."
    )

    args = parser.parse_args()
    assert args.src != "" and os.path.exists(args.src), "Must provide src folder (or src folder not found)!"
    assert args.dst != "" and not os.path.exists(args.dst), "Must provide dst folder (or dst folder must be empty)!"

    assert args.bits == 8 or args.bits == 4, "only 4-bit and 8-bit models are supported!"
    assert args.group == 128, "only group size of 128 is supported!"
    assert args.method == "awq" or args.method == "gptq", "only awq and gptq quantization methods are supported!"
    assert args.bits == 8 and args.method == "gptq", "8-bit gptq only, awq-8bit not supported!"

    try:
        src_directory = args.src
        if not os.path.exists(args.dst):
            os.makedirs(args.dst)
        transform_file(args.src, args.dst, args.bits, args.method, args.group, args.nk)
        shutil.copy2(src_directory + "/config.json", args.dst)
        shutil.copy2(src_directory + "/tokenizer.json", args.dst)
        shutil.copy2(src_directory + "/tokenizer_config.json", args.dst)
        if os.path.exists(src_directory + "/model.safetensors.index.json"):
            shutil.copy2(src_directory + "/model.safetensors.index.json", args.dst)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

"""Convert Qwen3.5 VLM checkpoint to standard HF text-only format.

VLM checkpoint uses:
  - model.language_model.layers.* prefix (instead of model.layers.*)
  - experts.gate_up_proj (fused 3D) instead of experts.{i}.gate_proj + up_proj (per-expert 2D)
  - experts.down_proj (3D) instead of experts.{i}.down_proj (per-expert 2D)
  - model.visual.* and mtp.* keys (not needed for text-only)

Usage:
  python convert_qwen3_5_vlm_to_hf.py \
      --src /dataset_rc_b1/models/Qwen3.5-35B-A3B \
      --dst /dataset_rc_b1/models/Qwen3.5-35B-A3B-text
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def convert_keys(
    state_dict: dict[str, torch.Tensor],
    keep_visual: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert VLM state dict to standard HF text-only format.

    Args:
        keep_visual: If True, keep model.visual.* weights (for future
            multimodal RLVR where visual encoder can be frozen).
    """
    out = {}
    for key, tensor in state_dict.items():
        # Skip MTP (multi-token prediction) weights — not needed for training
        if key.startswith("mtp."):
            continue
        # Skip visual encoder unless --keep-visual
        if key.startswith("model.visual.") and not keep_visual:
            continue

        # Remove language_model prefix
        new_key = key.replace("model.language_model.", "model.")

        # Handle fused expert weights: gate_up_proj → per-expert gate_proj + up_proj
        if new_key.endswith(".mlp.experts.gate_up_proj"):
            prefix = new_key.rsplit(".gate_up_proj", 1)[0]
            # tensor shape: [num_experts, gate_up_dim, hidden_size]
            num_experts = tensor.shape[0]
            half_dim = tensor.shape[1] // 2
            gate = tensor[:, :half_dim, :]  # [E, I, H]
            up = tensor[:, half_dim:, :]    # [E, I, H]
            for i in range(num_experts):
                out[f"{prefix}.{i}.gate_proj.weight"] = gate[i]
                out[f"{prefix}.{i}.up_proj.weight"] = up[i]
            continue

        # Handle fused expert down_proj: 3D → per-expert 2D
        if new_key.endswith(".mlp.experts.down_proj"):
            prefix = new_key.rsplit(".down_proj", 1)[0]
            num_experts = tensor.shape[0]
            for i in range(num_experts):
                out[f"{prefix}.{i}.down_proj.weight"] = tensor[i]
            continue

        out[new_key] = tensor

    return out


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3.5 VLM to HF text-only")
    parser.add_argument("--src", required=True, help="Source VLM checkpoint dir")
    parser.add_argument("--dst", required=True, help="Destination HF text-only dir")
    parser.add_argument("--keep-visual", action="store_true",
                        help="Keep visual encoder weights (for future multimodal RLVR)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy non-safetensor files (config, tokenizer, etc.)
    for f in src.iterdir():
        if f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            if f.is_file():
                shutil.copy2(f, dst / f.name)
                print(f"Copied {f.name}")

    # Convert safetensor shards
    shard_files = sorted(src.glob("*.safetensors"))
    print(f"\nConverting {len(shard_files)} shards...")

    weight_map = {}
    for shard_file in tqdm(shard_files, desc="Converting"):
        sd = load_file(str(shard_file))
        converted = convert_keys(sd, keep_visual=args.keep_visual)

        if not converted:
            print(f"  Skipping {shard_file.name} (all visual/mtp keys)")
            continue

        out_name = shard_file.name
        save_file(converted, str(dst / out_name))

        for key in converted:
            weight_map[key] = out_name

    # Write new index
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size()
                                        for sd_file in (dst).glob("*.safetensors")
                                        for t in load_file(str(sd_file)).values())},
        "weight_map": weight_map,
    }
    # Simpler: just compute from weight_map
    index_path = dst / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

    print(f"\nDone! Converted {len(weight_map)} keys to {dst}")
    print(f"Dropped visual/mtp keys. Text-only checkpoint ready.")


if __name__ == "__main__":
    main()

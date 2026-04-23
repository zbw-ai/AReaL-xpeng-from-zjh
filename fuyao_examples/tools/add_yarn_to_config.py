"""Add YaRN RoPE scaling to a HuggingFace model checkpoint config.

POLARIS Stage-2/3 uses YaRN factor 1.5 to extend Qwen3's context.
This script modifies config.json in-place (with backup).

Usage:
    python add_yarn_to_config.py /path/to/checkpoint --factor 1.5
    python add_yarn_to_config.py /path/to/checkpoint --revert  # restore backup
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir", help="Path to checkpoint directory containing config.json")
    p.add_argument("--factor", type=float, default=1.5, help="YaRN scaling factor (POLARIS uses 1.5)")
    p.add_argument("--type", default="yarn", choices=["yarn", "linear", "dynamic"])
    p.add_argument("--revert", action="store_true", help="Restore config.json.bak")
    args = p.parse_args()

    ckpt = Path(args.ckpt_dir)
    config_path = ckpt / "config.json"
    backup_path = ckpt / "config.json.bak"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {ckpt}")

    if args.revert:
        if not backup_path.exists():
            raise FileNotFoundError(f"No backup found at {backup_path}")
        shutil.copy(backup_path, config_path)
        print(f"Reverted config.json from backup")
        return

    # Backup if not exists
    if not backup_path.exists():
        shutil.copy(config_path, backup_path)
        print(f"Backup saved to {backup_path}")

    # Load and modify
    with open(config_path) as f:
        cfg = json.load(f)

    original_max_pos = cfg.get("max_position_embeddings", 32768)
    new_max_pos = int(original_max_pos * args.factor)

    cfg["rope_scaling"] = {
        "type": args.type,
        "factor": args.factor,
        "original_max_position_embeddings": original_max_pos,
    }
    cfg["max_position_embeddings"] = new_max_pos

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Updated {config_path}")
    print(f"  rope_scaling: type={args.type}, factor={args.factor}")
    print(f"  max_position_embeddings: {original_max_pos} → {new_max_pos}")
    print(f"  effective context: {new_max_pos} tokens")


if __name__ == "__main__":
    main()

"""Math SFT training entry point.

Usage:
    # Single node (8 GPU)
    bash fuyao_examples/fuyao_areal_run.sh \
        --run-type math_sft \
        --config fuyao_examples/math/qwen3_8b_sft.yaml

    # Or directly:
    python fuyao_examples/math/train_math_sft.py \
        --config fuyao_examples/math/qwen3_8b_sft.yaml
"""

import sys

from fuyao_examples.math.dataset_sft import get_math_sft_dataset
from fuyao_examples.tracking_patch import apply_tracking_patch

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, SFTConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_math_sft_dataset(
        path=config.train_dataset.path,
        split="train",
        tokenizer=tokenizer,
        max_length=config.train_dataset.max_length,
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_math_sft_dataset(
            path=config.valid_dataset.path,
            split="valid",
            tokenizer=tokenizer,
            max_length=getattr(config.valid_dataset, "max_length", None),
        )

    # Apply DeepInsight metric mapping before training
    apply_tracking_patch()

    with SFTTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])

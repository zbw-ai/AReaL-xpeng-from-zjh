"""Math RLVR training entry point for dapo_math_17k dataset.

Usage:
    python fuyao_examples/math/train_math_rlvr.py \
        --config fuyao_examples/math/qwen3_4b_rlvr.yaml
"""

import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer
from fuyao_examples.dataset.dapo_math import get_dapo_math_rl_dataset
from fuyao_examples.tracking_patch import apply_tracking_patch


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_dapo_math_rl_dataset(
        path=config.train_dataset.path,
        split="train",
        tokenizer=tokenizer,
    )
    valid_dataset = get_dapo_math_rl_dataset(
        path=config.valid_dataset.path,
        split="valid",
        tokenizer=tokenizer,
    )

    # Apply DeepInsight metric mapping before training
    apply_tracking_patch()

    workflow_kwargs = dict(
        reward_fn="fuyao_examples.reward.math_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])

"""Code DAPO Agentic RL training entry point.

Usage:
    python fuyao_examples/code_dapo/train_code_dapo.py \
        --config fuyao_examples/code_dapo/code_dapo_qwen3_4b.yaml
"""

import os
import sys

from datasets import load_dataset

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer

from fuyao_examples.configs import AgenticConfig


def load_dapo_math_dataset(path: str, split: str = "train"):
    """Load dapo_math_17k dataset for code execution RL."""
    ds = load_dataset("parquet", data_dir=path, split="train")

    # Keep prompt and solution for CodeExecWorkflow
    def process(sample):
        return {
            "prompt": sample["prompt"],
            "solution": sample["solution"],
        }

    ds = ds.map(process)
    cols_to_remove = [c for c in ds.column_names if c not in ("prompt", "solution")]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)
    return ds


def main(args):
    config, _ = load_expr_config(args, AgenticConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = load_dapo_math_dataset(config.train_dataset.path)
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = load_dapo_math_dataset(config.valid_dataset.path)

    # Resolve execd endpoint from env if not in config
    execd_endpoint = config.execd_endpoint
    if not execd_endpoint:
        execd_endpoint = os.environ.get("EXECD_ENDPOINT", "")

    sandbox_type = config.sandbox_type
    if execd_endpoint and sandbox_type == "local":
        sandbox_type = "execd"

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        sandbox_type=sandbox_type,
        execd_endpoint=execd_endpoint,
        code_timeout=config.code_timeout,
        max_turns=config.max_turns,
        max_tool_uses=config.max_tool_uses,
        max_total_tokens=config.max_total_tokens,
        system_prompt=config.system_prompt,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(config, train_dataset, valid_dataset) as trainer:
        trainer.train(
            workflow="fuyao_examples.code_dapo.code_exec_workflow.CodeExecWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="fuyao_examples.code_dapo.code_exec_workflow.CodeExecWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])

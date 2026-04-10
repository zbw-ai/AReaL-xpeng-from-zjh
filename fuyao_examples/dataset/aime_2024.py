"""AIME 2024 dataset loader for evaluation.

Parquet schema:
    - data_source (str): "math_rule"
    - prompt (list[dict]): chat messages, e.g. [{"content": "...", "role": "user"}]
    - ability (str): "math"
    - reward_model (dict): {"ground_truth": "540", "style": "rule"}
    - extra_info (dict): {"answer": "540", "index": 2, "question": "...", "split": "test"}
"""

from datasets import Dataset, load_dataset

from areal.utils import logging

logger = logging.getLogger("AIME2024Dataset")


def get_aime_2024_rl_dataset(
    path: str,
    split: str = "test",
    tokenizer=None,
    max_length: int | None = None,
    **kwargs,
) -> Dataset:
    """Load AIME 2024 dataset (30 problems) for RL evaluation.

    Args:
        path: Path to parquet file.
        split: Ignored (single split). Kept for API compatibility.
        tokenizer: Unused. Kept for API compatibility.
        max_length: Unused. Kept for API compatibility.

    Returns:
        HuggingFace Dataset with columns: messages, answer
    """
    dataset = load_dataset("parquet", data_files=path, split="train")
    logger.info(f"Loaded {len(dataset)} samples from {path}")

    def process(sample):
        messages = sample["prompt"]
        answer = sample["reward_model"]["ground_truth"]
        return {"messages": messages, "answer": answer}

    dataset = dataset.map(process)
    cols_to_remove = [
        c for c in dataset.column_names if c not in ("messages", "answer")
    ]
    dataset = dataset.remove_columns(cols_to_remove)

    logger.info(f"AIME 2024 eval dataset: {len(dataset)} problems")
    return dataset

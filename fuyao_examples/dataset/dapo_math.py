"""DAPO Math 17k dataset loader for AReaL RL training.

Dataset format (Parquet):
    - prompt (str): math problem text
    - solution (str): full solution, answer extractable via \\boxed{}
"""

import re
from pathlib import Path

from datasets import load_dataset

from areal.utils import logging

logger = logging.getLogger("DapoMathDataset")
DEFAULT_VALID_SPLIT_RATIO = 0.05
DEFAULT_SPLIT_SEED = 42


def _extract_boxed_answer(solution: str) -> str:
    """Extract the answer from \\boxed{...} in solution text."""
    # Find the last \\boxed{...} in the solution
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if matches:
        return matches[-1].strip()
    # Fallback: return the full solution stripped
    return solution.strip()


def get_dapo_math_rl_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    **kwargs,
):
    """Load DAPO Math 17k dataset for RL training.

    Args:
        path: Path to directory containing parquet files.
        split: Dataset split (only "train" available for dapo_math_17k).
        tokenizer: Tokenizer for length filtering.
        max_length: Max prompt token length for filtering.

    Returns:
        HuggingFace Dataset with columns: messages, answer
    """
    dataset = _load_split_dataset(path=path, split=split or "train")
    logger.info(f"Loaded {len(dataset)} samples from {path} for split={split}")

    def process(sample):
        prompt_text = sample["prompt"]
        solution = sample["solution"]
        answer = _extract_boxed_answer(solution)

        messages = [
            {
                "role": "user",
                "content": prompt_text
                + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
            }
        ]
        return {"messages": messages, "answer": answer}

    dataset = dataset.map(process)
    # Remove original columns, keep messages and answer
    cols_to_remove = [c for c in dataset.column_names if c not in ("messages", "answer")]
    dataset = dataset.remove_columns(cols_to_remove)

    if max_length is not None and tokenizer is not None:

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        before = len(dataset)
        dataset = dataset.filter(filter_length)
        logger.info(f"Filtered {before} → {len(dataset)} samples (max_length={max_length})")

    return dataset


def _load_split_dataset(path: str, split: str):
    dataset_path = Path(path)
    explicit_files = _find_split_files(dataset_path, split)
    if explicit_files is not None:
        logger.info(f"Using explicit parquet files for split={split}: {explicit_files}")
        return load_dataset("parquet", data_files=explicit_files, split="train")

    full_dataset = load_dataset("parquet", data_dir=path, split="train")
    if split == "train":
        train_dataset, _ = _split_train_valid(full_dataset)
        return train_dataset
    if split in {"test", "valid", "validation"}:
        _, valid_dataset = _split_train_valid(full_dataset)
        return valid_dataset
    raise ValueError(f"Unsupported split for dapo_math dataset: {split}")


def _find_split_files(dataset_path: Path, split: str) -> str | list[str] | None:
    if dataset_path.is_file():
        return str(dataset_path)

    split_aliases = {
        "train": ("train",),
        "test": ("test", "valid", "validation", "eval"),
        "valid": ("valid", "validation", "eval"),
        "validation": ("validation", "valid", "eval"),
    }
    aliases = split_aliases.get(split, (split,))

    for alias in aliases:
        split_dir = dataset_path / alias
        if split_dir.is_dir():
            files = sorted(str(p) for p in split_dir.glob("*.parquet"))
            if files:
                return files

    files = []
    for alias in aliases:
        files.extend(sorted(str(p) for p in dataset_path.glob(f"*{alias}*.parquet")))
    return files or None


def _split_train_valid(dataset):
    if len(dataset) < 2:
        return dataset, dataset
    split_dataset = dataset.train_test_split(
        test_size=DEFAULT_VALID_SPLIT_RATIO,
        seed=DEFAULT_SPLIT_SEED,
        shuffle=True,
    )
    return split_dataset["train"], split_dataset["test"]

"""Math SFT dataset loader for parquet files with prompt/solution columns.

Dataset format (Parquet):
    - prompt (str): math problem text
    - solution (str): full solution text (answer extractable via \\boxed{})

Returns HuggingFace Dataset with columns:
    - input_ids: tokenized [prompt + solution + eos]
    - loss_mask: 0 for prompt tokens, 1 for solution tokens
"""

from pathlib import Path

from datasets import load_dataset

from areal.utils import logging

logger = logging.getLogger("MathSFTDataset")


def get_math_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
):
    """Load math parquet data for SFT training.

    Args:
        path: Path to parquet file or directory containing parquet files.
        split: "train" or "valid"/"test".
        tokenizer: HuggingFace tokenizer.
        max_length: Max sequence token length for filtering.
        system_prompt: System instruction prepended to each problem.
    """
    dataset = _load_parquet(path, split)
    logger.info(f"Loaded {len(dataset)} samples from {path} (split={split})")

    def process(sample):
        prompt_text = sample["prompt"]
        solution_text = sample["solution"]

        # Build chat messages, then tokenize
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": solution_text},
        ]

        # Use chat template if available, else simple concat
        if hasattr(tokenizer, "apply_chat_template"):
            # Full sequence: system + user + assistant
            full_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            # Prompt-only: system + user (for loss masking)
            prompt_messages = messages[:2]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            full_text = f"{system_prompt}\n\n{prompt_text}\n\n{solution_text}{tokenizer.eos_token}"
            prompt_only = f"{system_prompt}\n\n{prompt_text}\n\n"
            full_ids = tokenizer.encode(full_text)
            prompt_ids = tokenizer.encode(prompt_only)

        # Loss mask: 0 for prompt tokens, 1 for response tokens
        prompt_len = len(prompt_ids)
        loss_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)

        return {"input_ids": full_ids, "loss_mask": loss_mask}

    dataset = dataset.map(process)

    # Remove original columns, keep only input_ids and loss_mask
    cols_to_remove = [
        c for c in dataset.column_names if c not in ("input_ids", "loss_mask")
    ]
    dataset = dataset.remove_columns(cols_to_remove)

    if max_length is not None:
        before = len(dataset)
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
        logger.info(
            f"Filtered {before} -> {len(dataset)} samples (max_length={max_length})"
        )

    logger.info(f"Final dataset size: {len(dataset)} samples")
    return dataset


def _load_parquet(path: str, split: str):
    """Load parquet dataset, auto-splitting if no explicit split files."""
    p = Path(path)

    # Single file
    if p.is_file():
        full = load_dataset("parquet", data_files=str(p), split="train")
        return _maybe_split(full, split)

    # Directory: look for split-specific files
    for alias in _split_aliases(split):
        split_dir = p / alias
        if split_dir.is_dir():
            files = sorted(str(f) for f in split_dir.glob("*.parquet"))
            if files:
                return load_dataset("parquet", data_files=files, split="train")
        # Or files matching pattern
        files = sorted(str(f) for f in p.glob(f"*{alias}*.parquet"))
        if files:
            return load_dataset("parquet", data_files=files, split="train")

    # Fallback: load all parquet files and auto-split
    full = load_dataset("parquet", data_dir=str(p), split="train")
    return _maybe_split(full, split)


def _maybe_split(dataset, split: str):
    """Auto-split into train/valid if needed."""
    if split == "train":
        if len(dataset) > 100:
            return dataset.train_test_split(test_size=0.05, seed=42)["train"]
        return dataset
    elif split in ("valid", "test", "validation"):
        if len(dataset) > 100:
            return dataset.train_test_split(test_size=0.05, seed=42)["test"]
        return dataset
    return dataset


def _split_aliases(split: str) -> tuple[str, ...]:
    return {
        "train": ("train",),
        "test": ("test", "valid", "validation"),
        "valid": ("valid", "validation", "test"),
    }.get(split, (split,))

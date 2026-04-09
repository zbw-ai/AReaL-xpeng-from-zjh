"""DeepMath 20k dataset loader for AReaL RL training.

Dataset format (Parquet):
    - prompt (list[dict]): messages list, e.g. [{"role":"user","content":"..."}]
    - reward_model (dict): {"ground_truth": "1", "style": "rule"}
    - extra_info (dict): {"answer": "1", "difficulty": 10, ...}

Key adaptation:
    The original prompt ends with "Answer: $Answer (without quotes)".
    This loader replaces it with "put your final answer within \\boxed{}"
    to align with the math_verify extractor used in reward computation.

Returns HuggingFace Dataset with columns: messages, answer
"""

import re
from pathlib import Path

from datasets import load_dataset

from areal.utils import logging

logger = logging.getLogger("DeepMathDataset")

# Original suffix in deepmath prompts
_ORIGINAL_SUFFIX = (
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem."
)

# Replacement suffix aligned with \boxed{} extractor
_BOXED_SUFFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def get_deepmath_rl_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    **kwargs,
):
    """Load DeepMath 20k dataset for RL training.

    Args:
        path: Path to parquet file or directory.
        split: "train" or "valid"/"test".
        tokenizer: Tokenizer for length filtering.
        max_length: Max prompt token length for filtering.

    Returns:
        HuggingFace Dataset with columns: messages, answer
    """
    dataset = _load_dataset(path, split)
    logger.info(f"Loaded {len(dataset)} samples from {path} (split={split})")

    def process(sample):
        # Extract user content from messages list
        prompt_messages = sample["prompt"]
        user_content = ""
        for msg in prompt_messages:
            if msg.get("role") == "user":
                user_content = msg["content"]
                break

        # Replace "Answer: $Answer" suffix with \boxed{} instruction
        user_content = _replace_answer_suffix(user_content)

        # Extract ground truth answer
        answer = _extract_answer(sample)

        messages = [{"role": "user", "content": user_content}]
        return {"messages": messages, "answer": answer}

    dataset = dataset.map(process)

    # Keep only messages and answer
    cols_to_remove = [
        c for c in dataset.column_names if c not in ("messages", "answer")
    ]
    dataset = dataset.remove_columns(cols_to_remove)

    if max_length is not None and tokenizer is not None:

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        before = len(dataset)
        dataset = dataset.filter(filter_length)
        logger.info(
            f"Filtered {before} -> {len(dataset)} samples (max_length={max_length})"
        )

    logger.info(f"Final dataset: {len(dataset)} samples")
    return dataset


def _replace_answer_suffix(text: str) -> str:
    """Replace 'Answer: $Answer ...' suffix with \\boxed{} instruction."""
    # Try exact match first
    if _ORIGINAL_SUFFIX in text:
        return text.replace(_ORIGINAL_SUFFIX, _BOXED_SUFFIX)

    # Fallback: regex for variations of "Answer: $Answer" pattern
    pattern = r"The last line.*?Answer:\s*\$Answer.*?(?:problem|question)\.?"
    replaced = re.sub(pattern, _BOXED_SUFFIX, text, flags=re.DOTALL)
    if replaced != text:
        return replaced

    # If no match, append \boxed{} instruction
    if "\\boxed{}" not in text:
        text = text.rstrip() + "\n" + _BOXED_SUFFIX
    return text


def _extract_answer(sample: dict) -> str:
    """Extract ground truth answer from reward_model or extra_info."""
    # Primary: reward_model.ground_truth
    rm = sample.get("reward_model", {})
    if isinstance(rm, dict) and rm.get("ground_truth"):
        return str(rm["ground_truth"])

    # Fallback: extra_info.answer
    ei = sample.get("extra_info", {})
    if isinstance(ei, dict) and ei.get("answer"):
        return str(ei["answer"])

    logger.warning(f"No answer found for sample: {sample.get('data_source', '?')}")
    return ""


def _load_dataset(path: str, split: str):
    """Load parquet dataset with auto train/valid split."""
    p = Path(path)

    if p.is_file():
        ds = load_dataset("parquet", data_files=str(p), split="train")
    elif p.is_dir():
        files = sorted(str(f) for f in p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {path}")
        ds = load_dataset("parquet", data_files=files, split="train")
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    # Auto split 95/5
    if split == "train":
        if len(ds) > 100:
            return ds.train_test_split(test_size=0.05, seed=42)["train"]
        return ds
    elif split in ("valid", "test", "validation"):
        if len(ds) > 100:
            return ds.train_test_split(test_size=0.05, seed=42)["test"]
        return ds
    return ds

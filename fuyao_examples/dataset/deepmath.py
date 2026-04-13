"""DeepMath 20k dataset loader for AReaL RL training.

Dataset format (Parquet):
    - prompt (list[dict]): messages list, e.g. [{"role":"user","content":"..."}]
    - reward_model (dict): {"ground_truth": "1", "style": "rule"}
    - extra_info (dict): {"answer": "1", "difficulty": 10, ...}

Key adaptations:
    1. Prompt: removes ALL "Answer: $Answer" instructions, replaces with \\boxed{}
    2. Answer: wraps non-numeric answers in \\boxed{} so math_verify can parse them
       (math_verify cannot parse bare "Yes", "No", "\\infty" etc.)

Returns HuggingFace Dataset with columns: messages, answer
"""

import re
from pathlib import Path

from datasets import load_dataset

from areal.utils import logging

logger = logging.getLogger("DeepMathDataset")

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

        # Clean prompt: remove ALL "Answer:" instructions, replace with \boxed{}
        user_content = _clean_prompt(user_content)

        # Extract ground truth answer and wrap for math_verify compatibility
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


def _clean_prompt(text: str) -> str:
    """Remove ALL 'Answer:' related instructions and replace with \\boxed{}.

    Handles multiple variations:
      - "The last line of your response should be of the form Answer: $Answer ..."
      - "Remember to put your answer on its own line after 'Answer:'."
      - "Solve the following math problem step by step."
    """
    # Remove "The last line ... Answer: $Answer ... problem."
    text = re.sub(
        r"The last line.*?Answer:\s*\$Answer.*?(?:problem|question)\.?\s*",
        "",
        text,
        flags=re.DOTALL,
    )
    # Remove "Remember to put your answer on its own line after 'Answer:'."
    text = re.sub(
        r"Remember to put your answer on its own line after ['\"]?Answer:['\"]?\.?\s*",
        "",
        text,
    )
    # Remove standalone "Solve the following math problem step by step." if we're adding our own
    text = re.sub(
        r"^Solve the following math problem step by step\.\s*",
        "",
        text,
    )
    # Clean up extra whitespace / newlines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Append \boxed{} instruction
    text = text + "\n\n" + _BOXED_SUFFIX
    return text


def _extract_answer(sample: dict) -> str:
    """Extract ground truth answer and wrap in \\boxed{} for math_verify.

    math_verify's gold extraction requires answers to be parseable.
    Bare strings like "Yes", "No", "\\infty" cause ValueError.
    Wrapping in \\boxed{} lets the LaTeX extractor handle them.
    """
    raw = ""
    # Primary: reward_model.ground_truth
    rm = sample.get("reward_model", {})
    if isinstance(rm, dict) and rm.get("ground_truth"):
        raw = str(rm["ground_truth"])
    # Fallback: extra_info.answer
    if not raw:
        ei = sample.get("extra_info", {})
        if isinstance(ei, dict) and ei.get("answer"):
            raw = str(ei["answer"])

    if not raw:
        logger.warning(f"No answer found for sample: {sample.get('data_source', '?')}")
        return ""

    # Wrap in \boxed{} so math_verify can parse it as LaTeX
    return "\\boxed{" + raw + "}"


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

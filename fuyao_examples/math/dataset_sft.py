"""Generic SFT dataset loader — auto-detects format from column names.

Supported formats:
  1. OpenThoughts3 (conversations): columns = [conversations, ...]
     conversations = [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]
  2. prompt/solution parquet: columns = [prompt, solution]
  3. messages jsonl/parquet: columns = [messages]
     messages = [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

All formats produce: {"input_ids": [...], "loss_mask": [...]}
  - loss_mask=0 for prompt/system tokens, loss_mask=1 for assistant response tokens.
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
    system_prompt: str | None = None,
):
    """Load SFT data from parquet/jsonl, auto-detecting format.

    Args:
        path: Path to file, directory, or HuggingFace dataset ID.
        split: "train" or "valid"/"test".
        tokenizer: HuggingFace tokenizer.
        max_length: Max sequence token length for filtering.
        system_prompt: Override system prompt (default: auto from data or none).
    """
    dataset = _load_dataset(path, split)
    logger.info(f"Loaded {len(dataset)} samples from {path} (split={split})")

    cols = dataset.column_names
    if "conversations" in cols:
        fmt = "conversations"
    elif "messages" in cols:
        fmt = "messages"
    elif "prompt" in cols and "solution" in cols:
        fmt = "prompt_solution"
    else:
        raise ValueError(
            f"Unknown dataset format. Columns: {cols}. "
            "Expected one of: conversations, messages, or prompt+solution."
        )
    logger.info(f"Detected format: {fmt}")

    def process(sample):
        messages = _to_messages(sample, fmt, system_prompt)
        return _tokenize_with_loss_mask(messages, tokenizer)

    dataset = dataset.map(process)

    # Keep only SFT columns
    cols_to_remove = [
        c for c in dataset.column_names if c not in ("input_ids", "loss_mask")
    ]
    dataset = dataset.remove_columns(cols_to_remove)

    if max_length is not None:
        before = len(dataset)
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
        logger.info(f"Filtered {before} -> {len(dataset)} (max_length={max_length})")

    logger.info(f"Final dataset: {len(dataset)} samples")
    return dataset


# ── Format converters ──


def _to_messages(sample, fmt: str, system_prompt: str | None):
    """Convert any format to standard [{"role": ..., "content": ...}] messages."""
    if fmt == "conversations":
        # OpenThoughts3: [{"from": "user", "value": "..."}, ...]
        convs = sample["conversations"]
        messages = []
        for turn in convs:
            role = "assistant" if turn["from"] == "assistant" else "user"
            messages.append({"role": role, "content": turn["value"]})
    elif fmt == "messages":
        messages = sample["messages"]
    elif fmt == "prompt_solution":
        messages = [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["solution"]},
        ]
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # Prepend system prompt if provided and not already present
    if system_prompt and (not messages or messages[0]["role"] != "system"):
        messages.insert(0, {"role": "system", "content": system_prompt})

    return messages


def _tokenize_with_loss_mask(messages, tokenizer):
    """Tokenize messages; mask prompt tokens (loss_mask=0), train on assistant (loss_mask=1)."""
    if hasattr(tokenizer, "apply_chat_template"):
        full_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        # Prompt = everything except the last assistant turn
        prompt_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                break
            prompt_messages.append(msg)
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation
        parts = [m["content"] for m in messages]
        full_text = "\n\n".join(parts) + tokenizer.eos_token
        prompt_parts = [m["content"] for m in messages if m["role"] != "assistant"]
        prompt_text = "\n\n".join(prompt_parts) + "\n\n"
        full_ids = tokenizer.encode(full_text)
        prompt_ids = tokenizer.encode(prompt_text)

    prompt_len = len(prompt_ids)
    loss_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
    return {"input_ids": full_ids, "loss_mask": loss_mask}


# ── Data loading ──


def _load_dataset(path: str, split: str):
    """Load dataset from file, directory, or HuggingFace hub."""
    p = Path(path)

    # Single file
    if p.is_file():
        suffix = p.suffix.lower()
        if suffix == ".parquet":
            ds = load_dataset("parquet", data_files=str(p), split="train")
        elif suffix in (".jsonl", ".json"):
            ds = load_dataset("json", data_files=str(p), split="train")
        else:
            ds = load_dataset("parquet", data_files=str(p), split="train")
        return _maybe_split(ds, split)

    # Directory with split subdirs (e.g., OpenThoughts3-1_2M/train/)
    if p.is_dir():
        for alias in _split_aliases(split):
            split_dir = p / alias
            if split_dir.is_dir():
                parquets = sorted(str(f) for f in split_dir.glob("*.parquet"))
                jsonls = sorted(str(f) for f in split_dir.glob("*.jsonl"))
                if parquets:
                    return load_dataset("parquet", data_files=parquets, split="train")
                if jsonls:
                    return load_dataset("json", data_files=jsonls, split="train")

        # No split subdir — load all files and auto-split
        parquets = sorted(str(f) for f in p.rglob("*.parquet"))
        jsonls = sorted(str(f) for f in p.rglob("*.jsonl"))
        if parquets:
            ds = load_dataset("parquet", data_files=parquets, split="train")
        elif jsonls:
            ds = load_dataset("json", data_files=jsonls, split="train")
        else:
            raise FileNotFoundError(f"No parquet/jsonl files found in {path}")
        return _maybe_split(ds, split)

    # Try as HuggingFace dataset ID
    return load_dataset(path, split=split)


def _maybe_split(dataset, split: str):
    """Auto-split 95/5 if no explicit split files."""
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

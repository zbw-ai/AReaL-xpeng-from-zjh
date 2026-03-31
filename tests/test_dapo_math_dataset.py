from datasets import Dataset

from areal.dataset import dapo_math


def test_dapo_math_dataset_uses_deterministic_holdout_for_validation(monkeypatch):
    dataset = Dataset.from_dict(
        {
            "prompt": [f"problem {i}" for i in range(20)],
            "solution": [f"work \\boxed{{{i}}}" for i in range(20)],
        }
    )

    def fake_load_dataset(*_args, **_kwargs):
        return dataset

    monkeypatch.setattr(dapo_math, "load_dataset", fake_load_dataset)

    train_dataset = dapo_math.get_dapo_math_rl_dataset("/tmp/dapo_math", split="train")
    valid_dataset = dapo_math.get_dapo_math_rl_dataset("/tmp/dapo_math", split="test")

    assert len(train_dataset) == 19
    assert len(valid_dataset) == 1
    assert train_dataset.column_names == ["messages", "answer"]
    assert valid_dataset.column_names == ["messages", "answer"]
    assert set(train_dataset["answer"]).isdisjoint(set(valid_dataset["answer"]))


def test_dapo_math_dataset_extracts_boxed_answer():
    answer = dapo_math._extract_boxed_answer("analysis \\boxed{42}")
    assert answer == "42"


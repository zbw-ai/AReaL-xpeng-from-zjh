"""Filter easy samples from training data (POLARIS-style).

After Stage-N training, run the trained model on the training set,
score with math_verify, and drop samples where avg_reward > threshold.
This produces a harder subset for the next training stage.

Usage (on GPU node with SGLang server running):
    # 1. Start SGLang server with trained checkpoint
    python -m sglang.launch_server \
        --model-path /path/to/checkpoint \
        --tp 4 --port 30000

    # 2. Run filtering
    python fuyao_examples/tools/filter_easy_data.py \
        --dataset-path /workspace/.../dapo_math_17k_processed \
        --dataset-type dapo_math \
        --server-url http://localhost:30000 \
        --n-samples 8 \
        --temperature 1.4 \
        --max-tokens 31744 \
        --threshold 0.9 \
        --output /workspace/.../dapo_math_filtered.parquet

    # 3. (Optional) Search optimal temperature for next stage
    python fuyao_examples/tools/filter_easy_data.py \
        --dataset-path /workspace/.../dapo_math_17k_processed \
        --dataset-type dapo_math \
        --server-url http://localhost:30000 \
        --mode search-temperature \
        --temps 1.4,1.45,1.5,1.55,1.6

Reference: POLARIS (HKU NLP + ByteDance Seed)
  - Between stages: drop samples with avg_reward > 0.9
  - Search temperature via Distinct 4-gram diversity metric
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fuyao_examples.reward import math_verify_with_fallback


def parse_args():
    p = argparse.ArgumentParser(description="Filter easy data (POLARIS-style)")
    p.add_argument("--dataset-path", required=True, help="Path to dataset dir or parquet")
    p.add_argument("--dataset-type", default="dapo_math", choices=["dapo_math", "deepmath"])
    p.add_argument("--server-url", default="http://localhost:30000",
                   help="SGLang server URL(s), comma-separated for multi-instance")
    p.add_argument("--n-samples", type=int, default=8, help="Rollouts per prompt")
    p.add_argument("--temperature", type=float, default=1.4)
    p.add_argument("--max-tokens", type=int, default=31744)
    p.add_argument("--threshold", type=float, default=0.9, help="Drop samples with avg_reward > threshold")
    p.add_argument("--output", required=True, help="Output parquet path")
    p.add_argument("--mode", default="filter", choices=["filter", "search-temperature", "stats-only"])
    p.add_argument("--temps", default="1.4,1.45,1.5,1.55,1.6", help="Temperatures to search (comma-separated)")
    p.add_argument("--batch-size", type=int, default=32, help="Concurrent requests")
    p.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    return p.parse_args()


def load_dataset_samples(path: str, dataset_type: str):
    """Load dataset and return list of dicts with prompt_text, answer, and raw row.

    Keeps the original raw row so we can save filtered data in the same format.
    """
    from datasets import load_dataset as hf_load_dataset

    # Load raw parquet (preserving original schema for output)
    p = Path(path)
    if p.is_file():
        raw_df = pd.read_parquet(str(p))
    elif p.is_dir():
        files = sorted(str(f) for f in p.glob("**/*.parquet"))
        raw_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        raise FileNotFoundError(f"Dataset path not found: {path}")

    print(f"Raw parquet columns: {list(raw_df.columns)}, rows: {len(raw_df)}")

    # Process each row to extract prompt_text + answer for scoring
    if dataset_type == "dapo_math":
        from fuyao_examples.dataset.dapo_math import _extract_boxed_answer
        samples = []
        for idx, row in raw_df.iterrows():
            prompt_text = row["prompt"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            answer = _extract_boxed_answer(row["solution"])
            samples.append({"idx": idx, "prompt": prompt_text, "answer": answer})
    elif dataset_type == "deepmath":
        import re
        samples = []
        for idx, row in raw_df.iterrows():
            prompt_messages = row.get("prompt", [])
            user_content = ""
            if isinstance(prompt_messages, list):
                for msg in prompt_messages:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        user_content = msg["content"]
                        break
            rm = row.get("reward_model", {})
            answer = str(rm.get("ground_truth", "")) if isinstance(rm, dict) else ""
            if not answer:
                ei = row.get("extra_info", {})
                answer = str(ei.get("answer", "")) if isinstance(ei, dict) else ""
            samples.append({"idx": idx, "prompt": user_content, "answer": answer})
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return samples, raw_df


def generate_one(server_url: str, prompt: str, n: int, temperature: float, max_tokens: int):
    """Generate N completions for one prompt via OpenAI-compatible API.

    Uses n=1 per request instead of n=N to avoid SGLang hanging on large
    batch-in-single-request. Sends N individual requests sequentially.
    """
    completions = []
    for _ in range(n):
        resp = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "n": 1,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 1.0,
            },
            timeout=(10, 180),  # (connect_timeout, read_timeout)
        )
        resp.raise_for_status()
        data = resp.json()
        completions.append(data["choices"][0]["message"]["content"])
    return completions


def score_sample(prompt: str, completions: list[str], answer: str) -> dict:
    """Score all completions for one sample, return per-sample stats."""
    rewards = []
    for comp in completions:
        try:
            r = math_verify_with_fallback(comp, answer)
        except Exception:
            r = 0.0
        rewards.append(r)

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return {
        "prompt": prompt,
        "answer": answer,
        "avg_reward": avg_reward,
        "n_correct": sum(1 for r in rewards if r > 0),
        "n_total": len(rewards),
        "rewards": rewards,
    }


def run_filter(args):
    """Main filter mode: generate, score, filter, save.

    Resilient design:
    - Per-future timeout (REQUEST_TIMEOUT_S) prevents single requests from blocking
    - Incremental checkpoint saves progress every CHECKPOINT_INTERVAL samples
    - Resumes from checkpoint if output.progress.json exists
    """
    REQUEST_TIMEOUT_S = 180  # per-request hard timeout (kill stuck futures)
    CHECKPOINT_INTERVAL = 500  # save progress every N completions

    print(f"Loading dataset from {args.dataset_path} (type={args.dataset_type})")
    samples, raw_df = load_dataset_samples(args.dataset_path, args.dataset_type)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Total samples: {len(samples)}")

    server_urls = [u.strip() for u in args.server_url.split(",")]
    print(f"Using {len(server_urls)} server(s): {server_urls}")

    # Resume from checkpoint if exists
    output_path = Path(args.output)
    progress_path = output_path.with_suffix(".progress.json")
    results = []
    done_indices = set()
    if progress_path.exists():
        with open(progress_path) as f:
            saved = json.load(f)
        results = saved.get("results", [])
        done_indices = {r["idx"] for r in results}
        print(f"Resumed from checkpoint: {len(results)} already scored, "
              f"skipping {len(done_indices)} samples")

    remaining = [s for s in samples if s["idx"] not in done_indices]
    print(f"Remaining to score: {len(remaining)}")
    if not remaining:
        print("All samples already scored!")
    else:
        print(f"\nGenerating {args.n_samples} completions per prompt "
              f"(temp={args.temperature}, timeout={REQUEST_TIMEOUT_S}s)...")

    failed = 0
    timed_out = 0
    total_done = len(done_indices)

    def process_one(idx, sample):
        url = server_urls[idx % len(server_urls)]
        try:
            completions = generate_one(
                url, sample["prompt"],
                args.n_samples, args.temperature, args.max_tokens,
            )
            result = score_sample(sample["prompt"], completions, sample["answer"])
            result["idx"] = sample["idx"]
            return result
        except Exception as e:
            return {"error": str(e), "prompt": sample["prompt"], "idx": sample["idx"]}

    # Process in batches — each batch has a hard wall-clock deadline
    batch_sz = args.batch_size * 2  # submit 2x batch_size at a time
    BATCH_DEADLINE_S = REQUEST_TIMEOUT_S * 3  # max time for entire batch

    for batch_start in range(0, len(remaining), batch_sz):
        batch = remaining[batch_start : batch_start + batch_sz]

        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            future_to_idx = {
                executor.submit(process_one, i + batch_start, s): s["idx"]
                for i, s in enumerate(batch)
            }

            # Use as_completed with batch-level timeout
            try:
                for future in as_completed(future_to_idx, timeout=BATCH_DEADLINE_S):
                    sample_idx = future_to_idx[future]
                    try:
                        result = future.result(timeout=10)  # already done, just collect
                    except Exception as e:
                        result = {"error": str(e), "idx": sample_idx}

                    if "error" in result:
                        failed += 1
                        if failed <= 10:
                            err_msg = result.get("error", "unknown")[:80]
                            print(f"  [WARN] Sample {result.get('idx','?')} failed: {err_msg}")
                    else:
                        results.append(result)

                    total_done += 1
                    if total_done % 100 == 0 or total_done == len(samples):
                        n_easy = sum(1 for r in results if r["avg_reward"] > args.threshold)
                        print(f"  [{total_done}/{len(samples)}] scored={len(results)} "
                              f"easy(>{args.threshold})={n_easy} "
                              f"failed={failed} timeout={timed_out}")

                    # Incremental checkpoint
                    if len(results) % CHECKPOINT_INTERVAL == 0 and len(results) > 0:
                        _save_checkpoint(progress_path, results)
                        print(f"  [checkpoint] saved {len(results)} results")

            except TimeoutError:
                # Batch deadline hit — cancel remaining futures, count as timeout
                n_pending = sum(1 for f in future_to_idx if not f.done())
                timed_out += n_pending
                total_done += n_pending
                for f in future_to_idx:
                    f.cancel()
                print(f"  [TIMEOUT] Batch deadline ({BATCH_DEADLINE_S}s) hit, "
                      f"skipped {n_pending} stuck requests. "
                      f"Total timeout={timed_out}")

        # Checkpoint after each batch
        if results:
            _save_checkpoint(progress_path, results)

    # Stats
    print(f"\n{'='*60}")
    print(f"Scoring complete: {len(results)} samples scored, {failed} failed")
    _print_stats(results, args.threshold)

    if args.mode == "stats-only":
        return

    # Filter
    kept = [r for r in results if r["avg_reward"] <= args.threshold]
    dropped = len(results) - len(kept)
    print(f"\nFiltering: drop {dropped} easy samples (avg_reward > {args.threshold})")
    print(f"Kept: {len(kept)} / {len(results)} ({100*len(kept)/max(len(results),1):.1f}%)")

    # Save as parquet — preserve original raw schema so loaders work unchanged
    kept_indices = [r["idx"] for r in kept]
    filtered_df = raw_df.loc[kept_indices].reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(str(output_path), index=False)
    print(f"Saved filtered dataset to: {output_path} ({len(filtered_df)} samples)")
    print(f"Schema preserved: {list(filtered_df.columns)}")

    # Also save full stats for analysis
    stats_path = output_path.with_suffix(".stats.json")
    stats = {
        "total_samples": len(results),
        "threshold": args.threshold,
        "dropped": dropped,
        "kept": len(kept),
        "temperature": args.temperature,
        "n_samples": args.n_samples,
        "avg_reward_distribution": {
            "0.0": sum(1 for r in results if r["avg_reward"] == 0.0),
            "0.0-0.25": sum(1 for r in results if 0 < r["avg_reward"] <= 0.25),
            "0.25-0.5": sum(1 for r in results if 0.25 < r["avg_reward"] <= 0.5),
            "0.5-0.75": sum(1 for r in results if 0.5 < r["avg_reward"] <= 0.75),
            "0.75-0.9": sum(1 for r in results if 0.75 < r["avg_reward"] <= 0.9),
            "0.9-1.0": sum(1 for r in results if r["avg_reward"] > 0.9),
        },
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to: {stats_path}")


def run_search_temperature(args):
    """Search optimal temperature for next stage via Distinct 4-gram diversity."""
    temps = [float(t) for t in args.temps.split(",")]
    print(f"Searching temperatures: {temps}")

    samples, _ = load_dataset_samples(args.dataset_path, args.dataset_type)
    # Use a subset for temperature search (faster)
    subset_size = min(200, len(samples))
    samples = samples[:subset_size]
    print(f"Using {subset_size} samples for temperature search")

    for temp in temps:
        print(f"\n--- Temperature: {temp} ---")
        all_tokens = []
        rewards = []

        for i, sample in enumerate(samples):
            try:
                completions = generate_one(
                    args.server_url, sample["prompt"],
                    args.n_samples, temp, args.max_tokens,
                )
                for comp in completions:
                    all_tokens.extend(comp.split())
                    try:
                        r = math_verify_with_fallback(comp, sample["answer"])
                    except Exception:
                        r = 0.0
                    rewards.append(r)
            except Exception:
                pass

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{subset_size}]")

        # Distinct 4-gram
        d4 = _distinct_n_gram(all_tokens, 4)
        avg_reward = sum(rewards) / max(len(rewards), 1)
        print(f"  Avg reward: {avg_reward:.4f}")
        print(f"  Distinct 4-gram: {d4:.4f}")
        print(f"  Total tokens: {len(all_tokens)}")


def _save_checkpoint(path: Path, results: list[dict]):
    """Save incremental progress to JSON."""
    # Strip large fields to keep checkpoint small
    slim = [
        {"idx": r["idx"], "avg_reward": r["avg_reward"],
         "n_correct": r["n_correct"], "n_total": r["n_total"]}
        for r in results
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"results": slim, "count": len(slim)}, f)


def _distinct_n_gram(tokens: list[str], n: int) -> float:
    """Compute Distinct N-gram metric (diversity indicator)."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def _print_stats(results: list[dict], threshold: float):
    """Print reward distribution stats."""
    rewards = [r["avg_reward"] for r in results]
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist = Counter()
    for r in rewards:
        for i in range(len(bins) - 1):
            if bins[i] <= r < bins[i + 1]:
                label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                hist[label] += 1
                break

    print(f"\nReward distribution (n={len(results)}):")
    print(f"  {'Bin':<12} {'Count':>6} {'Pct':>6}  {'Bar'}")
    for i in range(len(bins) - 1):
        label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        count = hist.get(label, 0)
        pct = 100 * count / max(len(results), 1)
        bar = "#" * int(pct / 2)
        marker = " <-- drop" if bins[i] >= threshold else ""
        print(f"  {label:<12} {count:>6} {pct:>5.1f}%  {bar}{marker}")

    above = sum(1 for r in rewards if r > threshold)
    print(f"\n  Above threshold ({threshold}): {above}/{len(results)} "
          f"({100*above/max(len(results),1):.1f}%)")
    print(f"  Avg reward: {sum(rewards)/max(len(rewards),1):.4f}")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "search-temperature":
        run_search_temperature(args)
    else:
        run_filter(args)

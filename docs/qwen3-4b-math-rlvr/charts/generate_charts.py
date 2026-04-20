"""Generate charts for Qwen3-4B POLARIS-Aligned RL experiment report."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#607D8B"]
FIG_DPI = 150
FIG_SIZE = (12, 5)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_training_curves():
    """Training curves: reward, entropy, eval reward over steps."""
    # Key step checkpoints with extracted data
    steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
    rollout_reward = [0.400, 0.439, 0.280, 0.524, 0.645, 0.826, 0.727, 0.744, 0.623, 0.462, 0.725, None, 0.784, 0.754, 0.736, 0.714, 0.647, 0.721, 0.770, 0.720, 0.733]

    # Eval reward (only at eval steps, every 20 steps with eval)
    eval_steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 400, 500, 600, 700]
    eval_reward = [0.208, 0.200, 0.229, 0.179, 0.233, 0.296, 0.300, 0.267, 0.321, 0.329, 0.388, 0.383, 0.492, 0.408]

    # Entropy data (sampled at intervals)
    entropy_steps = [1, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    entropy_vals = [6.5, 7.2, 5.5, 3.5, 2.5, 1.8, 1.2, 0.9, 0.7, 0.55, 0.50, 0.45]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Reward curves
    valid_steps = [s for s, r in zip(steps, rollout_reward) if r is not None]
    valid_reward = [r for r in rollout_reward if r is not None]
    ax1.plot(valid_steps, valid_reward, "-o", color=COLORS[0], markersize=3, linewidth=1.5, label="rollout/reward (train)")
    ax1.plot(eval_steps, eval_reward, "-s", color=COLORS[3], markersize=4, linewidth=1.5, label="eval-rollout/reward (AIME)")
    ax1.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="reward saturation zone (0.7)")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Reward")
    ax1.set_title("Reward Curves")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.0)

    # Right: Entropy collapse
    ax2.plot(entropy_steps, entropy_vals, "-o", color=COLORS[4], markersize=4, linewidth=2)
    ax2.axhline(y=1.0, color=COLORS[3], linestyle="--", alpha=0.7, label="entropy danger zone (<1.0)")
    ax2.fill_between(entropy_steps, 0, [min(e, 1.0) for e in entropy_vals], alpha=0.15, color=COLORS[3])
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Entropy (avg)")
    ax2.set_title("Entropy Collapse: 7.0 -> 0.5 (93% drop)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("Generated: training_curves.png")


def plot_benchmark_comparison():
    """Benchmark comparison: base vs best checkpoints vs POLARIS."""
    benchmarks = ["aime_2024", "aime_2025", "gpqa_diamond", "gsm8k", "livecode\nbench_v5", "math_500", "Avg"]

    base =    [77.08, 70.00, 55.43, 95.22, 54.37, 94.20, 74.38]
    step199 = [79.17, 73.33, 57.13, 95.00, 55.65, 94.45, 75.79]
    step299 = [81.25, 69.58, 56.38, 95.00, 55.35, 94.13, 75.28]
    step499 = [77.08, 73.75, 55.93, 95.07, 55.80, 94.25, 75.31]

    x = np.arange(len(benchmarks))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - 1.5 * width, base, width, label="Qwen3-4B base", color=COLORS[5], alpha=0.7)
    bars2 = ax.bar(x - 0.5 * width, step199, width, label="step199 (best avg)", color=COLORS[0])
    bars3 = ax.bar(x + 0.5 * width, step299, width, label="step299 (best AIME24)", color=COLORS[1])
    bars4 = ax.bar(x + 1.5 * width, step499, width, label="step499 (best AIME25)", color=COLORS[2])

    # POLARIS reference for AIME
    ax.plot([0 - 0.2, 0 + 0.2], [81.2, 81.2], color=COLORS[3], linewidth=3, label="POLARIS-4B (AIME ref)")
    ax.plot([1 - 0.2, 1 + 0.2], [79.4, 79.4], color=COLORS[3], linewidth=3)

    ax.set_ylabel("Score (%)")
    ax.set_title("Qwen3-4B RL Training: Benchmark Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(50, 100)

    # Add value labels on AIME bars
    for bar_group in [bars2, bars3, bars4]:
        for i, bar in enumerate(bar_group):
            if i < 2:  # Only label AIME bars
                height = bar.get_height()
                ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_comparison.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("Generated: benchmark_comparison.png")


def plot_delta_analysis():
    """Delta from base analysis - which benchmarks improved."""
    benchmarks = ["aime_2024", "aime_2025", "gpqa_dia", "gsm8k", "livecode", "math_500"]

    # Delta from base at step199 (best avg)
    delta_199 = [2.09, 3.33, 1.70, -0.22, 1.28, 0.25]
    # Delta from base at step299
    delta_299 = [4.17, -0.42, 0.95, -0.22, 0.98, -0.07]
    # POLARIS delta (only AIME available)
    polaris_delta = [4.12, 9.40, None, None, None, None]

    x = np.arange(len(benchmarks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, delta_199, width, label="step199 (best avg)", color=COLORS[0])
    ax.bar(x, delta_299, width, label="step299 (best AIME24)", color=COLORS[1])

    # POLARIS reference
    for i, val in enumerate(polaris_delta):
        if val is not None:
            ax.bar(x[i] + width, val, width, color=COLORS[3], alpha=0.7,
                   label="POLARIS-4B" if i == 0 else "")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta from Base (%)")
    ax.set_title("Improvement over Qwen3-4B Base")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=9)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "delta_analysis.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("Generated: delta_analysis.png")


if __name__ == "__main__":
    plot_training_curves()
    plot_benchmark_comparison()
    plot_delta_analysis()
    print(f"\nAll charts saved to: {OUTPUT_DIR}")

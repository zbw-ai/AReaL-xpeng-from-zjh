"""Generate experiment comparison charts for Qwen3-8B Math RLVR.

Run: python docs/qwen3-8b-math-rlvr/generate_experiment_charts.py
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
FIG_DPI = 150
FIG_SIZE = (10, 6)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def chart_p0_baseline():
    """P0 baseline evaluation: 4 models x 6 datasets."""
    datasets = ["AIME24", "AIME25", "GPQA", "GSM8K", "LiveCode", "Math500"]
    base = [12.08, 8.75, 29.23, 81.50, 15.21, 61.08]
    sft3905 = [72.08, 53.75, 57.83, 95.00, 43.75, 93.60]
    sft4450 = [71.25, 65.00, 56.57, 94.77, 48.19, 94.98]
    instruct = [79.58, 70.83, 61.55, 95.38, 55.65, 94.43]

    x = np.arange(len(datasets))
    width = 0.2

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(x - 1.5 * width, base, width, label="Base", color=COLORS[3], alpha=0.8)
    ax.bar(x - 0.5 * width, sft3905, width, label="SFT-3905", color=COLORS[2], alpha=0.8)
    ax.bar(x + 0.5 * width, sft4450, width, label="SFT-4450", color=COLORS[1], alpha=0.8)
    ax.bar(x + 1.5 * width, instruct, width, label="Instruct", color=COLORS[0], alpha=0.8)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("P0 Baseline Evaluation: 4 Models x 6 Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 105)

    for bars, vals in [(x - 1.5 * width, base), (x + 1.5 * width, instruct)]:
        for xi, v in zip(bars, vals):
            ax.text(xi, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_p0_baseline.png"), dpi=FIG_DPI)
    plt.close()
    print("Generated 07_p0_baseline.png")


def chart_p3_reward_comparison():
    """P3 three-experiment reward comparison snapshot."""
    experiments = [
        "R2a\nInstruct\n(no think)",
        "R2b\nSFT-4450\n(no think)",
        "R2c\nInstruct\n(+think)",
    ]
    reward = [0.398, 0.594, 0.594]
    incorrect_pct = [60, 41, 41]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reward
    bars1 = ax1.bar(experiments, reward, color=[COLORS[3], COLORS[1], COLORS[0]], alpha=0.85)
    ax1.set_ylabel("Reward (avg)")
    ax1.set_title("P3 Reward Comparison (snapshot)")
    ax1.set_ylim(0, 0.8)
    for bar, v in zip(bars1, reward):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Incorrect %
    bars2 = ax2.bar(experiments, incorrect_pct, color=[COLORS[3], COLORS[1], COLORS[0]], alpha=0.85)
    ax2.set_ylabel("Incorrect Sequences (%)")
    ax2.set_title("P3 Error Rate Comparison (snapshot)")
    ax2.set_ylim(0, 80)
    for bar, v in zip(bars2, incorrect_pct):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_p3_reward_comparison.png"), dpi=FIG_DPI)
    plt.close()
    print("Generated 08_p3_reward_comparison.png")


def chart_p3_infra_comparison():
    """P3 infra metrics comparison."""
    experiments = ["R2a Instruct", "R2b SFT-4450", "R2c Instruct+think"]
    train_step = [42, 54, 58]
    rollout_wait = [0, 29, 72]
    recompute = [8.7, 12.9, 14.3]
    ref_logp = [8.2, 12.2, 13.6]

    x = np.arange(len(experiments))
    width = 0.2

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(x - 1.5 * width, train_step, width, label="train_step", color=COLORS[0])
    ax.bar(x - 0.5 * width, rollout_wait, width, label="rollout wait", color=COLORS[2])
    ax.bar(x + 0.5 * width, recompute, width, label="recompute_logp", color=COLORS[1])
    ax.bar(x + 1.5 * width, ref_logp, width, label="ref_logp", color=COLORS[4])

    ax.set_ylabel("Time (seconds)")
    ax.set_title("P3 Phase Time Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_p3_infra_comparison.png"), dpi=FIG_DPI)
    plt.close()
    print("Generated 09_p3_infra_comparison.png")


def chart_4b_infra_comparison():
    """4B three-scenario infra comparison."""
    scenarios = ["RLVR Math", "Code DAPO\n(early)", "Code DAPO\n(late)", "Search-R1"]
    train_step = [12, 20, 70, 20]
    response_len = [2000, 1000, 4500, 1000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[4]]

    bars1 = ax1.bar(scenarios, train_step, color=colors, alpha=0.85)
    ax1.set_ylabel("train_step (seconds)")
    ax1.set_title("4B Train Step Time by Scenario")
    for bar, v in zip(bars1, train_step):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v}s",
                 ha="center", va="bottom", fontsize=10)

    bars2 = ax2.bar(scenarios, response_len, color=colors, alpha=0.85)
    ax2.set_ylabel("Response Length (tokens)")
    ax2.set_title("4B Response Length by Scenario")
    for bar, v in zip(bars2, response_len):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 50, f"{v}",
                 ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_4b_infra_comparison.png"), dpi=FIG_DPI)
    plt.close()
    print("Generated 10_4b_infra_comparison.png")


if __name__ == "__main__":
    chart_p0_baseline()
    chart_p3_reward_comparison()
    chart_p3_infra_comparison()
    chart_4b_infra_comparison()
    print(f"All charts saved to {OUTPUT_DIR}")

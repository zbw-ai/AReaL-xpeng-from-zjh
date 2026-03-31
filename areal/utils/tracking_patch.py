"""DeepInsight SwanLab metric mapping for AReaL.

Adds renamed metrics (deepinsight_infra/*, deepinsight_algorithm/*) to stats
before they are committed to SwanLab/WandB, matching the ROLL tracking convention.

Usage:
    In stats_logger.py commit(), call apply_metric_mapping(item) before logging.
    Or monkey-patch StatsLogger at startup.
"""

# AReaL metric name → DeepInsight metric name
DEEPINSIGHT_METRIC_MAPPING: dict[str, str] = {
    # ── Infra metrics ──
    "timeperf/recompute_logp": "deepinsight_infra/ref_logp_time",
    "timeperf/train_step": "deepinsight_infra/backward_step_time",
    "timeperf/rollout": "deepinsight_infra/rollout_step_time",
    "timeperf/update_weights": "deepinsight_infra/sync_weight_time",
    "timeperf/compute_advantage": "deepinsight_infra/compute_advantage_time",
    # ── Algorithm metrics ──
    "ppo_actor/task_reward/avg": "deepinsight_algorithm/reward",
    "ppo_actor/update/entropy/avg": "deepinsight_algorithm/entropy",
    "ppo_actor/update/actor_loss/avg": "deepinsight_algorithm/policy_loss",
    "ppo_actor/kl_rewards/avg": "deepinsight_algorithm/kl_loss",
    "ppo_actor/update/clip_ratio/avg": "deepinsight_algorithm/clip_ratio",
    "ppo_actor/seq_len/avg": "deepinsight_algorithm/response_length",
    "ppo_actor/update/grad_norm": "deepinsight_algorithm/grad_norm",
}


def apply_metric_mapping(values: dict) -> dict:
    """Add DeepInsight-renamed copies of metrics to values dict (in-place).

    Does not remove originals — both AReaL and DeepInsight names are logged.
    """
    for original_key in list(values.keys()):
        if original_key in DEEPINSIGHT_METRIC_MAPPING:
            new_key = DEEPINSIGHT_METRIC_MAPPING[original_key]
            values[new_key] = values[original_key]
    return values

"""DeepInsight SwanLab metric mapping for AReaL — external monkey-patch.

Adds renamed metrics (deepinsight_infra/*, deepinsight_algorithm/*) to stats
before they are committed to SwanLab/WandB, matching the ROLL tracking convention.

Usage:
    Call apply_tracking_patch() before trainer.train() in your train script.
"""

from areal.utils import logging

logger = logging.getLogger("TrackingPatch")

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


def _add_deepinsight_metrics(values: dict) -> dict:
    for original_key in list(values.keys()):
        if original_key in DEEPINSIGHT_METRIC_MAPPING:
            values[DEEPINSIGHT_METRIC_MAPPING[original_key]] = values[original_key]
    return values


def apply_tracking_patch():
    """Monkey-patch StatsLogger.commit to add DeepInsight metric mapping."""
    from areal.utils.stats_logger import StatsLogger

    original_commit = StatsLogger.commit

    def patched_commit(self, epoch, step, global_step, data):
        # Intercept data before committing to add DeepInsight metrics
        if isinstance(data, dict):
            _add_deepinsight_metrics(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    _add_deepinsight_metrics(item)
        return original_commit(self, epoch, step, global_step, data)

    StatsLogger.commit = patched_commit
    logger.info("Applied DeepInsight metric mapping patch to StatsLogger")

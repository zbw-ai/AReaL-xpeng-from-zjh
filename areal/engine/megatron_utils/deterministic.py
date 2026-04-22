import os

import torch

from areal.utils import logging

logger = logging.getLogger("MCoreDeterm")


def disable_qwen3_5_incompatible_fusions(model_config):
    """Disable Megatron fusion kernels that are incompatible with mbridge's
    Qwen3.5 gated attention (``_apply_output_gate`` shape mismatch).

    Matches veRL's ``qwen3_5`` preset which sets
    ``ENABLE_TRANSFORMER_FUSION_OVERRIDES=False`` for the same reason.

    See also: veRL ``scripts/run_rlvr_megatron.sh`` line 471 comment —
    "Some models (e.g. qwen3.5 with mbridge) are incompatible with
    apply_rope_fusion."
    """
    model_config.apply_rope_fusion = False
    model_config.masked_softmax_fusion = False
    model_config.bias_activation_fusion = False
    model_config.bias_dropout_fusion = False
    model_config.gradient_accumulation_fusion = False
    logger.info(
        "Disabled Megatron fusions (apply_rope_fusion, masked_softmax_fusion, "
        "bias_activation_fusion, bias_dropout_fusion, gradient_accumulation_fusion) "
        "for Qwen3.5 compatibility."
    )


def set_deterministic_algorithms(model_config):
    """
    args: Megatron args, acquired by get_args()
    """
    model_config.deterministic_mode = True
    model_config.cross_entropy_loss_fusion = False
    model_config.bias_dropout_fusion = False

    # Set env variables about deterministic mode
    if os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") != "0":
        logger.info(
            "For deterministic algo, env [NVTE_ALLOW_NONDETERMINISTIC_ALGO] will be set to '0'."
        )
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
    if os.getenv("NCCL_ALGO") not in all_reduce_choices:
        logger.info("For deterministic algo, env [NCCL_ALGO] will be set to 'Ring'.")
        os.environ["NCCL_ALGO"] = "Ring"

    cublas_workspace_config_choices = [":4096:8", ":16:8"]
    if os.getenv("CUBLAS_WORKSPACE_CONFIG") not in cublas_workspace_config_choices:
        logger.info(
            "For deterministic algo, env [CUBLAS_WORKSPACE_CONFIG] will be set to ':4096:8'."
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True, warn_only=True)

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
    before = {
        "apply_rope_fusion": getattr(model_config, "apply_rope_fusion", None),
        "masked_softmax_fusion": getattr(model_config, "masked_softmax_fusion", None),
        "bias_activation_fusion": getattr(model_config, "bias_activation_fusion", None),
        "bias_dropout_fusion": getattr(model_config, "bias_dropout_fusion", None),
        "gradient_accumulation_fusion": getattr(
            model_config, "gradient_accumulation_fusion", None
        ),
    }
    model_config.apply_rope_fusion = False
    model_config.masked_softmax_fusion = False
    model_config.bias_activation_fusion = False
    model_config.bias_dropout_fusion = False
    model_config.gradient_accumulation_fusion = False
    # Print to stderr so it's visible regardless of logger configuration.
    print(
        f"[disable_qwen3_5_incompatible_fusions] was={before}, "
        f"all now False (for Qwen3.5 compat; matches veRL qwen3_5 preset).",
        flush=True,
    )
    logger.info(
        "Disabled Megatron fusions (apply_rope_fusion, masked_softmax_fusion, "
        "bias_activation_fusion, bias_dropout_fusion, gradient_accumulation_fusion) "
        "for Qwen3.5 compatibility."
    )

    # Megatron-LM 0.18 dev (commit 20ba03f) transformer_layer.py:314-320 forwards
    # `cp_comm_type` to attention's build_module when `cp_size > 1`. This was
    # designed for standard self-attention's CP comm patterns, but Qwen3.5 uses
    # GatedDeltaNet (`experimental_attention_variant="gated_delta_net"`), and
    # GatedDeltaNet.__init__ does NOT accept `cp_comm_type` kwarg → TypeError.
    #
    # Mbridge sets `cp_comm_type="p2p"` unconditionally in _build_config; we have
    # to suppress it here so that Megatron's `config.cp_comm_type is not None`
    # check returns False and the kwarg is not forwarded to GDN.
    #
    # This is safe because: (a) cp_size=1 path doesn't read this field; (b) GDN
    # uses its own all-to-all (cp2hp/hp2cp) pattern, not config.cp_comm_type.
    if (
        getattr(model_config, "experimental_attention_variant", None) == "gated_delta_net"
        and getattr(model_config, "cp_comm_type", None) is not None
    ):
        before_cp = model_config.cp_comm_type
        model_config.cp_comm_type = None
        print(
            f"[disable_qwen3_5_incompatible_fusions] cp_comm_type was={before_cp}, "
            f"now None (GDN attention does not accept cp_comm_type kwarg).",
            flush=True,
        )
        logger.info(
            "Cleared cp_comm_type (was %r) — Qwen3.5 GDN attention does not "
            "accept cp_comm_type kwarg via Megatron's transformer_layer.",
            before_cp,
        )

    # Disable Multi-Token Prediction (MTP) for RL training:
    #   1. RL doesn't need MTP — it's a pretraining/inference acceleration feature.
    #   2. Megatron 0.18 dev MTP module (multi_token_prediction.py:905) has a hidden-dim
    #      mismatch bug under cp_size>1: _concat_embeddings does
    #      `torch.cat((decoder_input, hidden_states), -1)` where decoder_input has
    #      hidden=H/TP but hidden_states (after GDN CP) has hidden=H/TP/CP, raising
    #      "Sizes of tensors must match except in dimension 2" RuntimeError.
    #   3. mbridge's _build_mtp_config unconditionally enables MTP whenever
    #      hf_config.text_config.mtp_num_hidden_layers > 0 (Qwen3.5 ships with this set).
    if getattr(model_config, "mtp_num_layers", 0) and model_config.mtp_num_layers > 0:
        before_mtp = model_config.mtp_num_layers
        model_config.mtp_num_layers = 0
        # Also clear loss scaling so any residual MTP probe does not contribute to loss.
        if hasattr(model_config, "mtp_loss_scaling_factor"):
            model_config.mtp_loss_scaling_factor = 0.0
        print(
            f"[disable_qwen3_5_incompatible_fusions] mtp_num_layers was={before_mtp}, "
            f"now 0 (RL training does not use MTP; avoids cp>1 hidden-dim mismatch).",
            flush=True,
        )
        logger.info(
            "Disabled MTP (mtp_num_layers %d → 0): RL training does not need MTP, "
            "and Megatron 0.18 dev MTP _concat_embeddings has hidden-dim bug under cp>1.",
            before_mtp,
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

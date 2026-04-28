from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams


def preprocess_packed_seqs_context_parallel(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences.
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1 gets second and second last chunks, and so on),
    this is for load balancing with causal masking. See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = input_lens.max().item()
    batch_size = input_lens.shape[0]

    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    align_to_multiple_of = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    # assume input_ids and cu_seqlens are already padded to align_to_multiple_of
    if any(length % align_to_multiple_of for length in input_lens) != 0:
        raise ValueError(
            f"Some of the input sequence length ({input_lens}) is not a multiple of align_to_multiple_of {align_to_multiple_of} "
            "for context/sequence parallel in Megatron."
        )

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_seqlen,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_kv=max_seqlen,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
    )

    if cp_size <= 1:
        return input_ids.unsqueeze(0), packed_seq_params

    shape = (input_lens.sum().item() // cp_size,)
    splitted = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
    for i in range(batch_size):
        seqlen = input_lens[i] // cp_size
        half_seqlen = seqlen // 2
        start_idx = cu_seqlens[i] // cp_size
        # split to 2 chunks
        d = input_ids[cu_seqlens[i] : cu_seqlens[i + 1]]
        splitted[start_idx : start_idx + half_seqlen] = d[
            half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
        ]

        remain_start = input_lens[i] - half_seqlen * (cp_rank + 1)
        remain_end = input_lens[i] - half_seqlen * cp_rank
        remain_end = min(remain_end, d.shape[0])
        remain_len = remain_end - remain_start
        splitted[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
            remain_start:remain_end
        ]
    return splitted.unsqueeze(0), packed_seq_params


def _bshd_cp_zigzag_gather(output: torch.Tensor) -> torch.Tensor:
    """All-gather BSHD output `[B, S/cp, ...]` across CP group and unzigzag back to `[B, S, ...]`.

    GDN's CP scheme (mamba_context_parallel.py) and TE's CP both follow the same
    zigzag chunk distribution for causal-attention load balance:

        Total seq is split into ``2*cp_size`` chunks. Rank ``i`` holds two chunks:
            - chunk index ``i``               (front half)
            - chunk index ``2*cp_size-1-i``   (back half)
        Each rank's local seq is the concatenation: ``[chunk_i, chunk_{2*cp_size-1-i}]``.

    This function reverses that distribution by all-gathering and re-assembling.

    Args:
        output: Per-rank BSHD tensor of shape ``[B, S_local, ...]`` where
            ``S_local = S_full / cp_size`` (must be divisible by 2).
    Returns:
        Full BSHD tensor of shape ``[B, S_full, ...]``.
    """
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    cp_group = mpu.get_context_parallel_group()
    if cp_size <= 1:
        return output

    output = output.contiguous()
    # All-gather: gathered[i] is rank i's local output (NOT autograd-aware here;
    # we only run this in inference paths — compute_logp / eval forward — where
    # outputs need not be backward-differentiable through the all-gather. For
    # train-step BSHD CP we must use a differentiable variant before turning on.)
    gathered = [torch.empty_like(output) for _ in range(cp_size)]
    dist.all_gather(gathered, output.detach(), group=cp_group)
    # Make sure local rank's slot keeps the autograd-tracked tensor.
    gathered[cp_rank] = output

    # Each rank's local seq is two zigzag chunks: front_chunk + back_chunk.
    # local_len = S_full / cp_size, half = local_len // 2 = S_full / (2*cp_size)
    local_len = output.shape[1]
    assert local_len % 2 == 0, (
        f"BSHD CP local seq length must be even (zigzag), got {local_len}"
    )
    half = local_len // 2
    full_len = local_len * cp_size
    # 2*cp_size chunks each of size `half`
    chunks: list[torch.Tensor | None] = [None] * (2 * cp_size)
    for i in range(cp_size):
        local = gathered[i]
        front, back = local[:, :half, ...], local[:, half:, ...]
        chunks[i] = front
        chunks[2 * cp_size - 1 - i] = back

    # Concat in chunk order along seq dim
    full = torch.cat(chunks, dim=1)
    assert full.shape[1] == full_len, (
        f"BSHD CP unzigzag shape mismatch: got {full.shape[1]}, expected {full_len}"
    )
    return full


def postprocess_packed_seqs_context_parallel(
    output: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    post_process: bool,
) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    cp_size = mpu.get_context_parallel_world_size()
    if not post_process:
        return output
    if cp_size <= 1:
        return output.squeeze(0)
    if cu_seqlens is None:
        # BSHD path under cp_size>1: model returned `[B, S/cp, V]` after GDN's
        # internal head-parallel CP. Re-assemble full `[B, S, V]` via all-gather +
        # zigzag unshuffle so caller-side ops (logprobs gather, loss) can index
        # into full input_ids.
        return _bshd_cp_zigzag_gather(output)
    # shape = [batch_size, seq_len] + list(output.shape[2:])
    # [1, packed, dim] -> [batch_size, seq_len, dim]
    batch_size = cu_seqlens.shape[0] - 1
    output_len = int(cu_seqlens[-1].item())
    # output shape: [total_packed_seq_len] + list(output.shape[2:]
    output_new = torch.empty(
        (output_len, *output.shape[2:]), device=output.device, dtype=output.dtype
    )
    # all gather output across context parallel group
    # need to gather across cp group and concatenate in sequence dimension
    output_list = [torch.empty_like(output) for _ in range(cp_size)]
    dist.all_gather(
        output_list, output.detach(), group=mpu.get_context_parallel_group()
    )
    output_list[mpu.get_context_parallel_rank()] = output

    for i in range(batch_size):
        seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
        splitted_seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]) // cp_size
        half_splitted_seq_len = splitted_seq_len // 2

        tmp = torch.empty(
            (seq_len, *output.shape[2:]), device=output.device, dtype=output.dtype
        )
        for j in range(cp_size):
            o = output_list[j].squeeze(0)
            # split to 2 chunks
            start = cu_seqlens[i] // cp_size
            o0, o1 = (
                o[start : start + half_splitted_seq_len],
                o[start + half_splitted_seq_len : start + splitted_seq_len],
            )
            tmp[j * half_splitted_seq_len : (j + 1) * half_splitted_seq_len] = o0
            splitted_start = seq_len - (j + 1) * half_splitted_seq_len
            splitted_end = seq_len - j * half_splitted_seq_len
            tmp[splitted_start:splitted_end] = o1

        output_new[cu_seqlens[i] : cu_seqlens[i + 1]] = tmp[:seq_len]
    return output_new


def packed_context_parallel_forward(
    model: torch.nn.Module,
    input_: dict[str, Any],
):
    input_ids = input_["input_ids"]
    position_ids = input_["position_ids"]
    cu_seqlens = input_.get("cu_seqlens", None)
    # `attention_mask`: dense torch.Tensor (flex attention with Megatron) or None.
    # `tree_triton_data`: read from a separate key; takes priority over
    # attention_mask when forwarded as the final attention mask argument.
    attention_mask = input_.get("attention_mask", None)
    tree_triton_data = input_.get("tree_triton_data", None)
    packed_seq_params = None

    if cu_seqlens is not None:
        if attention_mask is not None or tree_triton_data is not None:
            raise ValueError(
                "Attention mask should be None when using packed sequences."
            )
        input_ids, packed_seq_params = preprocess_packed_seqs_context_parallel(
            input_ids, cu_seqlens
        )
        input_ids = input_ids.contiguous()

    # Pass tree_triton_data as attention_mask if present (for Triton tree attention)
    # Otherwise use the attention_mask from input (could be dense tensor for flex attention)
    final_attention_mask = (
        tree_triton_data if tree_triton_data is not None else attention_mask
    )

    # Diagnostic: print exactly what we pass to model (only once per process).
    # Previous runs had conflicting signals — our dict had no cu_seqlens but
    # the error message showed packed_seq_params populated. This confirms.
    global _packed_ctx_fwd_debug_printed
    try:
        _packed_ctx_fwd_debug_printed
    except NameError:
        _packed_ctx_fwd_debug_printed = False  # module-level flag
    if not _packed_ctx_fwd_debug_printed:
        try:
            import sys as _sys
            print(
                f"[packed_context_parallel_forward debug] "
                f"input_ids.shape={tuple(input_ids.shape)}, "
                f"attention_mask={type(final_attention_mask).__name__ if final_attention_mask is not None else 'None'}"
                f"{tuple(final_attention_mask.shape) if final_attention_mask is not None else ''}, "
                f"position_ids={type(position_ids).__name__ if position_ids is not None else 'None'}, "
                f"packed_seq_params={packed_seq_params}, "
                f"input_keys={list(input_.keys())}",
                file=_sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        _packed_ctx_fwd_debug_printed = True

    try:
        output = model(
            input_ids=input_ids,
            attention_mask=final_attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )
    except Exception as e:
        raise RuntimeError(
            f"Error occurred in packed context parallel forward pass on model {model} "
            f"with input_ids shape {input_ids.shape} and packed_seq_params {packed_seq_params} "
            f"and attention_mask {'None' if final_attention_mask is None else tuple(final_attention_mask.shape)}."
        ) from e

    model_vp_stage = getattr(model, "vp_stage", None)
    is_pipeline_last_stage = mpu.is_pipeline_last_stage(
        ignore_virtual=False, vp_stage=model_vp_stage
    )
    output = postprocess_packed_seqs_context_parallel(
        output, cu_seqlens, is_pipeline_last_stage
    )
    return output

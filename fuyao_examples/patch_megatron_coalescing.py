"""Patch megatron's param_and_grad_buffer.py to bypass NCCL-incompatible
`_coalescing_manager`.

PyTorch's NCCL backend never implemented `reduce_scatter_tensor_coalesced` /
`allgather_into_tensor_coalesced` (only Gloo has them). Megatron's
`start_grad_sync` and `start_param_sync` wrap per-bucket ops in
`with _coalescing_manager(...)`, which raises on __exit__:

  RuntimeError: Backend nccl does not support reduce_scatter_tensor_coalesced

This crashes step 1's optimizer phase. Upstream issues unfixed:
  - NVIDIA/Megatron-LM#1369
  - pytorch/pytorch#134833

Fix: shadow `_coalescing_manager` in this module with a function that returns
`nullcontext()`. The inner per-bucket reduce_scatter_tensor /
all_gather_into_tensor calls are NCCL-supported and execute fine without the
coalescing wrapper — just slightly less efficient.

Safe because we force `overlap_grad_reduce=False` / `overlap_param_gather=False`
in our recipes (handle is None either way).

Idempotent: re-running on a patched file is a no-op.
"""

import sys
import megatron.core.distributed.param_and_grad_buffer as pgb_mod

target = pgb_mod.__file__
with open(target) as f:
    src = f.read()

if "PATCHED_FOR_NCCL_COALESCING_BUG" in src:
    print(f"[OK] Patch already applied at {target}")
    sys.exit(0)

needle = "from torch.distributed import _coalescing_manager"
if needle not in src:
    print(f"[FAIL] Cannot find import in {target}", file=sys.stderr)
    sys.exit(1)

replacement = (
    "from torch.distributed import _coalescing_manager as _orig_coalescing_manager"
    "  # noqa: F401  PATCHED_FOR_NCCL_COALESCING_BUG\n"
    "from contextlib import nullcontext as _nullctx\n"
    "def _coalescing_manager(*_args, **_kwargs):\n"
    "    # PATCHED: NCCL backend lacks reduce_scatter_tensor_coalesced /\n"
    "    # allgather_into_tensor_coalesced. Return nullcontext so per-bucket\n"
    "    # NCCL ops run individually (which IS supported).\n"
    "    return _nullctx()\n"
)

with open(target, "w") as f:
    f.write(src.replace(needle, replacement, 1))

print(f"[OK] Patched {target}")

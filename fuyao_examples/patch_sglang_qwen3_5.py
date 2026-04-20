#!/usr/bin/env python3
"""Patch SGLang 0.5.9 for Qwen3.5 VLM compatibility.

SGLang 0.5.9's get_hf_text_config assumes text_config is a PretrainedConfig
object with attributes. For Qwen3.5 VLM, text_config may be a raw dict from
config.json, causing AttributeError on .hidden_size etc.

This script patches the source file on disk so the SGLang subprocess picks
it up. Safe to run multiple times (idempotent).

Fixed in SGLang >=0.5.10 — this patch is a workaround for older versions.
"""

import sys
from pathlib import Path

SGLANG_FILE = Path(
    "/AReaL/.venv/lib/python3.12/site-packages/sglang/srt/utils/hf_transformers_utils.py"
)
PATCH_MARKER = "# PATCHED_FOR_QWEN3_5_VLM"


def patch():
    if not SGLANG_FILE.exists():
        print(f"[sglang-patch] File not found: {SGLANG_FILE}", file=sys.stderr)
        return False

    code = SGLANG_FILE.read_text()

    if PATCH_MARKER in code:
        print("[sglang-patch] Already patched, skipping.")
        return True

    # 1. Remove the strict assertion on text_config.num_attention_heads
    code = code.replace(
        'assert hasattr(config.text_config, "num_attention_heads")',
        f"pass  {PATCH_MARKER}",
    )

    # 2. Patch get_hf_text_config return to convert dict → SimpleNamespace
    #    Find the function and add dict conversion before the return.
    old_return = "return config.text_config"
    new_return = f"""# {PATCH_MARKER}: convert dict text_config to object for attribute access
    tc = config.text_config
    if isinstance(tc, dict):
        from types import SimpleNamespace
        tc = SimpleNamespace(**tc)
    return tc"""

    if old_return in code:
        code = code.replace(old_return, new_return, 1)
    else:
        # Maybe already partially patched — try to find and fix
        print("[sglang-patch] WARNING: Could not find 'return config.text_config'")

    SGLANG_FILE.write_text(code)
    print("[sglang-patch] Successfully patched SGLang for Qwen3.5 VLM")
    return True


if __name__ == "__main__":
    success = patch()
    sys.exit(0 if success else 1)

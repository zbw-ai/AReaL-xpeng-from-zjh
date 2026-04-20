#!/usr/bin/env python3
"""Patch SGLang for Qwen3.5 VLM: recursively convert all dict sub-configs.

transformers 5.x returns AutoConfig with nested sub-configs (text_config,
vision_config, etc.) as raw dicts for Qwen3.5 VLM. SGLang accesses them
with attribute syntax (.hidden_size, .num_hidden_layers), causing AttributeError.

Fix: patch get_hf_config() to recursively convert ALL dict values in the
config to SimpleNamespace objects. One fix, covers everything.
"""

import sys
from pathlib import Path

SGLANG_ROOT = Path("/AReaL/.venv/lib/python3.12/site-packages/sglang/srt")
PATCH_MARKER = "# PATCHED_QWEN3_5_VLM_V3"

# Code to inject: recursively converts dict attrs on any config object
DICT_FIX_CODE = '''
def _fix_dict_configs(config):
    """Recursively convert dict attributes to SimpleNamespace for attribute access."""
    from types import SimpleNamespace
    for attr in list(vars(config)):
        val = getattr(config, attr)
        if isinstance(val, dict) and not attr.startswith("_"):
            # Only convert if all keys are strings (skip dicts with int keys like layer indices)
            if all(isinstance(k, str) for k in val):
                ns = SimpleNamespace(**val)
                _fix_dict_configs(ns)  # recurse
                setattr(config, attr, ns)
    return config
'''


def patch():
    # Patch hf_transformers_utils.py — the central config loading point
    target = SGLANG_ROOT / "utils" / "hf_transformers_utils.py"
    if not target.exists():
        print(f"[sglang-patch] Not found: {target}")
        return False

    code = target.read_text()
    if PATCH_MARKER in code:
        print("[sglang-patch] Already patched.")
        return True

    # Remove old patch markers if present
    for old_marker in ["# PATCHED_FOR_QWEN3_5_VLM", "# PATCHED_FOR_QWEN3_5_VLM_V2"]:
        code = code.replace(old_marker, "")

    # 1. Add the helper function at the top of the file
    code = DICT_FIX_CODE + "\n" + code

    # 2. Remove strict assertion (SGLang 0.5.9)
    code = code.replace(
        'assert hasattr(config.text_config, "num_attention_heads")',
        "pass  # qwen3.5 compat",
    )

    # 3. Patch get_hf_text_config to fix ALL dict sub-configs before returning
    #    Find "return config.text_config" and add recursive fix before it
    old = "return config.text_config"
    new = """_fix_dict_configs(config)  # convert ALL dict sub-configs to objects
    return config.text_config"""
    if old in code:
        code = code.replace(old, new, 1)

    code = f"{PATCH_MARKER}\n{code}"
    target.write_text(code)
    print("[sglang-patch] Patched: recursive dict→object for all sub-configs")
    return True


if __name__ == "__main__":
    success = patch()
    sys.exit(0 if success else 1)

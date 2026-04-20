#!/usr/bin/env python3
"""Patch SGLang for Qwen3.5 VLM compatibility (all versions).

Qwen3.5 VLM's text_config is a raw dict (not a PretrainedConfig object) when
loaded by certain transformers versions. SGLang accesses .hidden_size,
.num_hidden_layers etc. on it, causing AttributeError.

Root-cause fix: patch get_hf_text_config() to convert dict → SimpleNamespace.
Also patch model_config.py to ensure hf_text_config is always an object.

Safe to run multiple times (idempotent). Applies to SGLang 0.5.9 AND 0.5.10.
"""

import sys
from pathlib import Path

SGLANG_ROOT = Path("/AReaL/.venv/lib/python3.12/site-packages/sglang/srt")
PATCH_MARKER = "# PATCHED_FOR_QWEN3_5_VLM_V2"


def patch_file(filepath: Path, replacements: list[tuple[str, str]]) -> bool:
    if not filepath.exists():
        print(f"[sglang-patch] File not found: {filepath}")
        return False

    code = filepath.read_text()
    if PATCH_MARKER in code:
        print(f"[sglang-patch] Already patched: {filepath.name}")
        return True

    changed = False
    for old, new in replacements:
        if old in code:
            code = code.replace(old, new, 1)
            changed = True

    if changed:
        code = f"{PATCH_MARKER}\n{code}"
        filepath.write_text(code)
        print(f"[sglang-patch] Patched: {filepath.name}")
    else:
        print(f"[sglang-patch] No patterns matched in {filepath.name}")
    return True


def patch():
    ok = True

    # 1. Patch hf_transformers_utils.py — get_hf_text_config()
    ok &= patch_file(
        SGLANG_ROOT / "utils" / "hf_transformers_utils.py",
        [
            # Remove strict assertion (0.5.9)
            (
                'assert hasattr(config.text_config, "num_attention_heads")',
                "pass  # qwen3.5 compat",
            ),
            # Convert dict text_config → SimpleNamespace before returning
            (
                "return config.text_config",
                """tc = config.text_config
    if isinstance(tc, dict):
        from types import SimpleNamespace
        tc = SimpleNamespace(**tc)
        config.text_config = tc  # persist for downstream access
    return tc""",
            ),
        ],
    )

    # 2. Patch model_config.py — ensure hf_text_config is always object
    model_config = SGLANG_ROOT / "configs" / "model_config.py"
    if model_config.exists():
        code = model_config.read_text()
        if PATCH_MARKER not in code:
            # Add dict→object conversion after hf_text_config assignment
            old = "self.hf_text_config = get_hf_text_config(self.hf_config)"
            new = """self.hf_text_config = get_hf_text_config(self.hf_config)
        if isinstance(self.hf_text_config, dict):
            from types import SimpleNamespace
            self.hf_text_config = SimpleNamespace(**self.hf_text_config)"""
            if old in code:
                code = f"{PATCH_MARKER}\n" + code.replace(old, new, 1)
                model_config.write_text(code)
                print(f"[sglang-patch] Patched: model_config.py")
            else:
                print(f"[sglang-patch] model_config.py: pattern not found, skipping")

    print("[sglang-patch] Done.")
    return ok


if __name__ == "__main__":
    success = patch()
    sys.exit(0 if success else 1)

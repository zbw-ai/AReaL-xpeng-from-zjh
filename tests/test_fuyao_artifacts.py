from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_math_fuyao_config_uses_vllm_and_no_hardcoded_swanlab_key():
    config_text = (REPO_ROOT / "fuyao_examples/math/qwen3_4b_rlvr.yaml").read_text()

    assert 'backend: "vllm:d8p1t1"' in config_text
    assert "api_key:" not in config_text


def test_fuyao_run_script_avoids_global_process_kills():
    script_text = (REPO_ROOT / "fuyao_examples/fuyao_areal_run.sh").read_text()

    assert "pkill -9" not in script_text
    assert "kill -0" in script_text
    assert ".fuyao_run" in script_text


def test_fuyao_dockerfile_is_present_and_vllm_first():
    dockerfile_text = (REPO_ROOT / "areal_fuyao.dockerfile").read_text()

    assert "uv sync --extra cuda" in dockerfile_text
    assert "vLLM-first Fuyao patch runtime" in dockerfile_text
    assert "pip install swanlab" in dockerfile_text
    assert "pip install httpx" in dockerfile_text

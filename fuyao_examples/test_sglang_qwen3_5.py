# fuyao_examples/test_sglang_qwen3_5.py
"""Validate SGLang can serve Qwen3.5-35B-A3B with TP=4."""

import subprocess
import sys
import time

import requests

MODEL_PATH = "/dataset_rc_b1/models/Qwen3.5-35B-A3B"
TP_SIZE = 4
PORT = 30000


def main():
    # Start SGLang server
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--tp", str(TP_SIZE),
        "--port", str(PORT),
        "--dtype", "bfloat16",
        "--context-length", "32768",
        "--mem-fraction-static", "0.82",
        "--disable-custom-all-reduce",
    ]
    print(f"Starting SGLang: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)

    # Wait for server
    for i in range(120):
        try:
            resp = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if resp.status_code == 200:
                print(f"Server ready after {i}s")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        proc.kill()
        raise RuntimeError("SGLang server failed to start in 120s")

    # Test generation
    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": "What is 2+2? Think step by step."}],
        "max_tokens": 256,
        "temperature": 0.7,
    }
    resp = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    content = result["choices"][0]["message"]["content"]
    print(f"Response ({len(content)} chars): {content[:200]}...")

    proc.kill()
    print("SGLang Qwen3.5 validation PASSED")


if __name__ == "__main__":
    main()

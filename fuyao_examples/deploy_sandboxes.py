#!/usr/bin/env python3
"""Deploy search and code sandbox containers via Fuyao SDK.

Training runs locally; this script deploys remote sandbox services
(search retrieval, code execution) to the Fuyao cluster.

Usage:
    # Deploy search sandbox
    python fuyao_examples/deploy_sandboxes.py --sandbox search

    # Deploy code sandbox
    python fuyao_examples/deploy_sandboxes.py --sandbox code

    # Deploy and export endpoints
    eval $(python fuyao_examples/deploy_sandboxes.py --sandbox search --export)
"""

import argparse
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Optional, Tuple

# ── Fuyao SDK imports ──
try:
    import fuyao
    from xbigdata.fuyao.sdkv2.etl import constant, model

    HAS_FUYAO = True
except ImportError:
    HAS_FUYAO = False

# ── Configuration ─────────────────────────────────────────────────
FUYAO_API_URI = "http://fuyao-v2-api.xiaopeng.link"
FUYAO_API_KEY = "abf55cb029834a389ec944c6b1e5f06b"

SEARCH_IMAGE = "infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260106-0037"
CODE_IMAGE = "infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260316-0159"

DEFAULT_SITE = "fuyao_b1"
DEFAULT_GPU_PARTITION = "rc-llmrl-a100"
DEFAULT_CPU_PARTITION = "rc-cpu"
DEFAULT_EXPERIMENT = "zhangjh37/llm_rl"

SEARCH_PORT = 8001
EXECD_PORT = 39524

CODE_CPUS = 14
CODE_MEMORY_GIB = 28

SEARCH_START_CMD = (
    "cd /workspace/zhangjh37@xiaopeng.com/code/xpeng_retriever && "
    "/opt/conda/envs/retriever/bin/python3 main.py --config configs/default.yaml --port 8001"
)

SANDBOX_PORTS_DIR = "/workspace/zhangjh37@xiaopeng.com/shared_lib/sandbox_ports"

POLL_INTERVAL = 10
MAX_WAIT_RUNNING = {"search": 3600, "code": 600}
HEALTH_CHECK_INTERVAL = 5
MAX_HEALTH_WAIT = {"search": 1200, "code": 300}


def init_fuyao():
    """Initialize Fuyao SDK connection."""
    if not HAS_FUYAO:
        print("ERROR: fuyao SDK not installed. Install with: pip install fuyao", file=sys.stderr)
        sys.exit(1)
    user_name = os.getenv("AUTH_USER")
    if not user_name:
        raise ValueError("AUTH_USER environment variable is not set")
    fuyao.etl.init(
        fuyao_api_uri=FUYAO_API_URI,
        fuyao_api_key=FUYAO_API_KEY,
        user_name=user_name,
    )


def deploy_job(args) -> Optional[str]:
    """Deploy a job to Fuyao. Returns job name (run_name)."""
    init_fuyao()
    deploy_run_args = model.DeployRunArgs(
        docker_image=args.docker_image,
        site=args.site,
        partition=args.partition,
        node_count=args.node_count,
        gpus_per_node=args.gpus_per_node,
        experiment=args.experiment,
        start_command=args.start_command,
        artifact_path=getattr(args, "artifact_path", os.getcwd()),
        label=getattr(args, "label", "areal_sandbox"),
        cpus_per_node=getattr(args, "cpus_per_node", 0),
        gibs_per_node=getattr(args, "gibs_per_node", 0),
        device_type=getattr(args, "device_type", "A100"),
        envs={"enable_prometheus_metrics": "true"},
    )
    try:
        result = fuyao.etl.deploy_run(deploy_run_args)
        print(f"[deploy] run_name: {result.data.run_name}, code: {result.code}")
        return result.data.run_name
    except Exception as e:
        print(f"[deploy] Error: {e}", file=sys.stderr)
        return None


def query_job(job_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Query job status. Returns (state, host_ip)."""
    try:
        run_info = fuyao.etl.get_run_by_name(job_name=job_name)
        fuyao_job = run_info.fuyao_job
        state = fuyao_job.state
        host_ip = None
        if state == "JOB_RUNNING":
            pod_info = fuyao.etl.search_run_pods(
                run_name=job_name, site=fuyao_job.site
            )
            host_ip = pod_info.pods[0].host_ip if pod_info.pods else None
        return state, host_ip
    except Exception as e:
        print(f"[query] Error: {e}", file=sys.stderr)
        return None, None


def _make_code_start_cmd() -> Tuple[str, str]:
    """Build code sandbox start command. Returns (cmd, port_file_path)."""
    ts = int(time.time())
    port_file = f"{SANDBOX_PORTS_DIR}/agentic_code_{ts}.port"
    cmd = (
        "bash -lc '"
        "TASK_NAME=agentic-code "
        f"EXECD_PORT={EXECD_PORT} "
        f"EXECD_PORT_FILE={port_file} "
        "BOOTSTRAP_APT_MIRROR=1 "
        "EXECD_BINARY_PATH=/workspace/zhangjh37@xiaopeng.com/shared_lib/execd "
        "exec /bin/bash /workspace/zhangjh37@xiaopeng.com/shared_lib/fuyao_sandbox_start_code.sh"
        "'"
    )
    return cmd, port_file


def _read_port_file(port_file: Optional[str], default: int) -> int:
    """Read actual port from port file (Fuyao may override via MASTER_PORT)."""
    if port_file:
        try:
            with open(port_file) as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            pass
    return default


def deploy_sandbox(
    sandbox_type: str, label: str = None
) -> Tuple[str, Optional[str]]:
    """Deploy a sandbox. Returns (job_name, port_file_or_None)."""
    port_file = None
    if sandbox_type == "search":
        args = SimpleNamespace(
            docker_image=SEARCH_IMAGE,
            site=DEFAULT_SITE,
            partition=DEFAULT_GPU_PARTITION,
            node_count=1,
            gpus_per_node=8,
            experiment=DEFAULT_EXPERIMENT,
            start_command=SEARCH_START_CMD,
            label=label or "areal_search_sandbox",
            cpus_per_node=0,
            gibs_per_node=0,
            device_type="A100",
        )
    elif sandbox_type == "code":
        start_cmd, port_file = _make_code_start_cmd()
        args = SimpleNamespace(
            docker_image=CODE_IMAGE,
            site=DEFAULT_SITE,
            partition=DEFAULT_CPU_PARTITION,
            node_count=1,
            gpus_per_node=0,
            experiment=DEFAULT_EXPERIMENT,
            start_command=start_cmd,
            label=label or "areal_code_sandbox",
            cpus_per_node=CODE_CPUS,
            gibs_per_node=CODE_MEMORY_GIB,
            device_type="CPU",
        )
    else:
        raise ValueError(f"Unknown sandbox type: {sandbox_type}")

    print(f"[deploy] Deploying {sandbox_type} sandbox...")
    run_name = deploy_job(args)
    if not run_name:
        print(f"[deploy] ERROR: Failed to deploy {sandbox_type}", file=sys.stderr)
        sys.exit(1)
    return run_name, port_file


def wait_for_running(
    job_name: str, timeout: int = 600, sandbox_type: str = ""
) -> str:
    """Poll until job is JOB_RUNNING. Returns host_ip."""
    print(f"[wait] Waiting for {job_name} (timeout={timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        state, host_ip = query_job(job_name)
        if state == "JOB_RUNNING" and host_ip:
            print(f"[wait] RUNNING on {host_ip}")
            return host_ip
        if state and "FAIL" in state:
            print(f"[wait] ERROR: {job_name} entered {state}", file=sys.stderr)
            sys.exit(1)
        elapsed = int(time.time() - start)
        print(f"[wait] {state} (elapsed {elapsed}s)")
        time.sleep(POLL_INTERVAL)
    print(f"[wait] ERROR: Timeout waiting for {job_name}", file=sys.stderr)
    sys.exit(1)


def wait_for_health(
    host_ip: str, port: int, path: str = "/health", timeout: int = 300
) -> bool:
    """Poll HTTP health endpoint."""
    url = f"http://{host_ip}:{port}{path}"
    print(f"[health] Waiting for {url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[health] OK (elapsed {int(time.time()-start)}s)")
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
    print(f"[health] ERROR: Timeout for {url}", file=sys.stderr)
    return False


def wait_for_tcp(host_ip: str, port: int, timeout: int = 300) -> bool:
    """Poll TCP port."""
    print(f"[tcp] Waiting for {host_ip}:{port} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((host_ip, port))
            s.close()
            print(f"[tcp] OK (elapsed {int(time.time()-start)}s)")
            return True
        except (OSError, TimeoutError):
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
    print(f"[tcp] ERROR: Timeout for {host_ip}:{port}", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="Deploy Fuyao sandbox containers for AReaL")
    parser.add_argument(
        "--sandbox",
        nargs="+",
        choices=["search", "code"],
        required=True,
        help="Sandbox type(s) to deploy",
    )
    parser.add_argument("--export", action="store_true", help="Print shell export statements")
    parser.add_argument("--no-wait", action="store_true", help="Deploy without waiting")
    args = parser.parse_args()

    endpoints = {}
    jobs = {}
    port_files = {}

    for sb_type in args.sandbox:
        job_name, port_file = deploy_sandbox(sb_type)
        jobs[sb_type] = job_name
        if port_file:
            port_files[sb_type] = port_file

    if args.no_wait:
        for sb_type, name in jobs.items():
            print(f"[done] {sb_type}: {name}")
        return

    for sb_type, job_name in jobs.items():
        timeout = MAX_WAIT_RUNNING.get(sb_type, 600)
        host_ip = wait_for_running(job_name, timeout=timeout, sandbox_type=sb_type)

        if sb_type == "search":
            wait_for_health(
                host_ip,
                SEARCH_PORT,
                "/health",
                timeout=MAX_HEALTH_WAIT["search"],
            )
            endpoints["RETRIEVAL_ENDPOINT"] = f"http://{host_ip}:{SEARCH_PORT}/retrieve"
        elif sb_type == "code":
            actual_port = _read_port_file(port_files.get(sb_type), EXECD_PORT)
            wait_for_tcp(host_ip, actual_port, timeout=MAX_HEALTH_WAIT["code"])
            endpoints["EXECD_ENDPOINT"] = f"http://{host_ip}:{actual_port}"

    print("\n" + "=" * 60)
    for k, v in endpoints.items():
        print(f"  {k}={v}")
    print("=" * 60)

    if args.export:
        for k, v in endpoints.items():
            print(f'export {k}="{v}"')


if __name__ == "__main__":
    main()

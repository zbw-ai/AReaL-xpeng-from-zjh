FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV MAX_JOBS=1
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /code/AReaL_xpeng
COPY . .

# ── Core AReaL dependencies (vLLM-first Fuyao patch runtime) ──
RUN pip install uv && \
    uv sync --extra cuda

# ── Fuyao SDK compatibility for current base image ──
RUN pip install --ignore-installed blinker && \
    pip3 install -U fuyao-all \
        --extra-index-url http://nexus-wl.xiaopeng.link:8081/repository/ai_infra_pypi/simple \
        --trusted-host nexus-wl.xiaopeng.link

# ── Fuyao patch dependencies ──
# M1: SwanLab experiment tracking (AReaL already supports, ensure installed)
RUN pip install swanlab

# M3: httpx for Search R1 retrieval
RUN pip install httpx

# ── Environment ──
ENV PYTHONPATH="/code/AReaL_xpeng:${PYTHONPATH}"
ENV CUDA_DEVICE_MAX_CONNECTIONS="1"

CMD ["/bin/bash"]

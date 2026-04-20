FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v1-260417-0107
ENV MAX_JOBS=1

# 所有新装的包都用 --no-deps + --target 写进 venv。
# --no-deps 防止连锁升级 torch/transformers/protobuf（会砸坏 Ray）。
# --target 写进 venv 的 site-packages（训练进程跑的是 /AReaL/.venv/bin/python3）。
RUN pip install --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
        megatron-core==0.16.0 \
        mbridge==0.15.1 \
        megatron-bridge==0.3.0 \
        vllm==0.18.0 \
        sglang==0.5.10.post1 \
        sglang-kernel==0.4.1 \
        trackio==0.2.2 \
        googleapis-common-protos==1.74.0

# 验证：构建时 sanity check，如果任何包装不进去就让 build 失败
RUN /AReaL/.venv/bin/python3 -c "\
import megatron.core, megatron.bridge, mbridge, vllm, sglang, trackio; \
from google.rpc import code_pb2; \
print('[sanity] all imports OK')"

FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v1-260417-0107
ENV MAX_JOBS=1

# 不和 megatron/ 命名空间冲突的包直接 --target 装
RUN pip install --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
        vllm==0.18.0 \
        sglang==0.5.10.post1 \
        sglang-kernel==0.4.1 \
        trackio==0.2.2 \
        googleapis-common-protos==1.74.0

# megatron 家族特殊处理：装到 scratch → cp 合并进 venv/megatron/
# 原因：megatron-core / megatron-bridge / mbridge(依赖megatron-core) 共享
# venv 的 megatron/ 命名空间。pip --upgrade --target 会整目录删重建，
# 导致后装的包覆盖前装的包的 megatron/ 子目录。
RUN pip install --no-deps --target /tmp/megatron-scratch \
        megatron-core==0.16.0 \
        mbridge==0.15.1 \
        megatron-bridge==0.3.0 \
 && cp -r /tmp/megatron-scratch/. /AReaL/.venv/lib/python3.12/site-packages/ \
 && rm -rf /tmp/megatron-scratch

# 构建时 sanity check — 有任何 import 失败立刻让 build 挂，不会把坏镜像 push 出去
RUN /AReaL/.venv/bin/python3 -c "\
import megatron.core as mc; print('megatron-core:', mc.__version__); \
import megatron.bridge; print('megatron.bridge OK'); \
import mbridge; print('mbridge OK'); \
import vllm, sglang, trackio; \
from google.rpc import code_pb2; \
print('[sanity] all imports OK')"

# Build from the SWE-MILE repository root: docker build -t swe-mile:local .
ARG BASE_IMAGE=verlai/verl:vllm017.latest
FROM ${BASE_IMAGE}
ARG VERL_GIT_REF=v0.8.0
WORKDIR /workspace/SWE-MILE

RUN python -m pip install uv && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git patch ca-certificates cgroup-tools iproute2 iptables libseccomp2 skopeo umoci util-linux && \
    rm -rf /var/lib/apt/lists/*

# Keep the CUDA 12.9 / torch 2.10 stack and the Qwen gated-delta kernels aligned.
RUN printf '%s\n' 'transformers==5.8.0' 'numpy==1.26.4' 'torch==2.10.0' > /tmp/swe-mile-constraints.txt && \
    uv pip install --system --override /tmp/swe-mile-constraints.txt \
      'transformers==5.8.0' 'vllm==0.17.0' 'ray==2.54.0' 'renderers==0.1.11' \
      'qwen-vl-utils==0.0.14' 'mistral-common==1.10.0' 'cupy-cuda12x==13.6.0' \
      packaging ninja einops && \
    uv pip install --system --no-build-isolation --no-deps 'causal-conv1d==1.6.2.post1' && \
    uv pip install --system --override /tmp/swe-mile-constraints.txt \
      'flash-linear-attention[cuda]==0.5.1' 'apache-tvm-ffi==0.1.9' 'tilelang==0.1.9' \
      'fastapi==0.135.2' 'starlette==0.52.1' 'nvidia-ml-py==13.590.48'
RUN python -m pip uninstall -y pynvml

COPY pyproject.toml ./pyproject.toml
COPY rllm-model-gateway ./rllm-model-gateway
RUN uv pip install --system --override /tmp/swe-mile-constraints.txt \
      -e ./rllm-model-gateway -r pyproject.toml
RUN git clone --depth 1 --branch "${VERL_GIT_REF}" https://github.com/verl-project/verl.git /opt/verl
COPY rllm/patches ./rllm/patches
RUN git -C /opt/verl apply --check /workspace/SWE-MILE/rllm/patches/verl-0.8.0-checkpoint-worker-ipc-allocator.patch && \
    git -C /opt/verl apply /workspace/SWE-MILE/rllm/patches/verl-0.8.0-checkpoint-worker-ipc-allocator.patch && \
    git -C /opt/verl apply --check /workspace/SWE-MILE/rllm/patches/verl-0.8.0-checkpoint-save-hf-override.patch && \
    git -C /opt/verl apply /workspace/SWE-MILE/rllm/patches/verl-0.8.0-checkpoint-save-hf-override.patch && \
    uv pip install --system --no-deps -e /opt/verl

COPY . .
RUN uv pip install --system --no-deps -e . && \
    python -m rllm.utils.vllm_runtime_compat && \
    python -c "from rllm.utils.vllm_runtime_compat import verify_qwen_runtime; verify_qwen_runtime()" && \
    python -c "from rllm.sandbox.minisandbox_runtime.oci_helper import _load_seccomp_library; assert _load_seccomp_library() is not None" && \
    python -m rllm.utils.runtime_provenance --write-build /workspace/SWE-MILE/runtime-build.json
CMD ["/bin/bash"]

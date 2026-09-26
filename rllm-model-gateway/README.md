# Model gateway

Bundled SWE-MILE gateway, derived from [rLLM](https://github.com/rllm-org/rllm).
See the [English](../README.md) or [Chinese](../README.zh-CN.md) guide for installation and launch commands.

The trainer starts and supervises the gateway automatically. It provides group-aware sticky routing across vLLM workers, token/log-probability trace capture, session tombstones, cancellation propagation, and worker recovery. No separate service deployment is required.

The Python package uses `hatchling`; install with `uv pip install -e ./rllm-model-gateway` from the repository root.

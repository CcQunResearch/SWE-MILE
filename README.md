# SWE-MILE

[中文](README.zh-CN.md) | English

**Asynchronous Potential-Induced Milestone Credit Assignment for Long-Horizon Software Engineering Agents.** SWE-MILE combines final verifier outcomes with navigation, verification-potential changes, backward credit, and tool-format rewards. Codeflow rollouts use independent primary/shadow sandboxes; VERL performs fully asynchronous RLOO/PPO training through the bundled model gateway.

Built on [rLLM](https://github.com/rllm-org/rllm) and [VERL](https://github.com/verl-project/verl). The `rllm` Python namespace is retained. This release contains the SWE pipeline and its runtime dependencies; upstream license notices are preserved.

## 1. Environment

Use Linux, NVIDIA GPUs, and the provided image. It pins the Qwen3.5 runtime (CUDA 12.9, PyTorch 2.10.0, VERL 0.8.0, vLLM 0.17.0, Transformers 5.8.0) and applies the included runtime patches.

```bash
docker build -t swe-mile:local .
docker run --gpus all --ipc=host --network=host --privileged --cgroupns=host \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
  -v '<path/to/exp_root>:/experiments' \
  -v '<path/to/models>:/models:ro' \
  -v '<path/to/local_nvme>:/sandbox-local' \
  -it swe-mile:local
```

Inside the image, set real paths before running any launcher:

```bash
cd /workspace/SWE-MILE
export EXP_ROOT=/experiments/run1
export DATA_ROOT=/experiments/data
export MODEL_PATH='/models/<model_directory>'
```

Replace all `<...>` placeholders. `MODEL_PATH` must contain Hugging Face weights, model configuration, and tokenizer files. On a cluster, use the same image, code, model path, and shared `DATA_ROOT`/`EXP_ROOT` on every node. Local sandbox scratch storage must be node-local. If modifying the checkout in this image, reinstall with `uv pip install --system --no-deps -e ./rllm-model-gateway -e .`.

**Sandbox choices** (`swe.sandbox_backend`):

- `minisandbox` (default): based on [SWE-MiniSandbox](https://github.com/lblankl/SWE-MiniSandbox), with the necessary OCI runtime extensions bundled in `rllm/sandbox/minisandbox_runtime`. Requires Linux root, writable **cgroup v1**, namespaces, overlayfs, libseccomp, and the system tools installed by the Dockerfile. Reserve sufficient local disk for unpacked images and concurrent tasks; startup requires at least 32 GiB free. Container privileges do not convert a cgroup v2 host into v1. Prepare the OCI caches below before launching workers.
- `k8s`: connect your own Kubernetes sandbox service with `export SWE_MILE_K8S_ADAPTER=your_package.sandbox:create_sandbox`, install that package on every Ray node, and append `++swe.sandbox_backend=k8s` to train/eval. The adapter contract is described [below](#k8s-adapter-contract); no cluster service is bundled. Local privileged execution and OCI caches are unnecessary for this choice.

## 2. Materialize data

`launch/prepare_dataset.sh` prepares task directories and registry Parquet files, resumes completed work, and validates dataset contracts. All seven families are enabled by default; disable unneeded families with the switches below. Dataset credentials, when required, are supplied through `HF_TOKEN`. Images come from public registries unless `SWE_MILE_IMAGE_MAP_FILE` names a JSON object mapping original image references to your mirror references. For private OCI registries, supply `MINISANDBOX_REGISTRY_AUTH_FILE` pointing to a containers-auth JSON file.

```bash
# Default bug-repair training data + SWE-bench Verified evaluation, including OCI caches.
BUILD_R2EGYM=0 BUILD_SWEBENCH_PRO_PUBLIC=0 BUILD_DENOVOSWE=0 \
BUILD_NL2REPO=0 BUILD_DOC2REPO=0 \
BUILD_SWEREBENCH_V2_PYTHON_MINISANDBOX_CACHE=1 \
BUILD_SWEBENCH_VERIFIED_MINISANDBOX_CACHE=1 \
bash launch/prepare_dataset.sh
```

| Dataset | Materialization switch | MiniSandbox cache switch |
|---|---|---|
| R2E-Gym | `BUILD_R2EGYM` | `BUILD_R2EGYM_MINISANDBOX_CACHE` |
| SWE-bench Verified | `BUILD_VERIFIED` | `BUILD_SWEBENCH_VERIFIED_MINISANDBOX_CACHE` |
| SWE-bench Pro Public | `BUILD_SWEBENCH_PRO_PUBLIC` | `BUILD_SWEBENCH_PRO_PUBLIC_MINISANDBOX_CACHE` |
| SWE-rebench V2 | `BUILD_SWEREBENCH_V2` | `BUILD_SWEREBENCH_V2_PYTHON_MINISANDBOX_CACHE` or `BUILD_SWEREBENCH_V2_PROMATCHED_MINISANDBOX_CACHE` |
| DeNovoSWE | `BUILD_DENOVOSWE` | `BUILD_DENOVOSWE_MINISANDBOX_CACHE` |
| NL2Repo-Bench | `BUILD_NL2REPO` | `BUILD_NL2REPO_MINISANDBOX_CACHE` |
| BeyondSWE Doc2Repo | `BUILD_DOC2REPO` | `BUILD_DOC2REPO_MINISANDBOX_CACHE` |

Switch values are `0`/`1`; cache switches default to `0`. For DeNovoSWE, enable its materialization and cache switches and disable unneeded families. K8s users leave cache switches at `0` and make the task images available to their service. `RLLM_MATERIALIZATION_WORKERS` controls task preparation; `MINISANDBOX_IMAGE_WORKERS` controls image preparation. For R2E-Gym milestone training also set `R2EGYM_MATERIALIZE_MILESTONE_METADATA=1`. SWE-rebench preparation creates Python and language-matched views; training derives the default `python-filternorm` subset using the bundled task index.

Outputs live under `DATA_ROOT/{tasks,rllm_home,hf_home,minisandbox}`. Training can additionally enforce a supplied eligibility manifest with `++swe.eligibility_manifest=<path> ++swe.require_eligibility_manifest=true`; it is optional by default.

## 3. Train

Train profiles are `TASK_PROFILE=bug_repair` (default) and `TASK_PROFILE=denovoswe`. The former selects SWE-rebench V2 Python filternorm; use `++swe.train_dataset=r2egym` to select prepared R2E-Gym. The latter selects DeNovoSWE.

```bash
# On an existing Ray cluster: 16 training GPUs + 16 separate rollout GPUs.
export RAY_ADDRESS='<head_ip>:6379'
TASK_PROFILE=bug_repair bash launch/train.sh \
  trainer.nnodes=2 trainer.n_gpus_per_node=8 \
  rollout.nnodes=2 rollout.n_gpus_per_node=8 \
  ++swe.minisandbox.local_root=/sandbox-local
```

For a fresh allocation, run `launch/cluster.sh` on **every node**, setting `RANK=0..N-1`, a common `MASTER_ADDR`, `TRAIN_NODES`, `ROLLOUT_NODES`, and `GPUS_PER_NODE`. `N=TRAIN_NODES+ROLLOUT_NODES`; the helper waits for the GPUs and starts training on rank 0. Pass the same profile, paths, and overrides on all nodes. Its default allocation is 4 training + 4 rollout nodes, each with 8 GPUs.

The default sequence parallelism is 16: it must divide the **training** GPU count and be compatible with the model's attention heads. For another allocation, explicitly set both `actor_rollout_ref.actor.ulysses_sequence_parallel_size` and `actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size`. Size rollout concurrency and sandbox resources for your CPU/RAM capacity. `actor_rollout_ref.rollout.max_model_len` is the shared prompt-plus-completion context budget; the default is 262144 tokens. Smaller models/resources may require explicit overrides.

Checkpoints are saved to `EXP_ROOT/checkpoints/{training,huggingface}`; logs, complete rollout JSON, token audits, and automatic verification/action statistics go to `EXP_ROOT/logs/<run_time>`. W&B runs offline by default. Use a separate `EXP_ROOT` for each new run. Resume with the same profile, model, data, and resource settings:

```bash
bash launch/train.sh --resume-exp-root "$EXP_ROOT" \
  --resume-log-dir '<path/to/original_run_log>' trainer.resume_mode=auto \
  trainer.nnodes=2 trainer.n_gpus_per_node=8 \
  rollout.nnodes=2 rollout.n_gpus_per_node=8 \
  ++swe.minisandbox.local_root=/sandbox-local
```

### Method controls

Append Hydra overrides to the launcher. Use `++swe.KEY=value` for SWE-specific options; existing `rllm.*` keys use `key=value`. In the table, `M` means `rllm.stepwise_advantage.milestone`, and `D` means `rllm.dynamic_sampling`.

| Control | Meaning / current launcher defaults |
|---|---|
| `M.enable`, `M.verification_enable`, `M.format_enable` | Enable milestone, verification, and tool-format components; all `true`. |
| `M.navigation_enable`, `M.navigation_weight` | Enable and weight relevant-file exposure potential; default `false`, `0`. Search/read scores are `M.navigation_search_score=0.2`, `M.navigation_read_score=1.0`. |
| `M.verification_weight`, `M.beta_any`, `M.beta_frac` | Bug repair uses normalized F2P progress with any/fractional P2P regression penalties: `0.2`, `0.1`, `0.5`. DeNovo uses a baseline-relative, unnormalized passing-test count weighted by `0.1`. |
| `M.verification_reward_clip_lower/upper` | Clip pass-count reward increments before weighting; `-6` / `3`. |
| `M.backward_credit_lambda/gamma` | Backward credit strength/discount: `0.2` / `0.9`. |
| `rllm.stepwise_advantage.outcome_weight/tool_call_weight` | Final-outcome and format weights: `1.0` / `0.25`; format reward is `0` for valid calls and `-1` otherwise. |
| `rllm.rollout.n`, `rllm.async_training.mini_batch_size` | Rollouts per task / task groups per update: bug repair `16 / 12`; DeNovo `8 / 24`. |
| `D.enable`, `D.outcome_mode` | **Outcome-driven asymmetric dynamic sampling** filters using final outcomes, independently of process rewards. Bug repair: `reward_uniform`; DeNovo: `verifier_pass_count`, easy/hard mean pass-rate thresholds `0.97 / 0.03`. |
| `D.max_easy_rejections/max_hard_rejections` | Retire easy tasks sooner and give hard tasks more attempts: bug repair `1 / 2`; DeNovo `1 / 3`. Nonuniform useful groups proceed to training. |
| `D.async_step_sampling_multiplier` | Speculative over-sampling factor: bug repair `1.5`, DeNovo `2.0`; unused/filtered work is finalized or cancelled through the normal lifecycle. |
| `swe.limit_termination_outcome_mode`, `swe.limit_termination_success_reward` | **Reward at turn or context limits:** bug repair uses `discount_success` + `auto`. A successful capped rollout receives first-full-verification turn / total turns; if unavailable, fallback is `0.6`. Numeric values set a fixed discount. Final correctness remains unchanged. DeNovo uses `verifier_outcome`, preserving fractional acceptance-test scores. |
| `swe.denovo_background_finalize_enable` | `true`: overlap the primary verifier and shadow probes, with a batch finalization barrier before training consumes rewards. |
| `swe.denovo_shadow_sandbox_count`, `swe.denovo_shadow_probe_merge_enable/max_steps` | DeNovo uses `4` shadows and merges consecutive probes with the same changed-file set, up to `5` steps. |
| `swe.shadow_finalize_stall_timeout` | Shadow no-progress timeout: bug repair `3600`, DeNovo `1800` seconds; preserve completed partial probes when interrupted. |
| `swe.repository_difficulty_max_turns_enable` | DeNovo adaptive turn limits: low/medium/high = `200 / 350 / 500`. Adaptive shadow resources are controlled separately by `swe.repository_difficulty_shadow_resources_enable` (default `false`). |
| `swe.primary_sandbox_cpus/memory_mb`, `swe.shadow_sandbox_cpus/memory_mb` | Independent primary/shadow resources: bug repair `2 CPU / 8192 MiB`, DeNovo `4 CPU / 16384 MiB`. |
| `rllm.workflow.n_parallel_tasks`, `rllm.workflow.rollout_startup_window`, `rllm.workflow.denovo_shadow_overlap_budget_multiplier` | Active-rollout, startup, and detached-shadow budget controls. DeNovo overlap multiplier `2.0` bounds aggregate shadow count/CPU/RAM; it does not remove admission limits. |

For example, enable navigation with `rllm.stepwise_advantage.milestone.navigation_enable=true rllm.stepwise_advantage.milestone.navigation_weight=0.03`. Disabling verification also disables shadow provisioning. Keep matching data, resource, and reward configurations when resuming.

## 4. Evaluate

```bash
TASK_PROFILE=bug_repair bash launch/eval.sh \
  ++swe.minisandbox.local_root=/sandbox-local ++eval.pass_n=1
# Other evaluation profiles: TASK_PROFILE=nl2repo or TASK_PROFILE=doc2repo.
# SWE-bench Pro: append both overrides:
# ++eval.benchmark_profile=swebench_pro_public ++eval.dataset_name=swebench_pro_public
```

The default evaluation allocation is one 8-GPU node with tensor parallelism 1. Set `rollout.nnodes`, `rollout.n_gpus_per_node`, and `actor_rollout_ref.rollout.tensor_model_parallel_size` for your model and allocation. `node_parallel` keeps each replica on one node; use `++eval.scheduling_mode=sequential` for a replica spanning nodes.

Evaluation uses the base model at `MODEL_PATH` and discovers exported HF checkpoints under `EXP_ROOT/checkpoints/huggingface`. `++eval.evaluate_base=false` skips the base model; `++eval.checkpoint_interval=5` selects checkpoint steps; `++eval.pass_n=8` enables eight attempts per task. Use `++eval.max_tasks=10` for a small smoke run, or `++eval.task_ids_file=<path>` for a non-empty JSON list of task IDs.

Results and per-attempt rollouts are stored under `EXP_ROOT/evaluation`. Re-running the same configuration fills missing attempts; completed attempts are reused. Aggregation covers replicas, pass@k, and fractional score@k for repository generation. Infrastructure failures remain distinguishable from verifier failures. Evaluation uses final verifier outcomes and does not provision reward-only shadows.

### K8s adapter contract

The factory receives `name`, `image`, `cpus`, `memory_mb`, `create_timeout`, `working_dir`, `allow_internet`, and optional `storage_mb` / `env`, and returns a ready sandbox. Implement the methods in [k8s.py](rllm/sandbox/backends/k8s.py):

- `exec(command, timeout=None, user=None)` and `exec_readonly(...)` return captured text and honor the execution identity and timeout.
- `read_files(paths, timeout, user, max_bytes)` returns per-path records: `present` with UTF-8 `content`, byte `size`, and `sha256`; or `missing`, `invalid`, `unreadable`. The exact bounded format is in [file_snapshot.py](rllm/sandbox/file_snapshot.py).
- `upload_file`, `upload_dir`, `is_alive`, `cancel_pending_exec`, `pending_exec_count`, and `close` provide file transfer, liveness, cancellation, and confirmed cleanup.

Create isolated writable primary/shadow filesystems, enforce requested resource/network limits, and preserve the image's task environment. Clean up partial creation failures. Never resubmit an uncertain mutating command; raise `RolloutInfrastructureError` with an auditable reason. Pass service credentials through your adapter's environment or secret store. See [protocol.py](rllm/sandbox/protocol.py) for the base interface. Set `SWE_MILE_VERIFIER_PROXY_URL` only if your remote verifier needs an HTTP proxy.

## Code map and license

`launch/`: entrypoints · `rllm/data/`: materialization · `rllm/harnesses/`: Codeflow and shadow probes · `rllm/rewards/milestone_reward.py`: potentials/rewards · `rllm/trainer/`: sampling and VERL · `rllm/sandbox/`: backends · `rllm-model-gateway/`: routing and trace capture · `rllm/metrics/`: automatic run statistics.

[Apache-2.0](LICENSE). Models, datasets, task images, and upstream components remain subject to their own licenses and access conditions.

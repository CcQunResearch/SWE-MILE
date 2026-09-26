# SWE-MILE

中文 | [English](README.md)

**Asynchronous Potential-Induced Milestone Credit Assignment for Long-Horizon Software Engineering Agents.** SWE-MILE 将最终 verifier outcome 与代码导航、验证势能变化、向后信用分配及工具格式奖励结合。Codeflow 在独立的 primary/shadow sandbox 中采样，通过仓库内的 model gateway 接入 VERL，进行 fully-async RLOO/PPO 训练。

本项目基于 [rLLM](https://github.com/rllm-org/rllm) 和 [VERL](https://github.com/verl-project/verl)，保留 `rllm` Python 命名空间。本发行版包含 SWE 流程及必要运行依赖，并保留上游许可证声明。

## 1. 环境

使用 Linux、NVIDIA GPU 和提供的镜像。Dockerfile 固定 Qwen3.5 运行栈（CUDA 12.9、PyTorch 2.10.0、VERL 0.8.0、vLLM 0.17.0、Transformers 5.8.0），并应用仓库内的运行补丁。

```bash
docker build -t swe-mile:local .
docker run --gpus all --ipc=host --network=host --privileged --cgroupns=host \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
  -v '<path/to/exp_root>:/experiments' \
  -v '<path/to/models>:/models:ro' \
  -v '<path/to/local_nvme>:/sandbox-local' \
  -it swe-mile:local
```

进入镜像后，先设置实际路径：

```bash
cd /workspace/SWE-MILE
export EXP_ROOT=/experiments/run1
export DATA_ROOT=/experiments/data
export MODEL_PATH='/models/<model_directory>'
```

替换所有 `<...>` 占位符。`MODEL_PATH` 必须包含 Hugging Face 权重、模型配置和 tokenizer 文件。集群每个节点使用相同镜像、代码、模型路径及共享的 `DATA_ROOT`/`EXP_ROOT`；sandbox 临时目录必须位于节点本地磁盘。在镜像内修改源码后，可执行 `uv pip install --system --no-deps -e ./rllm-model-gateway -e .` 重新安装。

**Sandbox 选择**（`swe.sandbox_backend`）：

- `minisandbox`（默认）：基于 [SWE-MiniSandbox](https://github.com/lblankl/SWE-MiniSandbox)，必要的 OCI 运行扩展已包含在 `rllm/sandbox/minisandbox_runtime`。要求 Linux root、可写的 **cgroup v1**、namespace、overlayfs、libseccomp，以及 Dockerfile 安装的系统工具。为镜像解包和并发任务预留足够本地磁盘；启动检查至少要求 32 GiB 可用空间。容器特权无法将 cgroup v2 宿主机转换为 v1。启动 worker 前须完成下述 OCI 缓存物化。
- `k8s`：通过 `export SWE_MILE_K8S_ADAPTER=your_package.sandbox:create_sandbox` 接入自己的 Kubernetes sandbox 服务，在每个 Ray 节点安装该适配包，并向训练/评测命令追加 `++swe.sandbox_backend=k8s`。接口见[下文](#k8s-适配接口)，仓库不包含集群服务实现。此模式无需本地特权执行和 OCI 缓存。

## 2. 数据物化

`launch/prepare_dataset.sh` 生成任务目录和注册表 Parquet，跳过已完成工作并校验数据契约。默认开启全部七类数据集，可通过下表开关关闭不需要的部分。需要数据访问凭据时，通过 `HF_TOKEN` 提供。默认使用公开镜像；可设置 `SWE_MILE_IMAGE_MAP_FILE`，指向将原始镜像地址映射为自有镜像地址的 JSON 对象。私有 OCI 仓库认证通过 `MINISANDBOX_REGISTRY_AUTH_FILE` 指向 containers-auth JSON 文件。

```bash
# 默认 bug-repair 训练数据 + SWE-bench Verified 评测数据，包含 OCI 缓存。
BUILD_R2EGYM=0 BUILD_SWEBENCH_PRO_PUBLIC=0 BUILD_DENOVOSWE=0 \
BUILD_NL2REPO=0 BUILD_DOC2REPO=0 \
BUILD_SWEREBENCH_V2_PYTHON_MINISANDBOX_CACHE=1 \
BUILD_SWEBENCH_VERIFIED_MINISANDBOX_CACHE=1 \
bash launch/prepare_dataset.sh
```

| 数据集 | 物化开关 | MiniSandbox 缓存开关 |
|---|---|---|
| R2E-Gym | `BUILD_R2EGYM` | `BUILD_R2EGYM_MINISANDBOX_CACHE` |
| SWE-bench Verified | `BUILD_VERIFIED` | `BUILD_SWEBENCH_VERIFIED_MINISANDBOX_CACHE` |
| SWE-bench Pro Public | `BUILD_SWEBENCH_PRO_PUBLIC` | `BUILD_SWEBENCH_PRO_PUBLIC_MINISANDBOX_CACHE` |
| SWE-rebench V2 | `BUILD_SWEREBENCH_V2` | `BUILD_SWEREBENCH_V2_PYTHON_MINISANDBOX_CACHE` 或 `BUILD_SWEREBENCH_V2_PROMATCHED_MINISANDBOX_CACHE` |
| DeNovoSWE | `BUILD_DENOVOSWE` | `BUILD_DENOVOSWE_MINISANDBOX_CACHE` |
| NL2Repo-Bench | `BUILD_NL2REPO` | `BUILD_NL2REPO_MINISANDBOX_CACHE` |
| BeyondSWE Doc2Repo | `BUILD_DOC2REPO` | `BUILD_DOC2REPO_MINISANDBOX_CACHE` |

开关取值为 `0`/`1`，缓存开关默认均为 `0`。DeNovoSWE 用户开启对应物化与缓存开关、关闭不需要的数据集即可。K8s 用户保留缓存开关为 `0`，并确保服务可访问任务镜像。`RLLM_MATERIALIZATION_WORKERS` 控制任务准备并发，`MINISANDBOX_IMAGE_WORKERS` 控制镜像准备并发。R2E-Gym milestone 训练还须设置 `R2EGYM_MATERIALIZE_MILESTONE_METADATA=1`。SWE-rebench 准备过程会生成 Python 和语言匹配视图；训练根据仓库内的任务索引派生默认的 `python-filternorm` 子集。

输出位于 `DATA_ROOT/{tasks,rllm_home,hf_home,minisandbox}`。如需强制使用资格清单，训练时追加 `++swe.eligibility_manifest=<path> ++swe.require_eligibility_manifest=true`；默认不要求提供。

## 3. 训练

训练 profile 为 `TASK_PROFILE=bug_repair`（默认）或 `TASK_PROFILE=denovoswe`。前者选择 SWE-rebench V2 Python filternorm；追加 `++swe.train_dataset=r2egym` 可切换到已准备的 R2E-Gym。后者选择 DeNovoSWE。

```bash
# 已运行的 Ray 集群：16 张训练 GPU + 16 张独立采样 GPU。
export RAY_ADDRESS='<head_ip>:6379'
TASK_PROFILE=bug_repair bash launch/train.sh \
  trainer.nnodes=2 trainer.n_gpus_per_node=8 \
  rollout.nnodes=2 rollout.n_gpus_per_node=8 \
  ++swe.minisandbox.local_root=/sandbox-local
```

新建集群时，在分配到的**每个节点**执行 `launch/cluster.sh`，设置 `RANK=0..N-1`、共同的 `MASTER_ADDR`、`TRAIN_NODES`、`ROLLOUT_NODES` 和 `GPUS_PER_NODE`。`N=TRAIN_NODES+ROLLOUT_NODES`；脚本等待 GPU 就绪后在 rank 0 启动训练。各节点使用相同 profile、路径和参数。该脚本默认分配 4 个训练节点和 4 个采样节点，每节点 8 张 GPU。

默认序列并行度为 16，必须整除**训练** GPU 数量，并与模型 attention heads 兼容。调整配额时，显式设置 `actor_rollout_ref.actor.ulysses_sequence_parallel_size` 和 `actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size`。根据 CPU/内存容量配置采样并发和 sandbox 资源。`actor_rollout_ref.rollout.max_model_len` 统一约束 prompt 与 completion 总上下文，默认 262144 tokens；其他模型或资源规模可能需要显式调整。

Checkpoint 保存在 `EXP_ROOT/checkpoints/{training,huggingface}`；日志、完整 rollout JSON、token 审计及自动验证/动作统计位于 `EXP_ROOT/logs/<run_time>`。W&B 默认离线。每个新实验使用独立 `EXP_ROOT`。续训时保持 profile、模型、数据和资源设置一致：

```bash
bash launch/train.sh --resume-exp-root "$EXP_ROOT" \
  --resume-log-dir '<path/to/original_run_log>' trainer.resume_mode=auto \
  trainer.nnodes=2 trainer.n_gpus_per_node=8 \
  rollout.nnodes=2 rollout.n_gpus_per_node=8 \
  ++swe.minisandbox.local_root=/sandbox-local
```

### 方法参数

在启动命令末尾追加 Hydra 参数。SWE 专用参数写为 `++swe.KEY=value`；已有 `rllm.*` 参数写为 `key=value`。下表中 `M` 表示 `rllm.stepwise_advantage.milestone`，`D` 表示 `rllm.dynamic_sampling`。

| 参数 | 含义 / 当前启动默认值 |
|---|---|
| `M.enable`、`M.verification_enable`、`M.format_enable` | milestone、verification 和工具格式分量的开关，均为 `true`。 |
| `M.navigation_enable`、`M.navigation_weight` | 相关文件暴露势能的开关和权重，默认 `false`、`0`。搜索/读取分值为 `M.navigation_search_score=0.2`、`M.navigation_read_score=1.0`。 |
| `M.verification_weight`、`M.beta_any`、`M.beta_frac` | Bug repair 使用归一化 F2P 进展及 P2P 任意回退/回退比例惩罚，对应 `0.2`、`0.1`、`0.5`。DeNovo 使用相对 baseline 的非归一化通过测试数，权重为 `0.1`。 |
| `M.verification_reward_clip_lower/upper` | pass-count 奖励增量在加权前裁剪至 `-6` / `3`。 |
| `M.backward_credit_lambda/gamma` | 向后信用分配强度/折扣：`0.2` / `0.9`。 |
| `rllm.stepwise_advantage.outcome_weight/tool_call_weight` | 最终 outcome 和格式权重：`1.0` / `0.25`；合法调用格式奖励为 `0`，否则为 `-1`。 |
| `rllm.rollout.n`、`rllm.async_training.mini_batch_size` | 每任务轨迹数 / 每次更新任务组数：bug repair 为 `16 / 12`，DeNovo 为 `8 / 24`。 |
| `D.enable`、`D.outcome_mode` | **Outcome-driven asymmetric dynamic sampling** 仅根据最终 outcome 过滤，独立于过程奖励。Bug repair 使用 `reward_uniform`；DeNovo 使用 `verifier_pass_count`，easy/hard 平均通过率阈值为 `0.97 / 0.03`。 |
| `D.max_easy_rejections/max_hard_rejections` | 更早移除 easy 任务，为 hard 任务保留更多尝试：bug repair 为 `1 / 2`，DeNovo 为 `1 / 3`。具有非一致 outcome 的有效组进入训练。 |
| `D.async_step_sampling_multiplier` | 投机超采倍率：bug repair 为 `1.5`，DeNovo 为 `2.0`；未使用或已过滤工作仍通过正常生命周期完成收尾或取消。 |
| `swe.limit_termination_outcome_mode`、`swe.limit_termination_success_reward` | **Reward at turn or context limits：**bug repair 使用 `discount_success` + `auto`。到达轮次/上下文上限但修复成功时，奖励为首次 verification 达满分的轮次 / 总轮次；信息不可用时回退为 `0.6`。数值配置表示固定折扣。最终 correctness 不变。DeNovo 使用 `verifier_outcome`，保留验收测试的分数型 outcome。 |
| `swe.denovo_background_finalize_enable` | `true`：primary verifier 与 shadow probe 并行，训练消费奖励前执行批次级 finalize barrier。 |
| `swe.denovo_shadow_sandbox_count`、`swe.denovo_shadow_probe_merge_enable/max_steps` | DeNovo 使用 `4` 个 shadow，将修改文件集合相同的连续 probe 合并，每组最多 `5` 步。 |
| `swe.shadow_finalize_stall_timeout` | Shadow 无进展超时：bug repair 为 `3600`、DeNovo 为 `1800` 秒；中断时保留已完成的部分 probe。 |
| `swe.repository_difficulty_max_turns_enable` | DeNovo 难度自适应轮次上限：低/中/高为 `200 / 350 / 500`。Shadow 资源自适应由 `swe.repository_difficulty_shadow_resources_enable` 单独控制，默认 `false`。 |
| `swe.primary_sandbox_cpus/memory_mb`、`swe.shadow_sandbox_cpus/memory_mb` | Primary/shadow 独立资源配置：bug repair 均为 `2 CPU / 8192 MiB`，DeNovo 均为 `4 CPU / 16384 MiB`。 |
| `rllm.workflow.n_parallel_tasks`、`rllm.workflow.rollout_startup_window`、`rllm.workflow.denovo_shadow_overlap_budget_multiplier` | 控制活跃 rollout、启动窗口和后台 shadow 预算。DeNovo overlap 倍率为 `2.0`，约束 shadow 总数量、CPU 和内存，不取消准入限制。 |

例如，追加 `rllm.stepwise_advantage.milestone.navigation_enable=true rllm.stepwise_advantage.milestone.navigation_weight=0.03` 可开启导航奖励。关闭 verification 时也不再创建 shadow。续训须保持数据、资源和奖励设置匹配。

## 4. 评测

```bash
TASK_PROFILE=bug_repair bash launch/eval.sh \
  ++swe.minisandbox.local_root=/sandbox-local ++eval.pass_n=1
# 其他评测 profile：TASK_PROFILE=nl2repo 或 TASK_PROFILE=doc2repo。
# SWE-bench Pro：同时追加以下两个参数：
# ++eval.benchmark_profile=swebench_pro_public ++eval.dataset_name=swebench_pro_public
```

默认评测资源为一个 8-GPU 节点，tensor parallelism 为 1。根据模型与资源设置 `rollout.nnodes`、`rollout.n_gpus_per_node` 和 `actor_rollout_ref.rollout.tensor_model_parallel_size`。`node_parallel` 将每个 replica 限制在单节点；跨节点 replica 使用 `++eval.scheduling_mode=sequential`。

评测使用 `MODEL_PATH` 中的初始模型，并发现 `EXP_ROOT/checkpoints/huggingface` 下导出的 HF checkpoint。`++eval.evaluate_base=false` 跳过初始模型；`++eval.checkpoint_interval=5` 选择 checkpoint 步数间隔；`++eval.pass_n=8` 为每个任务生成八次尝试。小规模检查可用 `++eval.max_tasks=10`，指定任务可用 `++eval.task_ids_file=<path>`，文件内容为非空 JSON 任务 ID 列表。

结果及各次尝试的 rollout 保存到 `EXP_ROOT/evaluation`。相同配置再次运行时复用已完成尝试、补齐缺失尝试。跨 replica 汇总 pass@k，以及仓库生成任务的 fractional score@k；基础设施失败与 verifier 失败分别记录。评测仅使用最终 verifier outcome，不创建仅用于过程奖励的 shadow。

### K8s 适配接口

工厂接收 `name`、`image`、`cpus`、`memory_mb`、`create_timeout`、`working_dir`、`allow_internet`，以及可选的 `storage_mb` / `env`，返回已经就绪的 sandbox。实现 [k8s.py](rllm/sandbox/backends/k8s.py) 要求的方法：

- `exec(command, timeout=None, user=None)` 与 `exec_readonly(...)` 返回捕获的文本，并遵守执行用户和超时设置。
- `read_files(paths, timeout, user, max_bytes)` 返回逐路径记录：`present` 包含 UTF-8 `content`、字节 `size` 和 `sha256`；其他状态为 `missing`、`invalid`、`unreadable`。有界读取的完整格式见 [file_snapshot.py](rllm/sandbox/file_snapshot.py)。
- `upload_file`、`upload_dir`、`is_alive`、`cancel_pending_exec`、`pending_exec_count`、`close` 分别提供文件传输、存活检查、取消和确认清理。

Primary/shadow 必须使用独立可写文件系统，落实资源/网络限制并保留镜像任务环境。创建中途失败须清理已创建资源；不能重新提交完成状态不明的修改操作，应抛出带可审计原因的 `RolloutInfrastructureError`。服务凭据通过适配器环境变量或密钥存储提供。基础接口见 [protocol.py](rllm/sandbox/protocol.py)。仅在远程 verifier 需要 HTTP 代理时设置 `SWE_MILE_VERIFIER_PROXY_URL`。

## 代码位置与许可证

`launch/`：启动入口 · `rllm/data/`：物化 · `rllm/harnesses/`：Codeflow 与 shadow probe · `rllm/rewards/milestone_reward.py`：势能与奖励 · `rllm/trainer/`：采样与 VERL · `rllm/sandbox/`：后端 · `rllm-model-gateway/`：路由和轨迹捕获 · `rllm/metrics/`：自动运行统计。

[Apache-2.0](LICENSE)。模型、数据集、任务镜像和上游组件仍须遵守各自的许可证与访问条件。

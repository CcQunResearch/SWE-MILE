#!/usr/bin/env bash
set -euo pipefail
SWE_MILE_REQUIRE_MODEL=1
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

TASK_PROFILE="${TASK_PROFILE:-bug_repair}"
EVAL_PROFILE_ARGS=()
case "${TASK_PROFILE}" in
    nl2repo | doc2repo)
        DEFAULT_DATA_ROOT="${DATA_ROOT}"
        DEFAULT_EXP_ROOT="${EXP_ROOT}"
        if [[ "${TASK_PROFILE}" == "nl2repo" ]]; then
            DEFAULT_EVALUATION_SUFFIX="evaluation"
            BENCHMARK_PROFILE="nl2repo"
            DATASET_NAME="nl2repo-bench"
        else
            DEFAULT_EVALUATION_SUFFIX="evaluation"
            BENCHMARK_PROFILE="doc2repo"
            DATASET_NAME="beyondswe-doc2repo"
        fi
        EVAL_PROFILE_ARGS=(
            "+swe.sandbox_backend=minisandbox"
            "+eval.benchmark_profile=${BENCHMARK_PROFILE}"
            "+eval.dataset_name=${DATASET_NAME}"
            "+eval.dataset_split=test"
            "+swe.codeflow_tool_mode=structured"
            "+swe.codeflow_tool_restricted_mode=[]"
            "+swe.max_turns=500"
            "+swe.command_timeout=1200"
            "+swe.verifier_timeout=3600"
            "+swe.limit_termination_success_reward=1.0"
            "+swe.limit_termination_outcome_mode=verifier_outcome"
            "+swe.sandbox_cpus=4"
            "+swe.sandbox_memory_mb=16384"
            "rllm.rollout.val.max_tokens=16384"
            "rllm.workflow.n_parallel_tasks=110"
            "actor_rollout_ref.rollout.max_num_batched_tokens=65536"
            "actor_rollout_ref.rollout.max_model_len=262144"
        )
        ;;
    bug_repair)
        DEFAULT_DATA_ROOT="${DATA_ROOT}"
        DEFAULT_EXP_ROOT="${EXP_ROOT}"
        DEFAULT_EVALUATION_SUFFIX="evaluation"
        EVAL_PROFILE_ARGS=(
            "+swe.sandbox_backend=minisandbox"
            "+eval.benchmark_profile=swebench_verified" # swebench_pro_public | swebench_verified
            "+eval.dataset_name=swe_bench_verified" # swebench_pro_public | swe_bench_verified
            "+eval.dataset_split=test"
            "+swe.codeflow_tool_mode=structured" # structured | bash_only
            "+swe.codeflow_tool_restricted_mode=[]"
            "+swe.max_turns=200"
            "+swe.command_timeout=1200"
            "+swe.verifier_timeout=3600"
            "+swe.limit_termination_success_reward=1.0"
            "+swe.limit_termination_outcome_mode=discount_success"
            "+swe.sandbox_cpus=4"
            "+swe.sandbox_memory_mb=16384"
            "rllm.rollout.val.max_tokens=8192"
            "rllm.workflow.n_parallel_tasks=96"
            "rllm.workflow.rollout_startup_window=160"
            "actor_rollout_ref.rollout.max_num_batched_tokens=65536"
            "actor_rollout_ref.rollout.max_model_len=262144"
        )
        ;;
    "")
        echo "TASK_PROFILE is required; expected nl2repo, doc2repo, or bug_repair" >&2
        exit 2
        ;;
    *)
        echo "unsupported TASK_PROFILE=${TASK_PROFILE}; expected nl2repo, doc2repo, or bug_repair" >&2
        exit 2
        ;;
esac

MODEL_PATH="${MODEL_PATH}"
export DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
EXP_ROOT="${EXP_ROOT:-${DEFAULT_EXP_ROOT}}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${EXP_ROOT}/${DEFAULT_EVALUATION_SUFFIX}}"

export HYDRA_FULL_ERROR=1
export RLLM_HOME="${RLLM_HOME:-${DATA_ROOT}/rllm_home}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset RLLM_ROLLOUT_LOG_PATH RLLM_ROLLOUT_LOG_START_INDEX

RUN_ID="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
RUN_LOG_PATH="${EVALUATION_ROOT}/runs/${RUN_ID}/eval.log"
mkdir -p "${EVALUATION_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "$(dirname "${RUN_LOG_PATH}")"

exec > >(tee -a "${RUN_LOG_PATH}") 2>&1

# eval.scheduling_mode: sequential | node_parallel
exec python3 -u -m launch.eval \
    +eval.scheduling_mode=node_parallel \
    +eval.base_model_path="${MODEL_PATH}" \
    +eval.evaluate_base=true \
    +eval.checkpoint_stride=1 \
    +eval.checkpoint_interval=5 \
    +eval.pass_n=1 \
    +eval.temperature=0.6 \
    +eval.top_p=0.95 \
    +eval.top_k=20 \
    +eval.experiment_root="${EXP_ROOT}" \
    +eval.output_root="${EVALUATION_ROOT}" \
    +eval.run_id="${RUN_ID}" \
    +eval.console_log_path="${RUN_LOG_PATH}" \
    +eval.gpu_release_timeout=300 \
    +eval.gpu_quiescence_seconds=30 \
    +eval.server_init_max_attempts=3 \
    +eval.server_init_retry_quiescence_seconds=15 \
    +eval.standalone_master_port_base=20000 \
    +eval.standalone_master_port_span=128 \
    +eval.max_tasks=null \
    +eval.task_ids_file=null \
    rllm/backend=verl \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +model.name="${MODEL_PATH}" \
    data.trust_remote_code=True \
    rllm.agent.name=codeflow \
    rllm.gateway.store=memory \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.routing.mode=group_striped_adaptive \
    rllm.gateway.renderer_family=qwen3.5 \
    rllm.data.dynamic_sequence_budget=true \
    rllm.rollout.n_val=1 \
    rllm.rollout.val.temperature=0.6 \
    rllm.rollout.val.top_p=0.95 \
    +rllm.rollout.val.top_k=20 \
    rllm.workflow.retry_limit=3 \
    rllm.workflow.raise_on_error=False \
    rllm.workflow.warm_queue_size=0 \
    rllm.async_training.enable=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enforce_eager=false \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    +swe.protocol=native_tool_call \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=qwen3 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen3_coder \
    "${EVAL_PROFILE_ARGS[@]}" \
    "$@"

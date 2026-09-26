#!/usr/bin/env bash
set -euo pipefail
SWE_MILE_REQUIRE_MODEL=1
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

TASK_PROFILE="${TASK_PROFILE:-bug_repair}"
TRAIN_PROFILE_ARGS=()
case "${TASK_PROFILE}" in
    denovoswe)
        DEFAULT_DATA_ROOT="${DATA_ROOT}"
        DEFAULT_EXP_ROOT="${EXP_ROOT}"
        TRAIN_PROFILE_ARGS=(
            "+swe.sandbox_backend=minisandbox" # explicit CLI override may select minisandbox
            "+swe.train_dataset=denovoswe"
            "+swe.train_split=train"
            "+swe.val_dataset=denovoswe"
            "+swe.val_split=train"
            "+swe.require_eligibility_manifest=false"
            "+swe.codeflow_tool_mode=structured" # structured | bash_only
            "+swe.codeflow_tool_restricted_mode=[]"
            "+swe.max_turns=500"
            "+swe.command_timeout=1200"
            "+swe.shadow_finalize_stall_timeout=1800"
            "+swe.denovo_background_finalize_enable=true"
            "+swe.denovo_shadow_sandbox_count=4"
            "+swe.denovo_shadow_probe_merge_enable=true"
            "+swe.denovo_shadow_probe_merge_max_steps=5"
            "+swe.verifier_timeout_is_failure=true"
            "+swe.limit_termination_success_reward=auto"
            "+swe.limit_termination_outcome_mode=verifier_outcome"
            "+swe.repository_difficulty_max_turns_enable=true"
            "+swe.repository_difficulty_low_max_turns=200"
            "+swe.repository_difficulty_medium_max_turns=350"
            "+swe.repository_difficulty_high_max_turns=500"
            "+swe.repository_difficulty_shadow_resources_enable=false"
            "+swe.repository_difficulty_low_shadow_cpus=2"
            "+swe.repository_difficulty_low_shadow_memory_mb=8192"
            "+swe.repository_difficulty_medium_shadow_cpus=4"
            "+swe.repository_difficulty_medium_shadow_memory_mb=16384"
            "+swe.repository_difficulty_high_shadow_cpus=8"
            "+swe.repository_difficulty_high_shadow_memory_mb=32768"
            "+swe.primary_sandbox_cpus=4"
            "+swe.primary_sandbox_memory_mb=16384"
            "+swe.shadow_sandbox_cpus=4"
            "+swe.shadow_sandbox_memory_mb=16384"
            "rllm.rollout.train.max_tokens=16384"
            "rllm.rollout.val.max_tokens=16384"
            "rllm.rollout.n=8"
            "rllm.dynamic_sampling.outcome_mode=verifier_pass_count"
            "rllm.dynamic_sampling.max_easy_rejections=1"
            "rllm.dynamic_sampling.max_hard_rejections=3"
            "rllm.dynamic_sampling.easy_pass_rate_threshold=0.97"
            "rllm.dynamic_sampling.hard_pass_rate_threshold=0.03"
            "rllm.dynamic_sampling.async_step_sampling_multiplier=2.0"
            "rllm.dynamic_sampling.cancel_wait_timeout_seconds=120"
            "rllm.agent.trajectory_timeout=14400"
            "rllm.workflow.n_parallel_tasks=640"
            "rllm.workflow.rollout_startup_window=48"
            "rllm.workflow.denovo_shadow_overlap_budget_multiplier=2.0"
            "rllm.workflow.cancelled_teardown_timeout_seconds=60"
            "rllm.workflow.teardown_executor_workers=32"
            "rllm.stepwise_advantage.milestone.enable=true"
            "rllm.stepwise_advantage.milestone.verification_enable=true"
            "rllm.stepwise_advantage.milestone.format_enable=true"
            "rllm.stepwise_advantage.milestone.beta_any=0.1"
            "rllm.stepwise_advantage.milestone.beta_frac=0.5"
            "rllm.stepwise_advantage.milestone.verification_weight=0.1"
            "rllm.stepwise_advantage.milestone.navigation_enable=false"
            "rllm.stepwise_advantage.milestone.navigation_weight=0.0"
            "rllm.stepwise_advantage.milestone.navigation_search_score=0.2"
            "rllm.stepwise_advantage.milestone.navigation_read_score=1.0"
            "rllm.stepwise_advantage.milestone.backward_credit_lambda=0.2"
            "rllm.stepwise_advantage.milestone.backward_credit_gamma=0.9"
            "rllm.stepwise_advantage.milestone.verification_reward_clip_lower=-6.0"
            "rllm.stepwise_advantage.milestone.verification_reward_clip_upper=3.0"
            "rllm.async_training.mini_batch_size=24"
            "rllm.async_training.fwd_bwd_group_size=24"
            "actor_rollout_ref.actor.ppo_mini_batch_size=24"
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size=16"
            "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=16"
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384"
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384"
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384"
            "actor_rollout_ref.rollout.max_num_batched_tokens=65536"
            "actor_rollout_ref.rollout.max_model_len=262144"
            "rllm.data.seed=2026"
            "rllm.trainer.save_freq=5"
            "rllm.async_training.staleness_threshold=0.67"
        )
        ;;
    bug_repair)
        DEFAULT_DATA_ROOT="${DATA_ROOT}"
        DEFAULT_EXP_ROOT="${EXP_ROOT}"
        TRAIN_PROFILE_ARGS=(
            "+swe.sandbox_backend=minisandbox"
            "+swe.train_dataset=swe-rebench-v2-filtered-verified-python-filternorm"
            "+swe.train_split=train"
            "+swe.require_eligibility_manifest=false"
            "+swe.codeflow_tool_mode=structured" # structured | bash_only
            "+swe.codeflow_tool_restricted_mode=[]"
            "+swe.max_turns=200"
            "+swe.command_timeout=1200"
            "+swe.shadow_finalize_stall_timeout=3600"
            "+swe.bug_repair_verification_potential_mode=normalized" # normalized | pass_count
            "+swe.verifier_timeout_is_failure=true"
            "+swe.shadow_partition_timeout_recovery=true"
            "+swe.shadow_partition_recovery_command_timeout=300"
            "+swe.shadow_partition_recovery_max_commands=6"
            "+swe.limit_termination_success_reward=auto"
            "+swe.limit_termination_outcome_mode=discount_success"
            "+swe.primary_sandbox_cpus=2"
            "+swe.primary_sandbox_memory_mb=8192"
            "+swe.shadow_sandbox_cpus=2"
            "+swe.shadow_sandbox_memory_mb=8192"
            "rllm.rollout.train.max_tokens=8192"
            "rllm.rollout.val.max_tokens=8192"
            "rllm.rollout.n=16"
            "rllm.dynamic_sampling.outcome_mode=reward_uniform"
            "rllm.dynamic_sampling.max_easy_rejections=1"
            "rllm.dynamic_sampling.max_hard_rejections=2"
            "rllm.dynamic_sampling.async_step_sampling_multiplier=1.5"
            "rllm.dynamic_sampling.cancel_wait_timeout_seconds=120"
            "rllm.workflow.n_parallel_tasks=288"
            "rllm.workflow.rollout_startup_window=120"
            "rllm.workflow.cancelled_teardown_timeout_seconds=60"
            "rllm.stepwise_advantage.milestone.enable=true"
            "rllm.stepwise_advantage.milestone.verification_enable=true"
            "rllm.stepwise_advantage.milestone.format_enable=true"
            "rllm.stepwise_advantage.milestone.beta_any=0.1"
            "rllm.stepwise_advantage.milestone.beta_frac=0.5"
            "rllm.stepwise_advantage.milestone.verification_weight=0.2"
            "rllm.stepwise_advantage.milestone.navigation_enable=true"
            "rllm.stepwise_advantage.milestone.navigation_weight=0.05"
            "rllm.stepwise_advantage.milestone.navigation_search_score=0.2"
            "rllm.stepwise_advantage.milestone.navigation_read_score=1.0"
            "rllm.stepwise_advantage.milestone.backward_credit_lambda=0.2"
            "rllm.stepwise_advantage.milestone.backward_credit_gamma=0.9"
            "rllm.stepwise_advantage.milestone.verification_reward_clip_lower=-6.0"
            "rllm.stepwise_advantage.milestone.verification_reward_clip_upper=3.0"
            "rllm.async_training.mini_batch_size=12"
            "rllm.async_training.fwd_bwd_group_size=12"
            "actor_rollout_ref.actor.ppo_mini_batch_size=12"
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size=16"
            "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=16"
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384"
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384"
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384"
            "actor_rollout_ref.rollout.max_num_batched_tokens=65536"
            "actor_rollout_ref.rollout.max_model_len=262144"
            "rllm.data.seed=42"
            "rllm.trainer.save_freq=5"
            "rllm.async_training.staleness_threshold=0.67"
        )
        ;;
    "")
        echo "TASK_PROFILE is required; expected denovoswe or bug_repair" >&2
        exit 2
        ;;
    *)
        echo "unsupported TASK_PROFILE=${TASK_PROFILE}; expected denovoswe or bug_repair" >&2
        exit 2
        ;;
esac
export TASK_PROFILE

MODEL_PATH="${MODEL_PATH}"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
EXP_ROOT="${EXP_ROOT:-${DEFAULT_EXP_ROOT}}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"
RUN_TS="$(TZ=Asia/Shanghai date +%Y-%m-%d-%H-%M-%S)"

RESUME_EXP_ROOT=""
RESUME_LOG_DIR=""
RESUME_MODE="disable"
RESUME_FROM_PATH=""
HYDRA_ARGS=()
while (($#)); do
    case "$1" in
        --resume-exp-root)
            RESUME_EXP_ROOT="$2"
            shift 2
            ;;
        --resume-exp-root=*)
            RESUME_EXP_ROOT="${1#*=}"
            shift
            ;;
        --resume-log-dir)
            RESUME_LOG_DIR="$2"
            shift 2
            ;;
        --resume-log-dir=*)
            RESUME_LOG_DIR="${1#*=}"
            shift
            ;;
        trainer.resume_mode=*)
            RESUME_MODE="${1#*=}"
            HYDRA_ARGS+=("$1")
            shift
            ;;
        trainer.resume_from_path=*)
            RESUME_FROM_PATH="${1#*=}"
            HYDRA_ARGS+=("$1")
            shift
            ;;
        *)
            HYDRA_ARGS+=("$1")
            shift
            ;;
    esac
done

RESUME_ENABLED=0
RUN_LOG_DIR="${LOG_DIR}/${RUN_TS}"
if [[ "${RESUME_MODE}" == "auto" || "${RESUME_MODE}" == "resume_path" ]]; then
    RESUME_ENABLED=1
    EXP_ROOT="${RESUME_EXP_ROOT}"
    RUN_LOG_DIR="${RESUME_LOG_DIR}"
fi
EXPERIMENT_BASENAME="$(basename "${EXP_ROOT%/}")"

if ((RESUME_ENABLED)); then
    SAVE_PATH="${EXP_ROOT}/checkpoints"
    TRAINING_SAVE_PATH="${SAVE_PATH}/training"
    HF_SAVE_PATH="${SAVE_PATH}/huggingface"
else
    SAVE_PATH="${SAVE_PATH:-${EXP_ROOT}/checkpoints}"
    TRAINING_SAVE_PATH="${TRAINING_SAVE_PATH:-${SAVE_PATH}/training}"
    HF_SAVE_PATH="${HF_SAVE_PATH:-${SAVE_PATH}/huggingface}"
fi
ROLLOUT_LOG_PATH="${RUN_LOG_DIR}/rollout"
VERIFICATION_STATS_ENABLE="${RLLM_VERIFICATION_STATS_ENABLE:-1}"
VERIFICATION_STATS_EVERY="${RLLM_VERIFICATION_STATS_EVERY:-100}"
VERIFICATION_STATS_WORKERS="${RLLM_VERIFICATION_STATS_WORKERS:-128}"
VERIFICATION_STATS_POLL_SECONDS="${RLLM_VERIFICATION_STATS_POLL_SECONDS:-5}"
VERIFICATION_STATS_CACHE_PATH="${RUN_LOG_DIR}/codeflow_verification_potential.cache.json"
VERIFICATION_STATS_OUTPUT_PATH="${RUN_LOG_DIR}/codeflow_verification_potential.statistics"
VERIFICATION_STATS_MONITOR_LOG_PATH="${RUN_LOG_DIR}/codeflow_verification_potential.monitor.log"
VERIFICATION_STATS_STOP_PATH="${RUN_LOG_DIR}/.codeflow_verification_potential.stop.${RUN_TS}.$$"
ACTION_STATS_ENABLE="${RLLM_ACTION_STATS_ENABLE:-1}"
ACTION_STATS_EVERY="${RLLM_ACTION_STATS_EVERY:-100}"
ACTION_STATS_WORKERS="${RLLM_ACTION_STATS_WORKERS:-12}"
ACTION_STATS_POLL_SECONDS="${RLLM_ACTION_STATS_POLL_SECONDS:-5}"
ACTION_STATS_CACHE_PATH="${RUN_LOG_DIR}/action_distribution.cache.json"
ACTION_STATS_OUTPUT_PATH="${RUN_LOG_DIR}/action_distribution.statistics"
ACTION_STATS_MONITOR_LOG_PATH="${RUN_LOG_DIR}/action_distribution.monitor.log"
ACTION_STATS_STOP_PATH="${RUN_LOG_DIR}/.action_distribution.stop.${RUN_TS}.$$"
SWE_LAUNCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ((RESUME_ENABLED)); then
    TRAIN_LOG_PATH="${RUN_LOG_DIR}/train.resume.${RUN_TS}.log"
    METRICS_LOG_PATH="${RUN_LOG_DIR}/metrics.resume.${RUN_TS}.jsonl"
    TRAINING_PARAMS_PATH="${RUN_LOG_DIR}/training_params.resume.${RUN_TS}.json"
    WANDB_RUN_DIR="${RUN_LOG_DIR}/wandb.resume.${RUN_TS}"
    WANDB_REPLAY_HISTORY_PATH="${WANDB_RUN_DIR}/history.before_resume.jsonl"
    EXPERIMENT_NAME="${EXPERIMENT_BASENAME}-resume-${RUN_TS}"
else
    TRAIN_LOG_PATH="${RUN_LOG_DIR}/train.log"
    METRICS_LOG_PATH="${RUN_LOG_DIR}/metrics.jsonl"
    TRAINING_PARAMS_PATH="${RUN_LOG_DIR}/training_params.json"
    WANDB_RUN_DIR="${RUN_LOG_DIR}"
    WANDB_REPLAY_HISTORY_PATH=""
    EXPERIMENT_NAME="${EXPERIMENT_BASENAME}-${RUN_TS}"
fi

export HYDRA_FULL_ERROR=1
export DATA_ROOT
export RLLM_HOME="${RLLM_HOME:-${DATA_ROOT}/rllm_home}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export VERL_FILE_LOGGER_PATH="${METRICS_LOG_PATH}"
export RLLM_TRAINING_PARAMS_PATH="${TRAINING_PARAMS_PATH}"
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_RUN_DIR}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RLLM_ROLLOUT_LOG_START_INDEX=0

mkdir -p \
    "${EXP_ROOT}" \
    "${SAVE_PATH}" \
    "${TRAINING_SAVE_PATH}" \
    "${HF_SAVE_PATH}" \
    "${RUN_LOG_DIR}" \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}"

export EXP_ROOT

if ((!RESUME_ENABLED)); then
    mkdir -p "${ROLLOUT_LOG_PATH}"
fi

exec > >(tee "${TRAIN_LOG_PATH}") 2>&1

RESUME_OVERRIDES=()
if ((RESUME_ENABLED)); then
    exec {RESUME_LOCK_FD}>"${RUN_LOG_DIR}/.resume.lock"
    flock -n "${RESUME_LOCK_FD}"

    if [[ "${RESUME_MODE}" == "auto" ]]; then
        LATEST_MARKER="${TRAINING_SAVE_PATH}/latest_checkpointed_iteration.txt"
        RESUME_STEP="$(tr -d '[:space:]' < "${LATEST_MARKER}")"
        RESUME_CHECKPOINT_PATH="${TRAINING_SAVE_PATH}/global_step_${RESUME_STEP}"
    else
        RESUME_CHECKPOINT_PATH="${RESUME_FROM_PATH%/}"
        RESUME_STEP="${RESUME_CHECKPOINT_PATH##*global_step_}"
    fi

    python3 -m launch.resume_run \
        --experiment-root "${EXP_ROOT}" \
        --log-dir "${RUN_LOG_DIR}" \
        --checkpoint-dir "${RESUME_CHECKPOINT_PATH}" \
        --resume-step "${RESUME_STEP}" \
        --run-timestamp "${RUN_TS}" \
        --wandb-destination "${WANDB_RUN_DIR}"

    RESUME_MANIFEST_PATH="${RUN_LOG_DIR}/resume.${RUN_TS}.manifest.json"
    RLLM_ROLLOUT_LOG_START_INDEX="$(
        sed -nE 's/^[[:space:]]*"retained_max_sample_index":[[:space:]]*([0-9]+),?[[:space:]]*$/\1/p' \
            "${RESUME_MANIFEST_PATH}"
    )"
    export RLLM_ROLLOUT_LOG_START_INDEX

    RESUME_OVERRIDES+=(
        "trainer.default_local_dir=${TRAINING_SAVE_PATH}"
        "trainer.resume_mode=${RESUME_MODE}"
        "rllm.trainer.wandb_replay_history_path=${WANDB_REPLAY_HISTORY_PATH}"
        "rllm.trainer.wandb_replay_through_step=${RESUME_STEP}"
    )
    if [[ "${RESUME_MODE}" == "resume_path" ]]; then
        RESUME_OVERRIDES+=("trainer.resume_from_path=${RESUME_CHECKPOINT_PATH}")
    fi
fi

VERIFICATION_STATS_MONITOR_PID=""
VERIFICATION_STATS_FINALIZED=0

finalize_verification_statistics() {
    local original_status=$?
    if ((VERIFICATION_STATS_FINALIZED)); then
        return "${original_status}"
    fi
    VERIFICATION_STATS_FINALIZED=1

    if [[ -n "${VERIFICATION_STATS_MONITOR_PID}" ]]; then
        : >"${VERIFICATION_STATS_STOP_PATH}"
        if ! wait "${VERIFICATION_STATS_MONITOR_PID}"; then
            echo "WARNING: background verification-potential final audit failed; see ${VERIFICATION_STATS_MONITOR_LOG_PATH}" >&2
        fi
        VERIFICATION_STATS_MONITOR_PID=""
    fi

    # Run once more in the launcher process.  This also covers an external
    # termination that prevented the monitor from observing its stop file.
    if [[ "${VERIFICATION_STATS_ENABLE}" != "0" ]] && \
       compgen -G "${ROLLOUT_LOG_PATH}/*.json" >/dev/null; then
        if ! python3 -u "${SWE_MILE_ROOT}/rllm/metrics/verification_statistics.py" \
            "${ROLLOUT_LOG_PATH}" \
            --workers "${VERIFICATION_STATS_WORKERS}" \
            --mode train \
            --cache-file "${VERIFICATION_STATS_CACHE_PATH}" \
            --output-file "${VERIFICATION_STATS_OUTPUT_PATH}" \
            >>"${VERIFICATION_STATS_MONITOR_LOG_PATH}" 2>&1; then
            echo "WARNING: launcher verification-potential final audit failed; see ${VERIFICATION_STATS_MONITOR_LOG_PATH}" >&2
        fi
    fi
    rm -f -- "${VERIFICATION_STATS_STOP_PATH}"
    return "${original_status}"
}


ACTION_STATS_MONITOR_PID=""
ACTION_STATS_FINALIZED=0

finalize_action_statistics() {
    local original_status=$?
    if ((ACTION_STATS_FINALIZED)); then
        return "${original_status}"
    fi
    ACTION_STATS_FINALIZED=1

    if [[ -n "${ACTION_STATS_MONITOR_PID}" ]]; then
        : >"${ACTION_STATS_STOP_PATH}"
        if ! wait "${ACTION_STATS_MONITOR_PID}"; then
            echo "WARNING: background action-distribution final audit failed; see ${ACTION_STATS_MONITOR_LOG_PATH}" >&2
        fi
        ACTION_STATS_MONITOR_PID=""
    fi

    # Run once more in the launcher process.  This also covers an external
    # termination that prevented the monitor from observing its stop file.
    if [[ "${ACTION_STATS_ENABLE}" != "0" ]] && \
       compgen -G "${ROLLOUT_LOG_PATH}/*.json" >/dev/null; then
        if ! python3 -u "${SWE_MILE_ROOT}/rllm/metrics/action_statistics.py" \
            "${ROLLOUT_LOG_PATH}" \
            --workers "${ACTION_STATS_WORKERS}" \
            --mode train \
            --cache-file "${ACTION_STATS_CACHE_PATH}" \
            --output-file "${ACTION_STATS_OUTPUT_PATH}" \
            >>"${ACTION_STATS_MONITOR_LOG_PATH}" 2>&1; then
            echo "WARNING: launcher action-distribution final audit failed; see ${ACTION_STATS_MONITOR_LOG_PATH}" >&2
        fi
    fi
    rm -f -- "${ACTION_STATS_STOP_PATH}"
    return "${original_status}"
}


finalize_training_statistics() {
    local original_status=$?
    finalize_verification_statistics || true
    finalize_action_statistics || true
    return "${original_status}"
}
trap finalize_training_statistics EXIT

if [[ "${VERIFICATION_STATS_ENABLE}" != "0" ]]; then
    if [[ ! "${VERIFICATION_STATS_EVERY}" =~ ^[1-9][0-9]*$ ]]; then
        echo "RLLM_VERIFICATION_STATS_EVERY must be an integer >= 1" >&2
        exit 2
    fi
    if [[ ! "${VERIFICATION_STATS_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "RLLM_VERIFICATION_STATS_WORKERS must be an integer >= 1" >&2
        exit 2
    fi
    if [[ ! "${VERIFICATION_STATS_POLL_SECONDS}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
       [[ -z "${VERIFICATION_STATS_POLL_SECONDS//[0.]/}" ]]; then
        echo "RLLM_VERIFICATION_STATS_POLL_SECONDS must be a number > 0" >&2
        exit 2
    fi
    rm -f -- "${VERIFICATION_STATS_STOP_PATH}"
    python3 -u "${SWE_MILE_ROOT}/rllm/metrics/monitor.py" \
        "${ROLLOUT_LOG_PATH}" \
        --statistics-script "${SWE_MILE_ROOT}/rllm/metrics/verification_statistics.py" \
        --cache-file "${VERIFICATION_STATS_CACHE_PATH}" \
        --output-file "${VERIFICATION_STATS_OUTPUT_PATH}" \
        --stop-file "${VERIFICATION_STATS_STOP_PATH}" \
        --every "${VERIFICATION_STATS_EVERY}" \
        --workers "${VERIFICATION_STATS_WORKERS}" \
        --poll-seconds "${VERIFICATION_STATS_POLL_SECONDS}" \
        --mode train \
        >>"${VERIFICATION_STATS_MONITOR_LOG_PATH}" 2>&1 &
    VERIFICATION_STATS_MONITOR_PID=$!
    echo "Verification-potential statistics monitor started: pid=${VERIFICATION_STATS_MONITOR_PID}, every=${VERIFICATION_STATS_EVERY}, workers=${VERIFICATION_STATS_WORKERS}"
fi

if [[ "${ACTION_STATS_ENABLE}" != "0" ]]; then
    if [[ ! "${ACTION_STATS_EVERY}" =~ ^[1-9][0-9]*$ ]]; then
        echo "RLLM_ACTION_STATS_EVERY must be an integer >= 1" >&2
        exit 2
    fi
    if [[ ! "${ACTION_STATS_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "RLLM_ACTION_STATS_WORKERS must be an integer >= 1" >&2
        exit 2
    fi
    if [[ ! "${ACTION_STATS_POLL_SECONDS}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
       [[ -z "${ACTION_STATS_POLL_SECONDS//[0.]/}" ]]; then
        echo "RLLM_ACTION_STATS_POLL_SECONDS must be a number > 0" >&2
        exit 2
    fi
    rm -f -- "${ACTION_STATS_STOP_PATH}"
    python3 -u "${SWE_MILE_ROOT}/rllm/metrics/monitor.py" \
        "${ROLLOUT_LOG_PATH}" \
        --statistics-script "${SWE_MILE_ROOT}/rllm/metrics/action_statistics.py" \
        --cache-file "${ACTION_STATS_CACHE_PATH}" \
        --output-file "${ACTION_STATS_OUTPUT_PATH}" \
        --stop-file "${ACTION_STATS_STOP_PATH}" \
        --every "${ACTION_STATS_EVERY}" \
        --workers "${ACTION_STATS_WORKERS}" \
        --poll-seconds "${ACTION_STATS_POLL_SECONDS}" \
        --mode train \
        >>"${ACTION_STATS_MONITOR_LOG_PATH}" 2>&1 &
    ACTION_STATS_MONITOR_PID=$!
    echo "Action-distribution statistics monitor started: pid=${ACTION_STATS_MONITOR_PID}, every=${ACTION_STATS_EVERY}, workers=${ACTION_STATS_WORKERS}"
fi

TRAIN_EXIT_CODE=0
python3 -u -m launch.train \
    +swe.rollout_log_path="${ROLLOUT_LOG_PATH}" \
    rllm/backend=verl \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +model.name="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=false \
    data.trust_remote_code=true \
    rllm.agent.name=codeflow \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.capture_raw_payloads=false \
    rllm.gateway.renderer_family=qwen3.5 \
    rllm.gateway.routing.mode=group_striped_adaptive \
    rllm.gateway.routing.migration_min_active_gap=4 \
    rllm.gateway.routing.migration_sustain_seconds=30 \
    rllm.gateway.routing.max_migrations_per_session=1 \
    rllm.gateway.routing.metrics_interval_seconds=300 \
    rllm.gateway.session_cleanup_concurrency=32 \
    rllm.gateway.session_cleanup_phase_timeout=30 \
    rllm.gateway.session_cleanup_shutdown_timeout=30 \
    rllm.gateway.session_cleanup_max_pending=4096 \
    rllm.gateway.session_cleanup_stall_timeout=600 \
    rllm.gateway.session_start_admission_rate_per_second=16 \
    rllm.gateway.session_start_admission_burst=16 \
    rllm.gateway.supervision.enable=true \
    rllm.gateway.supervision.health_interval_seconds=5 \
    rllm.gateway.supervision.health_timeout_seconds=2 \
    rllm.gateway.supervision.failure_threshold=3 \
    rllm.gateway.supervision.max_restarts=1 \
    rllm.gateway.supervision.recovery_stall_timeout_seconds=120 \
    rllm.data.train_batch_size=1 \
    rllm.data.val_batch_size=16 \
    rllm.data.dynamic_sequence_budget=true \
    rllm.rollout.n_val=1 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=0.95 \
    +rllm.rollout.train.top_k=20 \
    rllm.rollout.val.temperature=0.6 \
    rllm.rollout.val.top_p=0.95 \
    +rllm.rollout.val.top_k=20 \
    rllm.rejection_sample.multiplier=1 \
    rllm.rejection_sample.filter_uniform_groups=false \
    rllm.dynamic_sampling.enable=true \
    rllm.workflow.retry_limit=3 \
    rllm.workflow.raise_on_error=false \
    rllm.workflow.warm_queue_size=0 \
    rllm.algorithm.adv_estimator=rloo \
    rllm.algorithm.loss_fn=vanilla \
    rllm.algorithm.loss_agg_mode=seq-mean-token-mean \
    rllm.algorithm.kl_beta=0.001 \
    rllm.algorithm.rollout_correction.bypass_mode=false \
    rllm.algorithm.rollout_correction.tis_mode=token \
    rllm.algorithm.rollout_correction.tis_cap=2.0 \
    rllm.stepwise_advantage.enable=true \
    rllm.stepwise_advantage.mode=per_step \
    rllm.stepwise_advantage.outcome_weight=1.0 \
    rllm.stepwise_advantage.tool_call_weight=0.25 \
    rllm.stepwise_advantage.tool_call_correct_reward=0.0 \
    rllm.stepwise_advantage.tool_call_incorrect_reward=-1.0 \
    rllm.async_training.enable=true \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.async_training.server_init_max_attempts=3 \
    rllm.async_training.server_init_retry_quiescence_seconds=15 \
    rllm.async_training.server_init_gpu_release_timeout_seconds=300 \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.ref.use_torch_compile=false \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=null \
    rllm.algorithm.eps_clip=0.2 \
    rllm.algorithm.eps_clip_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    +actor_rollout_ref.rollout.enable_sleep_mode=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=null \
    trainer.default_local_dir="${TRAINING_SAVE_PATH}" \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    trainer.max_actor_ckpt_to_keep=1000 \
    +trainer.hf_save_dir="${HF_SAVE_PATH}" \
    +trainer.hf_save_dtype=bfloat16 \
    +trainer.hf_save_max_shard_size=5GB \
    +trainer.max_hf_ckpt_to_keep=1000 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    actor_rollout_ref.actor.checkpoint.load_contents='["model","optimizer","extra"]' \
    rllm.trainer.logger="['console','file','wandb']" \
    'rllm.trainer.project_name=${swe.train_dataset}-codeflow-milestone-rloo' \
    rllm.trainer.experiment_name="${EXPERIMENT_NAME}" \
    rllm.trainer.wandb_mode=offline \
    rllm.trainer.wandb_dir="${WANDB_RUN_DIR}" \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.total_batches=-1 \
    rllm.trainer.test_freq=0 \
    rllm.trainer.val_before_train=false \
    rllm.episode_logging.log_episodes=false \
    +swe.protocol=native_tool_call \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=qwen3 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen3_coder \
    "${TRAIN_PROFILE_ARGS[@]}" \
    "${HYDRA_ARGS[@]}" \
    "${RESUME_OVERRIDES[@]}" || TRAIN_EXIT_CODE=$?

exit "${TRAIN_EXIT_CODE}"

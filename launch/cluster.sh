#!/usr/bin/env bash
# Run on every node in the allocation, with a unique RANK and a common MASTER_ADDR.
set -euo pipefail
SWE_MILE_REQUIRE_MODEL=1
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
: "${RANK:?Set RANK to 0 on the head, then 1, 2, ... on workers}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the head node address}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TRAIN_NODES="${TRAIN_NODES:-4}"
ROLLOUT_NODES="${ROLLOUT_NODES:-4}"
RAY_PORT="${RAY_PORT:-6379}"
EXPECTED_GPUS=$(( (TRAIN_NODES + ROLLOUT_NODES) * GPUS_PER_NODE ))
export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
if [[ "${RANK}" == "0" ]]; then
    ray start --head --node-ip-address="${MASTER_ADDR}" --port="${RAY_PORT}" \
        --num-gpus="${GPUS_PER_NODE}" --disable-usage-stats
    export SWE_MILE_EXPECTED_GPUS="${EXPECTED_GPUS}"
    python - <<'PY'
import os, time
import ray
ray.init(address=os.environ["RAY_ADDRESS"])
expected = int(os.environ["SWE_MILE_EXPECTED_GPUS"])
deadline = time.monotonic() + float(os.environ.get("RAY_BOOT_TIMEOUT", "1800"))
while ray.cluster_resources().get("GPU", 0) < expected:
    if time.monotonic() >= deadline:
        raise SystemExit(f"Ray startup timed out: need {expected} GPUs, got {ray.cluster_resources().get('GPU', 0)}")
    time.sleep(5)
ray.shutdown()
PY
    bash launch/train.sh trainer.nnodes="${TRAIN_NODES}" rollout.nnodes="${ROLLOUT_NODES}" \
        trainer.n_gpus_per_node="${GPUS_PER_NODE}" rollout.n_gpus_per_node="${GPUS_PER_NODE}" "$@"
else
    python - <<'PY'
import os, socket, time
host, port = os.environ["RAY_ADDRESS"].rsplit(":", 1)
deadline = time.monotonic() + float(os.environ.get("RAY_BOOT_TIMEOUT", "1800"))
while True:
    try:
        with socket.create_connection((host, int(port)), timeout=5):
            break
    except OSError:
        if time.monotonic() >= deadline:
            raise SystemExit("Ray head startup timed out")
        time.sleep(5)
PY
    ray start --address="${RAY_ADDRESS}" --num-gpus="${GPUS_PER_NODE}" --disable-usage-stats --block
fi

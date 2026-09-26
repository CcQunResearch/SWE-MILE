#!/bin/bash
# rllm-doc2repo-verifier-version: 1
# Preserve the pinned benchmark's pip install / unzip / pytest / scorer flow.
# Only failures in host-owned verifier assets and control steps are infra faults.
set -uo pipefail

mkdir -p /logs/verifier /tmp/rllm || exit 1
REWARD=/tmp/rllm/reward.json
INSTALL_LOG=/tmp/rllm/doc2repo_install.log
EXTRACT_LOG=/tmp/rllm/doc2repo_extract.log
PYTEST_LOG=/tmp/rllm/doc2repo_pytest.log
SCORE_LOG=/tmp/rllm/doc2repo_score.log
rm -f "$REWARD" /logs/verifier/reward.txt "$INSTALL_LOG" "$EXTRACT_LOG" "$PYTEST_LOG" "$SCORE_LOG" || exit 1

fail() {
    python3 - "$1" <<'PY'
import json, sys
reason = sys.argv[1]
details = {}
for name in ("install", "extract", "pytest", "score"):
    try:
        with open("/tmp/rllm/doc2repo_{}.log".format(name), errors="replace") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 4000))
            details[name + "_tail"] = stream.read()
    except (OSError, UnicodeError):
        pass
json.dump({"reward": 0.0, "is_correct": False, "metadata": {
    "error": reason, "verifier_diagnostics": details,
    "infrastructure_failure": {"reason": reason, "stage": "verifier",
                               "exception_type": reason, "error_summary": reason, "retryable": False}
}}, open("/tmp/rllm/reward.json", "w"))
PY
    exit 1
}

write_reward() {
    python3 - "$1" "$2" "$3" <<'PY'
import json, math, sys
score = float(sys.argv[1])
if not math.isfinite(score) or not 0.0 <= score <= 1.0:
    raise ValueError("invalid Doc2Repo score")
details = {"install_exit_code": int(sys.argv[2]),
           "pytest_exit_code": int(sys.argv[3]) if sys.argv[3] else None}
for name in ("install", "extract", "pytest", "score"):
    try:
        with open("/tmp/rllm/doc2repo_{}.log".format(name), errors="replace") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 4000))
            details[name + "_tail"] = stream.read()
    except (OSError, UnicodeError):
        pass
json.dump({"reward": score, "is_correct": score >= 1.0,
           "metadata": {"verifier_diagnostics": details}}, open("/tmp/rllm/reward.json", "w"))
PY
}

cd /workspace || fail verifier_workspace_missing
for tool in python3 pip unzip; do
    command -v "$tool" >/dev/null 2>&1 || fail verifier_tool_missing
done
for asset in /tests/test_suite.zip /tests/score_pytest.py; do
    [ -f "$asset" ] || fail verifier_asset_missing
done
# Detect corrupt cached/uploaded suites even when the model cannot install.
unzip -tq /tests/test_suite.zip > "$EXTRACT_LOG" 2>&1 || fail verifier_test_suite_invalid

pip install -e . > "$INSTALL_LOG" 2>&1
install_exit=$?
if [ "$install_exit" -ne 0 ]; then
    # An unbuildable generated project is a normal benchmark zero.
    write_reward 0.0 "$install_exit" '' || fail verifier_score_invalid
    exit 1
fi

unzip -o /tests/test_suite.zip -d . > "$EXTRACT_LOG" 2>&1 || fail verifier_test_suite_extract_failed
[ -d test_case ] || fail verifier_test_suite_invalid
python3 -m pytest test_case/ -o 'python_files=*.py' -v > "$PYTEST_LOG" 2>&1
pytest_exit=$?
# Keep the upstream scoring helper, including its handling of collection errors.
# A nonzero pytest exit is expected for failed tests and may still earn partial credit.
score=$(python3 /tests/score_pytest.py < "$PYTEST_LOG" 2> "$SCORE_LOG") || fail verifier_scorer_failed
write_reward "$score" "$install_exit" "$pytest_exit" || fail verifier_score_invalid
echo "Score: $score"
if [ "$score" = "1.0" ] || [ "$score" = "1.000000" ]; then
    echo '<pytest>true</pytest>'
    exit 0
fi
echo '<pytest>false</pytest>'
exit 1

import pytest

from rllm.eval.repo_generation_status import command_failure, parse_doc2repo_summary


@pytest.mark.parametrize("output,expected", [
    ("2 passed, 11 failed, 12 errors in 1.0s", (2, 11, 12)),
    ("Results (1.0s):\n  6 failed\n  2 passed\n", (2, 6, 0)),
    ("Results (1.0s):\n  2 passed\n  6 failed\n", (2, 6, 0)),
    ("====== 90 passed in 1s ======\n====== 1 passed, 2 errors in 1s ======", (1, 0, 2)),
    ("\x1b[31m6 failed\x1b[0m\n\x1b[32m2 passed\x1b[0m", (2, 6, 0)),
])
def test_final_summary_counts(output, expected):
    result = parse_doc2repo_summary(output)
    assert (result["passed"], result["failed"], result["errors"]) == expected


def test_model_missing_file_does_not_become_stale_image_path():
    log = "FileNotFoundError: No such file or directory: '/workspace/generated_missing.py'"
    assert command_failure("pip install -e .", 1, log, trusted_install=True)["status"] == "install_failed"
    log = "FileNotFoundError: No such file or directory: '/records'"
    assert command_failure("pip install -e .", 1, log)["status"] == "install_failed"


@pytest.mark.parametrize('detail', [
    "E SyntaxError: unterminated triple-quoted string literal",
    "E ImportError: cannot import name 'BearerTransport' from 'fastapi_users.authentication'",
    "E ModuleNotFoundError: No module named 'rich_click.rich_click'",
    'E PydanticSchemaGenerationError: unable to generate schema',
])
def test_pytest_conftest_submission_failure_is_not_infrastructure(detail):
    output = "ImportError while loading conftest '/workspace/tests/conftest.py'.\n" + detail
    assert command_failure('pytest tests', 4, output)['status'] == 'collection_failed'


def test_real_runner_usage_and_missing_tool_remain_infrastructure():
    assert command_failure('pytest tests', 4, 'ERROR: unrecognized arguments: --missing-plugin')['status'] == 'infrastructure_failure'
    assert command_failure('pytest tests', 127, 'bash: pytest: command not found')['reason'] == 'verifier_toolchain_missing'

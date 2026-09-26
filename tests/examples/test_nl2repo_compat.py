import importlib
from pathlib import Path
import subprocess

from omegaconf import OmegaConf

from rllm.harnesses.swe_scaffold import CodeFlowScaffoldHarness
from rllm.types import AgentConfig, Episode, Task


def launcher():
    root = Path(__file__).parents[2]
    return importlib.import_module('launch.eval' if (root / 'launch/eval.py').exists() else 'examples.swe.eval')


def test_nl2repo_proxy_is_task_scoped_and_changes_semantics():
    module = launcher()
    task = Task(id='demo', instruction='build', metadata={'task_profile': 'repo_generation_nl2repo'})
    config = _config()
    for key, value in {'benchmark_profile': 'nl2repo', 'dataset_name': 'nl2repo-bench', 'dataset_split': 'test', 'expected_tasks': 103}.items():
        OmegaConf.update(config, 'eval.' + key, value, force_add=True)
    direct = module._semantic_config(config, [task])
    OmegaConf.update(config, 'eval.repo_generation_proxy_url', 'http://proxy:80', force_add=True)
    proxied = module._semantic_config(config, [task])
    assert direct != proxied
    assert proxied['agent']['nl2repo_setup'] == 'offline_git_primary_cleanup_image_contract_v4'
    module._configure_tasks([task], benchmark_profile='nl2repo', repo_generation_proxy_url='http://proxy:80')
    assert task.metadata['rllm']['repo_generation_proxy_url'] == 'http://proxy:80'
    assert 'agent_environment_policy' not in task.metadata['rllm']


def test_nl2repo_bash_uses_repaired_index_and_proxy(monkeypatch):
    harness = CodeFlowScaffoldHarness()
    task = Task(id='demo', instruction='build', metadata={'task_profile': 'repo_generation_nl2repo', 'rllm': {'repo_generation_proxy_url': 'http://proxy:80'}})
    monkeypatch.setenv('PIP_INDEX_URL', 'https://bytedpypi.byted.org/simple')

    def run(script, payload, *args, **kwargs):
        result = subprocess.run(['bash', '-c', payload['command']], text=True, capture_output=True)
        return {'ok': result.returncode == 0, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'timed_out': False}

    monkeypatch.setattr(harness, '_codeflow_run_json_script', run)
    result = harness._codeflow_execute_bash_unchecked({'command': 'printf "%s|%s" "$PIP_INDEX_URL" "$https_proxy"', 'timeout': 10}, object(), task)
    assert result.exit_code == 0
    assert 'https://pypi.org/simple|http://proxy:80' in result.observation


def test_nl2repo_generation_prompt_and_snapshot_are_used(monkeypatch):
    harness = CodeFlowScaffoldHarness()
    task = Task(id='demo', instruction='build', metadata={'task_profile': 'repo_generation_nl2repo'})
    captured = []
    snapshot = {'schema_version': 1, 'packages': {'output': 'pytest==8.4.1'}, 'editable': {'output': '[]'}}
    monkeypatch.setattr('rllm.sandbox.repo_generation_environment.inspect_repo_generation_environment', lambda *args, **kwargs: snapshot)

    def model(client, task, config, env, messages, **kwargs):
        captured.extend(messages)
        return Episode(id='test')

    monkeypatch.setattr(harness, '_run_native_tool_call', model)
    result = harness._run_with_client(object(), task, AgentConfig(base_url='http://gateway/v1', model='model', session_uid='test'), env=object())
    assert 'implementing a complete software package' in captured[0]['content']
    assert 'pytest==8.4.1' in captured[1]['content']
    assert result.metadata['repo_generation_environment'] == snapshot


def _config():
    return OmegaConf.create(
        {
            "eval": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "pass_n": 4,
                "checkpoint_stride": 1,
            },
            "swe": {
                "max_turns": 100,
                "command_timeout": 300,
                "verifier_timeout": 600,
                "sandbox_backend": "gemini",
                "sandbox_cpus": 2,
                "sandbox_memory_mb": 8192,
                "shadow_sandbox_cpus": 2,
                "shadow_sandbox_memory_mb": 8192,
                "shadow_validation_enabled": True,
            },
            "rllm": {
                "data": {"dynamic_sequence_budget": True},
                "gateway": {
                    "cumulative_token_mode": True,
                    "routing": {"mode": "group_striped_adaptive"},
                    "renderer_family": "qwen3.5",
                },
                "rollout": {"val": {"max_tokens": 2048}},
                "workflow": {"n_parallel_tasks": 288},
            },
            "actor_rollout_ref": {
                "rollout": {
                    "max_model_len": 98304,
                    "engine_kwargs": {
                        "vllm": {
                            "reasoning_parser": "qwen3",
                            "tool_call_parser": "qwen3_coder",
                        }
                    },
                }
            },
        }
    )

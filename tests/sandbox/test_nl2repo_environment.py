import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rllm.sandbox.nl2repo_environment import ASSET_PATH, compatible_nl2repo_command, prepare_nl2repo_image
from rllm.sandbox.repo_generation_environment import repo_generation_environment_exports
from rllm.types import RolloutInfrastructureError


def asset():
    spec = importlib.util.spec_from_file_location('nl2repo_asset', ASSET_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stale_target_editable_is_removed_without_touching_dependencies(tmp_path):
    link = tmp_path / 'python_box.egg-link'
    old = '/rllm-test-obsolete-image-build-root'
    link.write_text(old + '\n.\n')
    dependency = tmp_path / 'dependency.egg-link'
    dependency.write_text('/another-missing-build-root\n.\n')
    pth = tmp_path / 'easy-install.pth'
    pth.write_text(old + '\n/another-missing-build-root\nimport image_hook\n')
    repairs = asset().remove_stale_editables([tmp_path], 'python-box')
    assert repairs[0]['old_source'] == old and not link.exists()
    assert dependency.is_file()
    assert pth.read_text() == '/another-missing-build-root\nimport image_hook\n'
    assert asset().remove_stale_editables([tmp_path], 'python-box') == []


def test_live_editable_and_symlink_metadata_are_preserved(tmp_path):
    (tmp_path / 'target.egg-link').write_text(str(tmp_path) + '\n.\n')
    assert asset().remove_stale_editables([tmp_path], 'target') == []
    metadata = tmp_path / 'outside'
    metadata.write_text('/rllm-test-stale-root\n.\n')
    (tmp_path / 'alias.egg-link').symlink_to(metadata)
    assert asset().remove_stale_editables([tmp_path], 'alias') == []
    assert metadata.read_text() == '/rllm-test-stale-root\n.\n'


def test_image_user_packages_become_readable_without_upgrading_or_granting_write(tmp_path):
    home = tmp_path / 'home'
    local = home / '.local'
    site = local / 'lib/python3.9/site-packages'
    site.mkdir(parents=True)
    binary = local / 'bin'
    binary.mkdir()
    tool = binary / 'pytest'
    tool.write_text('#!/usr/bin/python\n')
    tool.chmod(0o700)
    home.chmod(0o700)
    site.chmod(0o700)
    metadata = site / 'original.py'
    metadata.write_text('VERSION = 1\n')
    metadata.chmod(0o600)
    system_site = tmp_path / 'system-site'
    system_site.mkdir()
    result = asset().expose_image_user_site(home, system_site, '3.9')
    assert result['kind'] == 'image_user_site_exposed'
    assert (system_site / 'rllm_nl2repo_image_user_site.pth').read_text() == str(site) + '\n'
    assert metadata.read_text() == 'VERSION = 1\n'
    assert metadata.stat().st_mode & 0o007 == 0o004
    assert tool.stat().st_mode & 0o007 == 0o005
    assert home.stat().st_mode & 0o007 == 0o001


def test_mismatched_boto_packaging_is_rejected_without_executing_it(tmp_path):
    (tmp_path / 'setup.py').write_text("raise RuntimeError('must not execute')\nsetup(name='boto3')\n")
    result = asset().prepare({'instance_id': 'boto', 'workdir': str(tmp_path), 'package_name': 'boto'})
    assert result['ok'] is False
    assert result['reason'] == 'nl2repo_asset_contract_mismatch'


def test_tqdm_exact_command_repair_produces_valid_python(tmp_path):
    (tmp_path / 'tqdm').mkdir()
    broken = "echo __version__ = " + "\\" * 2 + "'0.0.1" + "\\" * 2 + "' > tqdm/version.py"
    fixed = compatible_nl2repo_command(broken, {'instance_id': 'tqdm'})
    subprocess.run(['bash', '-c', fixed], cwd=tmp_path, check=True)
    content = (tmp_path / 'tqdm/version.py').read_text()
    compile(content, 'version.py', 'exec')
    assert content == "__version__ = '0.0.1'\n"
    assert compatible_nl2repo_command(broken, {'instance_id': 'other'}) == broken
    assert compatible_nl2repo_command('echo unrelated', {'instance_id': 'tqdm'}) == 'echo unrelated'


@pytest.mark.parametrize('index,expected', [
    ('https://bytedpypi.byted.org/simple', 'https://pypi.org/simple'),
    ('https://bytedpypi.byted.org/simple/', 'https://pypi.org/simple'),
    ('https://approved-mirror.example/simple', 'https://approved-mirror.example/simple'),
    ('', ''),
])
def test_only_known_baked_private_index_is_normalized(index, expected):
    env = {**os.environ, 'PIP_INDEX_URL': index}
    output = subprocess.check_output(['bash', '-c', repo_generation_environment_exports('http://proxy:80') + 'printf "%s|%s" "$PIP_INDEX_URL" "$https_proxy"'], env=env, text=True)
    assert output == expected + '|http://proxy:80'


def test_preparation_wrapper_propagates_asset_reason(tmp_path):
    (tmp_path / 'setup.py').write_text("setup(name='boto3')\n")

    class Shell:
        def exec_setup(self, command, timeout=None, user=None):
            assert 'python3 -I -S -c ' in command
            assert 'python3 -c ' not in command
            assert user == 'root'
            return subprocess.check_output(['bash', '-c', command], timeout=timeout, text=True)

    with pytest.raises(RolloutInfrastructureError) as caught:
        prepare_nl2repo_image(Shell(), {'instance_id': 'boto', 'package_name': 'boto', 'workdir': str(tmp_path)})
    assert caught.value.reason == 'nl2repo_asset_contract_mismatch'

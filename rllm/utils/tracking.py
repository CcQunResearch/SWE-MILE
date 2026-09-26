# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from verl/verl/utils/tracking.py
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import json
import numbers
import os
import pprint
import sys
import time
from collections.abc import Mapping
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

from rllm.env import env_float

_UI_HTTP_TIMEOUT_S = env_float("RLLM_UI_HTTP_TIMEOUT_S", 5.0)  # set env var: export RLLM_UI_HTTP_TIMEOUT_S=xxx


def concat_dict_to_str(dict: dict, step):
    output = [f"step:{step}"]
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{pprint.pformat(v)}")
    output_str = " - ".join(output)
    return output_str


def _replay_wandb_history(wandb_module, history_path: str, through_step: int) -> int:
    """Replay exported offline metrics into a newly initialized W&B run."""

    path = Path(history_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"W&B replay history does not exist: {path}")
    previous_step = -1
    replayed = 0
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid W&B replay JSON at {path}:{line_number}: {exc}"
                ) from exc
            step = record.get("step") if isinstance(record, dict) else None
            data = record.get("data") if isinstance(record, dict) else None
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(f"invalid W&B replay step at {path}:{line_number}")
            if not isinstance(data, dict):
                raise ValueError(f"invalid W&B replay data at {path}:{line_number}")
            if step <= previous_step:
                raise ValueError(
                    f"W&B replay steps must be strictly increasing at {path}:{line_number}"
                )
            if step > through_step:
                raise ValueError(
                    f"W&B replay step {step} exceeds configured cutoff {through_step}"
                )
            wandb_module.log(data, step=step)
            previous_step = step
            replayed += 1
    if replayed == 0:
        raise ValueError(f"W&B replay history is empty: {path}")
    return replayed


class LocalLogger:
    """
    A local logger that logs messages to the console.

    Args:
        print_to_console (bool): Whether to print to the console.
    """

    def __init__(self, print_to_console=True):
        self.print_to_console = print_to_console

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = ["wandb", "console", "file"]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None, source_metadata=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}
        self._finished = False  # Track whether finish() has been called

        rllm_config = config.get("rllm", {}) if config is not None else {}
        trainer_config = (
            rllm_config.get("trainer", {})
            if isinstance(rllm_config, Mapping)
            else {}
        )

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            settings = None
            if trainer_config.get("wandb_proxy"):
                settings = wandb.Settings(https_proxy=trainer_config["wandb_proxy"])

            wandb_kwargs = {
                "project": project_name,
                "name": experiment_name,
                "config": config,
            }
            if settings is not None:
                wandb_kwargs["settings"] = settings
            if trainer_config.get("wandb_mode"):
                wandb_kwargs["mode"] = trainer_config["wandb_mode"]
            if trainer_config.get("wandb_dir"):
                wandb_kwargs["dir"] = str(trainer_config["wandb_dir"])

            wandb.init(**wandb_kwargs)
            replay_path = trainer_config.get("wandb_replay_history_path")
            replay_through_step = trainer_config.get("wandb_replay_through_step")
            if replay_path:
                if isinstance(replay_through_step, bool) or not isinstance(
                    replay_through_step, int
                ):
                    wandb.finish(exit_code=1)
                    raise ValueError(
                        "rllm.trainer.wandb_replay_through_step must be an integer"
                    )
                try:
                    replayed = _replay_wandb_history(
                        wandb,
                        str(replay_path),
                        replay_through_step,
                    )
                except BaseException:
                    wandb.finish(exit_code=1)
                    raise
                print(
                    f"Replayed {replayed} W&B history records through step "
                    f"{replay_through_step} from {replay_path}",
                    flush=True,
                )
            self.logger["wandb"] = wandb






        if "console" in default_backend:
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger


        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)


    def log(self, data, step, backend=None, episodes=None, trajectory_groups=None):
        """Log metrics and optionally episodes/trajectory_groups to configured backends.

        Args:
            data: Dictionary of metrics to log
            step: Current training step
            backend: Optional list of backends to log to (default: all)
            episodes: Optional list of Episode objects (only used by UILogger)
            trajectory_groups: Optional list of TrajectoryGroup objects (only used by UILogger)
        """
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                if default_backend == "ui":
                    logger_instance.log(data=data, step=step, episodes=episodes, trajectory_groups=trajectory_groups)
                else:
                    logger_instance.log(data=data, step=step)

    def finish(self):
        """Explicitly finish and cleanup all loggers.

        This method should be called during controlled shutdown to ensure proper cleanup.
        It's safe to call multiple times - subsequent calls will be no-ops.
        """
        if self._finished:
            return

        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "file" in self.logger:
            self.logger["file"].finish()

        self.logger.clear()
        self._finished = True

    def __del__(self):
        """Destructor that ensures cleanup if finish() wasn't called explicitly.

        Note: Prefer calling finish() explicitly during shutdown rather than relying
        on __del__, as garbage collection timing can be unpredictable.
        """
        self.finish()








class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
        else:
            directory = os.path.dirname(os.path.abspath(self.filepath))
            if directory:
                os.makedirs(directory, exist_ok=True)
        print(f"Creating file logger at {self.filepath}")
        self.fp = open(self.filepath, "w", encoding="utf-8")

    @staticmethod
    def _json_default(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, set | tuple):
            return list(obj)

        item = getattr(obj, "item", None)
        if callable(item):
            try:
                value = item()
                if value is None or isinstance(value, str | bool | int | float):
                    return value
            except (TypeError, ValueError, RuntimeError):
                pass

        tolist = getattr(obj, "tolist", None)
        if callable(tolist):
            try:
                return tolist()
            except (TypeError, ValueError, RuntimeError):
                pass

        return str(obj)

    def log(self, data, step):
        data = {"step": step, "data": data}
        self.fp.write(json.dumps(data, default=self._json_default) + "\n")
        self.fp.flush()

    def finish(self):
        if not self.fp.closed:
            self.fp.close()













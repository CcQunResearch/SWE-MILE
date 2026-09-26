import logging

import ray
from omegaconf import DictConfig

from rllm.data import Dataset
from rllm.trainer.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
from rllm.trainer.verl.train_agent_ppo import TaskRunner
from rllm.trainer.verl.verl_backend import VerlBackend
from rllm.workflows.workflow import Workflow

logger = logging.getLogger(__name__)


def _validate_ray_gpu_capacity(config: DictConfig, resources: dict[str, float] | None = None) -> int:
    """Verify separated trainer+rollout placement before starting workers."""
    if not bool(config.rllm.get("async_training", {}).get("enable", False)):
        return 0
    trainer_gpus = int(config.trainer.nnodes) * int(config.trainer.n_gpus_per_node)
    rollout_gpus = int(config.rollout.nnodes) * int(config.rollout.n_gpus_per_node)
    required = trainer_gpus + rollout_gpus
    available = float((ray.cluster_resources() if resources is None else resources).get("GPU", 0.0))
    if available < required:
        raise RuntimeError(
            "Ray cluster has insufficient GPUs for separated training: "
            f"available={available:g}, required={required} "
            f"(trainer={trainer_gpus}, rollout={rollout_gpus})"
        )
    logger.info(
        "Validated separated Ray GPU capacity: available=%s required=%s (trainer=%s rollout=%s)",
        f"{available:g}",
        required,
        trainer_gpus,
        rollout_gpus,
    )
    return required


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class VerlTaskRunner(TaskRunner):
    """Ray remote class for executing training with the unified trainer."""

    def run(self, config, workflow_class: type[Workflow], workflow_args: dict, hydra_overrides: list[str] | None = None, train_dataset=None, val_dataset=None, **kwargs):  # type: ignore
        import os
        import socket
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.trainer.ppo.utils import need_reference_policy
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.config import validate_config
        from verl.utils.fs import copy_to_local

        from rllm.trainer.verl.utils import sync_config

        print(f"VerlTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        sync_config(config, hydra_overrides=hydra_overrides)
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))

        is_separated = config.rllm.get("async_training", {}).get("enable", False)
        if is_separated:
            from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
            from verl.trainer.ppo.utils import Role

            # Propagate rollout GPU config into actor_rollout_ref (verl convention)
            config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
            config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

            role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
            # Trainer resource pool: all roles except Rollout (rollout runs standalone via AgentLoopManager)
            trainer_roles = {r: cls for r, cls in role_worker_mapping.items() if r != Role.Rollout}
            resource_pool_manager = create_resource_pool_manager(config, roles=list(trainer_roles.keys()))
        else:
            actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
            self.add_ref_policy_worker(config, actor_rollout_cls)
            trainer_roles = self.role_worker_mapping
            resource_pool_manager = self.init_resource_pool_mgr(config)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=False,
        )

        # Download the checkpoint from HDFS to the local machine.
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Assemble backend-specific arguments for initializing the verl backend.
        backend_args = {
            "tokenizer": tokenizer,
            "processor": processor,
            "role_worker_mapping": trainer_roles,
            "resource_pool_manager": resource_pool_manager,
            "ray_worker_group_cls": ray_worker_group_cls,
        }

        trainer = None
        primary_error = None
        try:
            trainer = UnifiedTrainer(
                backend_cls=VerlBackend,
                config=config,
                workflow_class=workflow_class,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                workflow_args=workflow_args,
                backend_args=backend_args,
                **kwargs,
            )
            trainer.fit()
        except BaseException as e:
            primary_error = e
            try:
                from rllm.utils.diagnostic_events import emit_diagnostic
                from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
                emit_diagnostic(
                    "TRAINER_INITIALIZATION_ERROR" if trainer is None else "TRAINER_EXECUTION_ERROR",
                    error_type=type(e).__name__, error=str(e)[:4000],
                    exception_chain=infrastructure_exception_chain(e),
                )
            except Exception as diagnostic_error:
                print(f"Failed to record training error diagnostics: {diagnostic_error}")
            print(f"Error training Verl: {e}")
            raise
        finally:
            try:
                if trainer is not None:
                    from rllm.utils.shutdown import shutdown_preserving_error
                    shutdown_preserving_error(trainer, primary_error)
            finally:
                from rllm.utils.diagnostic_events import close_diagnostics
                close_diagnostics()


class VerlTrainerLauncher(TrainerLauncher):
    """
    Verl trainer launcher that handles the necessary setup for the verl backend.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        **kwargs,
    ):
        """Initialize the VerlTrainerLauncher. The heavy lifting is done in the `run` method of the `TaskRunner` class."""
        super().__init__(config, workflow_class, train_dataset, val_dataset, workflow_args, **kwargs)

    def train(self):
        from rllm.utils.diagnostic_events import configure_diagnostics, close_diagnostics
        diagnostic_dir = configure_diagnostics(self.config.rllm.get("trainer", {}).get("wandb_dir"))
        if diagnostic_dir:
            logger.info("Training diagnostics: %s", diagnostic_dir)
        own_ray = False
        runner = None
        if not ray.is_initialized():
            from rllm.trainer.ray_init_utils import get_ray_init_settings

            ray_init_settings = get_ray_init_settings(self.config)
            ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)
            own_ray = True

        try:
            _validate_ray_gpu_capacity(self.config)
            from rllm.utils.runtime_provenance import verify_training_runtime

            verify_training_runtime(ray, self.config)

            # Capture Hydra CLI overrides while we're still in the Hydra-decorated
            # process; the Ray actor below cannot read HydraConfig itself.
            try:
                from hydra.core.hydra_config import HydraConfig

                hydra_overrides = list(HydraConfig.get().overrides.task)
            except (ValueError, AttributeError, ImportError):
                hydra_overrides = []

            runner_type = VerlTaskRunner
            if diagnostic_dir:
                # Also propagate into an already initialized Ray job. Children
                # inherit this explicit actor runtime_env.
                import os
                diagnostic_env = {key: os.environ[key] for key in (
                    "RLLM_DIAGNOSTICS_DIR", "RLLM_DIAGNOSTICS_RUN_ID", "RLLM_TRAINING_PARAMS_PATH",
                ) if key in os.environ}
                runner_type = runner_type.options(runtime_env={"env_vars": diagnostic_env})
            runner = runner_type.remote()  # type: ignore

            ray.get(
                runner.run.remote(
                    config=self.config,
                    workflow_class=self.workflow_class,
                    workflow_args=self.workflow_args,
                    store=self.store,
                    hydra_overrides=hydra_overrides,
                    train_dataset=self.train_dataset,
                    val_dataset=self.val_dataset,
                    **self.kwargs,
                )
            )
        finally:
            # The TaskRunner normally tears down its trainer (and every
            # TaskContext-owned sandbox) in ``run``'s finally block.  If the
            # driver is interrupted while ray.get is waiting, explicitly stop
            # the actor before the launch shell consumes the run-scoped sandbox
            # registry; otherwise it could keep creating sandboxes while the
            # compensating cleanup is already in progress.
            if runner is not None:
                try:
                    ray.kill(runner, no_restart=True)
                except Exception:
                    logger.exception("Could not terminate VerlTaskRunner during launcher cleanup")
            if own_ray:
                try:
                    ray.shutdown()
                except Exception:
                    logger.exception("ray.shutdown during launcher cleanup failed")
            close_diagnostics()

import contextlib
import logging
import math
import time
from typing import Any, Iterable

import torch
from monarch.actor import current_rank, current_size, endpoint
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

from apps.sft.data import setup_dataloader
from apps.sft.metrics import MetricsTracker
from forge.controller import ForgeActor
from forge.data.utils import StopAfterOneEpoch
from forge.observability import get_or_create_metric_logger

logger = logging.getLogger(__name__)

try:
    from torchtitan.distributed.context_parallel import prepare_context_parallel_input

    HAS_CONTEXT_PARALLEL = True
except ImportError:
    HAS_CONTEXT_PARALLEL = False
    prepare_context_parallel_input = None


def get_optional_mesh(parallel_dims, name: str):
    if name == "pp" and parallel_dims.pp_enabled:
        return parallel_dims.world_mesh["pp"]
    if name == "dp" and parallel_dims.dp_enabled:
        return parallel_dims.world_mesh["dp"]
    if name == "tp" and parallel_dims.tp_enabled:
        return parallel_dims.world_mesh["tp"]
    if name == "cp" and parallel_dims.cp_enabled:
        return parallel_dims.world_mesh["cp"]
    if name == "loss" and parallel_dims.dp_cp_enabled:
        if hasattr(parallel_dims, "world_mesh"):
            mesh_names = parallel_dims.world_mesh.mesh_dim_names
            if "dp_cp" in mesh_names:
                return parallel_dims.world_mesh["dp_cp"]
            elif "dp" in mesh_names:
                return parallel_dims.world_mesh["dp"]
    return None


class TitanSFTTrainer(ForgeActor, ForgeEngine):
    def __init__(self, config: DictConfig | ListConfig):
        job_config_dict = ForgeJobConfig().to_dict()
        job_config: DictConfig | ListConfig = OmegaConf.merge(job_config_dict, config)

        ForgeEngine.__init__(self, job_config)

        self.step = 0
        self.ntokens_seen = 0
        self.num_training_steps = job_config.training.steps
        self._rank = current_rank().rank
        self._size = math.prod(current_size().values())

        self.validation_enabled = False
        self.val_dataloaders = {}
        self.eval_every_n_steps = None
        self.max_eval_steps = None
        self.mlogger = None

        log_freq = job_config.get("metrics", {}).get("log_freq", 1)
        self.metrics = MetricsTracker(
            rank=self._rank,
            world_size=self._size,
            num_flops_per_token=self.num_flops_per_token,
            log_freq=log_freq,
        )

        self._has_checkpointer_wait = False
        self._pp_mesh = None
        self._loss_mesh = None
        self._dp_mesh = None
        self._tp_degree = 1
        self._accumulated_loss = None

        self._register_train_state()

    def _register_train_state(self):
        if hasattr(self, "checkpointer") and self.checkpointer is not None:
            if hasattr(self.checkpointer, "states"):
                self.checkpointer.states["train_state"] = self
                logger.info(
                    f"[Rank {self._rank}] Registered train_state with checkpointer"
                )

    async def _setup_metric_logger(self):
        if self._rank != 0:
            return None
        try:
            return await get_or_create_metric_logger()
        except Exception as e:
            logger.warning(
                f"[Rank {self._rank}] Failed to initialize metric logger: {e}. "
                "Continuing without metric logging."
            )
            return None

    def _cache_parallel_config(self):
        self._has_checkpointer_wait = (
            hasattr(self, "checkpointer")
            and self.checkpointer is not None
            and hasattr(self.checkpointer, "maybe_wait_for_staging")
        )
        self._pp_mesh = get_optional_mesh(self.parallel_dims, "pp")
        self._loss_mesh = get_optional_mesh(self.parallel_dims, "loss")
        self._dp_mesh = get_optional_mesh(self.parallel_dims, "dp")
        self._tp_degree = self._size if self.parallel_dims.tp_enabled else 1
        self._accumulated_loss = torch.tensor(0.0, device=self.device)

    @endpoint
    async def setup(self):
        print(f"[Rank {self._rank}] Starting setup...", flush=True)

        self.rank_should_record_loss = self._rank == 0
        if self.parallel_dims.pp_enabled and not self.pp_has_last_stage:
            self.rank_should_record_loss = False

        print(f"[Rank {self._rank}] Setting up metric logger...", flush=True)
        self.mlogger = await self._setup_metric_logger()

        print(f"[Rank {self._rank}] Setting training datasets...", flush=True)
        train_dataset_config = self.job_config.training.dataset
        self.train_dataloader = setup_dataloader(
            dataset_configs=train_dataset_config,
            max_seq_len=self.job_config.training.seq_len,
            hf_assets_path=self.job_config.model.hf_assets_path,
            batch_size=self.job_config.training.local_batch_size,
            parallel_dims=self.parallel_dims,
        )

        print(f"[Rank {self._rank}] Setting eval config...", flush=True)
        eval_config = self.job_config.get("eval", {})
        self.eval_every_n_steps = eval_config.get("eval_every_n_steps")
        max_eval_steps = eval_config.get("max_eval_steps")
        self.max_eval_steps = (
            max_eval_steps if max_eval_steps and max_eval_steps > 0 else None
        )
        self.validation_enabled = (
            self.eval_every_n_steps is not None and self.eval_every_n_steps > 0
        )

        if self.validation_enabled:
            print(f"[Rank {self._rank}] Setting eval datasets...", flush=True)
            self.eval_dataset_config = eval_config.get("dataset", [])
            for i, dataset_config in enumerate(self.eval_dataset_config):
                ds_name = dataset_config.get("dataset_name", i)
                dataloader = setup_dataloader(
                    dataset_configs=[dataset_config],
                    max_seq_len=self.job_config.training.seq_len,
                    hf_assets_path=self.job_config.model.hf_assets_path,
                    batch_size=self.job_config.training.local_batch_size,
                    parallel_dims=self.parallel_dims,
                )
                self.val_dataloaders[ds_name] = dataloader

        self._cache_parallel_config()

        print(f"[Rank {self._rank}] Loading checkpoint...", flush=True)
        self._load_checkpoint()

        print(
            f"[Rank {self._rank}] Setup complete! Starting from step {self.step}",
            flush=True,
        )

    def _load_checkpoint(self):
        load_step = -1
        if hasattr(self.job_config, "checkpoint") and hasattr(
            self.job_config.checkpoint, "load_step"
        ):
            load_step = self.job_config.checkpoint.load_step

        self.checkpointer.load(step=load_step)

        print(
            f"[Rank {self._rank}] Checkpoint loaded. "
            f"Restored step: {self.step}, ntokens_seen: {self.ntokens_seen}",
            flush=True,
        )

        if self.step > 0:
            current_lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
            print(
                f"[Rank {self._rank}] Resuming training from step {self.step}. "
                f"Current LR: {current_lr:.2e}",
                flush=True,
            )

    def batch_generator(
        self,
        data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        data_iterator = iter(data_iterable)
        tp_degree = self._tp_degree
        device = self.device
        metrics = self.metrics

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration:
                logger.info(f"[Rank {self._rank}] Dataloader exhausted, restarting...")
                data_iterator = iter(data_iterable)
                batch = next(data_iterator)
            data_load_time = time.perf_counter() - data_load_start

            input_tensor = batch.get("input") or batch.get("tokens")
            labels = batch.get("labels")

            if input_tensor is None:
                raise ValueError(
                    f"Batch must contain 'input' or 'tokens' key. Got keys: {batch.keys()}"
                )
            if labels is None:
                raise ValueError(
                    f"Batch must contain 'labels' key. Got keys: {batch.keys()}"
                )

            seq_len = input_tensor.shape[1]
            if tp_degree > 1 and seq_len % tp_degree != 0:
                pad_size = ((seq_len // tp_degree) + 1) * tp_degree - seq_len
                input_tensor = torch.nn.functional.pad(
                    input_tensor, (0, pad_size), value=0
                )
                labels = torch.nn.functional.pad(labels, (0, pad_size), value=-100)

            ntokens_batch = labels.numel()
            self.ntokens_seen += ntokens_batch
            metrics.record_batch(ntokens_batch, data_load_time)

            yield input_tensor.to(device), labels.to(device)

    def forward_backward_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        skip_backward: bool = False,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        if parallel_dims.cp_enabled:
            if not HAS_CONTEXT_PARALLEL:
                raise RuntimeError(
                    "Context parallelism is enabled but torchtitan.distributed.context_parallel "
                    "is not available in your installation."
                )
            extra_kwargs: dict[str, Any] = {}
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                parallel_dims.world_mesh["cp"],
                self.device,
                self.job_config.parallelism.context_parallel_load_balancer,
            )
        else:
            extra_kwargs = {}

        if parallel_dims.pp_enabled:
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )

            if self.pp_has_last_stage:
                loss = torch.sum(torch.stack(losses)).to(self.device)
            else:
                loss = torch.tensor(-1.0, device=self.device)

            if skip_backward:
                loss = loss.detach()
        else:
            with self.train_context():
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                del pred

                if not skip_backward:
                    loss.backward()

        return loss

    def train_step(
        self,
        data_iterator: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, float]:
        self.optimizers.zero_grad()

        self._accumulated_loss.zero_()
        for _ in range(self.gradient_accumulation_steps):
            inputs, labels = next(data_iterator)
            loss = self.forward_backward_step(inputs, labels)
            self._accumulated_loss += loss.detach()

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self._pp_mesh,
            ep_enabled=self.parallel_dims.ep_enabled,
        )

        if self._has_checkpointer_wait:
            self.checkpointer.maybe_wait_for_staging()

        self.optimizers.step()
        self.lr_schedulers.step()

        grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
        return self._accumulated_loss, grad_norm_val

    def should_continue_training(self) -> bool:
        return self.step < self.num_training_steps

    @endpoint
    async def train(self) -> None:
        print(
            f"[Rank {self._rank}] Starting training loop at step {self.step + 1}",
            flush=True,
        )

        self.optimizers.zero_grad()
        self.metrics.start_training()

        data_iterator = self.batch_generator(self.train_dataloader)

        rank_should_record = self.rank_should_record_loss
        dp_cp_enabled = self.parallel_dims.dp_cp_enabled
        loss_mesh = self._loss_mesh
        num_training_steps = self.num_training_steps
        world_size = self._size
        mlogger = self.mlogger
        is_rank_zero = self._rank == 0

        while self.should_continue_training():
            self.step += 1
            step = self.step

            self.gc_handler.run(step)
            self.metrics.start_step()

            accumulated_loss, grad_norm_val = self.train_step(data_iterator)

            if dp_cp_enabled:
                if loss_mesh is not None:
                    global_avg_loss = dist_utils.dist_mean(accumulated_loss, loss_mesh)
                else:
                    global_avg_loss = accumulated_loss.item()
            else:
                global_avg_loss = accumulated_loss.item()

            if rank_should_record:
                self.metrics.record_loss(global_avg_loss)

            should_log = self.metrics.should_log(step)
            if should_log and rank_should_record:
                lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
                ntokens_global = self.ntokens_seen * world_size
                step_metrics = self.metrics.get_step_metrics(step)

                self.metrics.log_to_forge(
                    step=step,
                    loss=global_avg_loss,
                    grad_norm=grad_norm_val,
                    learning_rate=lr,
                    ntokens_seen=ntokens_global,
                    metrics=step_metrics,
                )

                self.metrics.log_to_console(
                    step=step,
                    num_steps=num_training_steps,
                    loss=global_avg_loss,
                    grad_norm=grad_norm_val,
                    learning_rate=lr,
                    ntokens_seen=ntokens_global,
                    metrics=step_metrics,
                )

                self.metrics.reset_logging_window()

            if self.validation_enabled and step % self.eval_every_n_steps == 0:
                await self.evaluate()

            self.checkpointer.save(
                curr_step=step,
                last_step=step == num_training_steps,
            )

            if self.mlogger and is_rank_zero:
                try:
                    await self.mlogger.flush.call_one(global_step=step)
                except Exception:
                    pass

        self.metrics.log_training_complete(self.ntokens_seen)

        if self.validation_enabled:
            logger.info("Running final evaluation at end of training...")
            await self.evaluate()

    @torch.no_grad()
    async def evaluate(self) -> None:
        for model_part in self.model_parts:
            model_part.eval()

        dp_group = self._dp_mesh.get_group() if self._dp_mesh else None

        maybe_no_grad = (
            contextlib.nullcontext()
            if self.parallel_dims.pp_enabled
            else torch.no_grad()
        )

        device = self.device
        rank_should_record = self.rank_should_record_loss

        for dataset_name, val_dataloader in self.val_dataloaders.items():
            logger.info(f"[Rank {self._rank}] Evaluating dataset: {dataset_name}")

            total_loss = torch.tensor(0.0, device=device)
            total_tokens = 0
            num_steps = 0

            eval_start_time = time.perf_counter()

            batch_iter = StopAfterOneEpoch(
                iter=iter(val_dataloader),
                device=device,
                dp_mesh=dp_group,
            )

            max_eval_steps = self.max_eval_steps

            with maybe_no_grad:
                for batch in batch_iter:
                    if max_eval_steps is not None and num_steps >= max_eval_steps:
                        break

                    input_tensor = batch.get("input") or batch.get("tokens")
                    labels = batch.get("labels")

                    if input_tensor is None or labels is None:
                        continue

                    input_tensor = input_tensor.to(device)
                    labels = labels.to(device)

                    total_tokens += labels.numel()

                    loss = self.forward_backward_step(
                        input_tensor, labels, skip_backward=True
                    )
                    total_loss += loss
                    num_steps += 1

            eval_time = time.perf_counter() - eval_start_time
            avg_loss = (total_loss / max(num_steps, 1)).item()
            tps = total_tokens / eval_time if eval_time > 0 else 0

            if rank_should_record:
                self.metrics.log_eval_to_forge(
                    dataset_name=dataset_name,
                    loss=avg_loss,
                    ntokens=total_tokens,
                    eval_time=eval_time,
                )

            logger.info(
                f"[Rank {self._rank}] Eval {dataset_name} | "
                f"Loss: {avg_loss:.4f} | "
                f"PPL: {math.exp(avg_loss):.2f} | "
                f"TPS: {tps * self._size:,.0f} | "
                f"Steps: {num_steps} | "
                f"Time: {eval_time:.1f}s"
            )

        for model_part in self.model_parts:
            model_part.train()

        if self.mlogger and self._rank == 0:
            await self.mlogger.flush.call_one(global_step=self.step)

    @endpoint
    async def cleanup(self) -> None:
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if getattr(self, "mlogger", None):
            await self.mlogger.shutdown.call_one()

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "ntokens_seen": self.ntokens_seen,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict.get("step", 0)
        self.ntokens_seen = state_dict.get("ntokens_seen", 0)
        logger.info(
            f"[Rank {self._rank}] Loaded train_state: step={self.step}, "
            f"ntokens_seen={self.ntokens_seen}"
        )

    def __repr__(self) -> str:
        return "TitanSFTTrainer"

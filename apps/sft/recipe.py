import contextlib
import logging
import math
import time
from typing import Any, Iterable

import torch
from apps.sft.data import setup_dataloader
from apps.sft.metrics import TrainingMetrics, log_training_metrics, log_eval_metrics
from forge.controller import ForgeActor
from forge.data.utils import StopAfterOneEpoch
from forge.observability import get_or_create_metric_logger, record_metric, Reduce
from monarch.actor import current_rank, current_size, endpoint
from omegaconf import DictConfig, OmegaConf
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

logger = logging.getLogger(__name__)

try:
    from torchtitan.distributed.context_parallel import prepare_context_parallel_input
    HAS_CONTEXT_PARALLEL = True
except ImportError:
    HAS_CONTEXT_PARALLEL = False
    prepare_context_parallel_input = None


def get_optional_mesh(parallel_dims, name: str):
    """Helper to get mesh if the dimension is enabled, otherwise return None."""
    if name == "pp" and parallel_dims.pp_enabled:
        return parallel_dims.world_mesh["pp"]
    if name == "dp" and parallel_dims.dp_enabled:
        return parallel_dims.world_mesh["dp"]
    if name == "tp" and parallel_dims.tp_enabled:
        return parallel_dims.world_mesh["tp"]
    if name == "cp" and parallel_dims.cp_enabled:
        return parallel_dims.world_mesh["cp"]
    if name == "loss" and parallel_dims.dp_cp_enabled:
        if hasattr(parallel_dims, "world_mesh") and "dp_cp" in parallel_dims.world_mesh.mesh_dim_names:
            return parallel_dims.world_mesh["dp_cp"]
        elif parallel_dims.dp_enabled:
            return parallel_dims.world_mesh["dp"]
    return None


class ForgeSFTRecipe(ForgeActor, ForgeEngine):
    def __init__(self, config: DictConfig):
        job_config_dict = ForgeJobConfig().to_dict()
        job_config = OmegaConf.merge(job_config_dict, config)

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
        self.metrics_tracker = TrainingMetrics(self._rank, self._size)

        ForgeEngine.__init__(self, job_config)

    async def _setup_metric_logger(self):
        return await get_or_create_metric_logger()

    def _record_batch_metrics(self, data_metrics: list):
        for metric in data_metrics:
            record_metric(metric.key, metric.value, metric.reduction)

    @endpoint
    async def setup(self):
        print(f"[Rank {self._rank}] Starting setup...", flush=True)

        self.rank_should_record_loss = True
        if self.parallel_dims.pp_enabled and not self.pp_has_last_stage:
            self.rank_should_record_loss = False

        print(f"[Rank {self._rank}] Setting up metric logger...", flush=True)
        self.mlogger = await self._setup_metric_logger()

        print(f"[Rank {self._rank}] Setting training datasets...", flush=True)
        train_datasets_config = self.job_config.training.datasets
        self.train_dataloader = setup_dataloader(
            dataset_configs=train_datasets_config,
            hf_assets_path=self.job_config.model.hf_assets_path,
            batch_size=self.job_config.training.local_batch_size,
            parallel_dims=self.parallel_dims,
        )

        print(f"[Rank {self._rank}] Setting eval config...", flush=True)
        eval_config = self.job_config.get("eval", {})
        self.eval_every_n_steps = eval_config.get("eval_every_n_steps")
        max_eval_steps = eval_config.get("max_eval_steps")
        self.max_eval_steps = max_eval_steps if max_eval_steps and max_eval_steps > 0 else None
        self.validation_enabled = self.eval_every_n_steps is not None and self.eval_every_n_steps > 0

        if self.validation_enabled:
            print(f"[Rank {self._rank}] Setting eval datasets...", flush=True)
            self.eval_datasets_config = eval_config.get("datasets", [])
            for i, dataset_config in enumerate(self.eval_datasets_config):
                ds_name = dataset_config.get("dataset_name", i)
                dataloader = setup_dataloader(
                    dataset_configs=[dataset_config],
                    hf_assets_path=self.job_config.model.hf_assets_path,
                    batch_size=self.job_config.training.local_batch_size,
                    parallel_dims=self.parallel_dims,
                )
                self.val_dataloaders[ds_name] = dataloader

        print(f"[Rank {self._rank}] Loading checkpoint...", flush=True)
        self.checkpointer.load()

        print(f"[Rank {self._rank}] Setup complete!", flush=True)

    def batch_generator(
        self,
        data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration:
                print(f"[Rank {self._rank}] Dataloader exhausted, restarting...", flush=True)
                data_iterator = iter(data_iterable)
                batch = next(data_iterator)

            self._record_batch_metrics(batch.pop("metrics", []))

            input_tensor = batch.get("input", batch.get("tokens"))
            labels = batch.get("labels")

            if input_tensor is None:
                raise ValueError(f"Batch must contain 'input' or 'tokens' key. Got keys: {batch.keys()}")
            if labels is None:
                raise ValueError(f"Batch must contain 'labels' key. Got keys: {batch.keys()}")

            ntokens_batch = labels.numel()
            self.ntokens_seen += ntokens_batch
            self.metrics_tracker.ntokens_since_last_log += ntokens_batch
            self.metrics_tracker.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            input_tensor = input_tensor.to(self.device)
            labels = labels.to(self.device)

            yield input_tensor, labels

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
                    "is not available in your installation. Please upgrade torchtitan or disable "
                    "context parallelism in your config."
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
                targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
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

            loss = (
                torch.sum(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor(-1.0, device=self.device)
            )

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
    ) -> tuple[float, float]:
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        accumulated_losses = []
        for _ in range(self.gradient_accumulation_steps):
            inputs, labels = next(data_iterator)

            batch_size = labels.shape[0]
            seq_len = labels.shape[1] if len(labels.shape) > 1 else self.job_config.training.seq_len
            self.metrics_tracker.record_batch(batch_size, seq_len)

            loss = self.forward_backward_step(inputs, labels)
            accumulated_losses.append(loss.detach())

        pp_mesh = get_optional_mesh(self.parallel_dims, "pp")
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=pp_mesh,
            ep_enabled=self.parallel_dims.ep_enabled,
        )

        if hasattr(self.checkpointer, "maybe_wait_for_staging"):
            self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        loss = torch.sum(torch.stack(accumulated_losses))

        if self.parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = get_optional_mesh(self.parallel_dims, "loss")
            if loss_mesh is not None:
                global_avg_loss = dist_utils.dist_mean(loss, loss_mesh)
            else:
                global_avg_loss = loss.item()
        else:
            global_avg_loss = loss.detach().item()

        return global_avg_loss, lr

    def should_continue_training(self) -> bool:
        return self.step < self.num_training_steps

    @endpoint
    async def train(self) -> None:
        print(f"[Rank {self._rank}] Starting training loop at step {self.step + 1}", flush=True)

        self.optimizers.zero_grad()
        self.metrics_tracker.start_training()

        data_iterator = self.batch_generator(self.train_dataloader)
        log_interval = self.job_config.get("metrics", {}).get("log_freq", 10)

        while self.should_continue_training():
            self.step += 1
            self.gc_handler.run(self.step)
            self.metrics_tracker.start_step()

            loss_val, lr = self.train_step(data_iterator)

            if self.rank_should_record_loss:
                self.metrics_tracker.record_loss(
                    loss_val if isinstance(loss_val, float) else loss_val.item()
                )

            step_metrics = self.metrics_tracker.end_step(self.step)

            if self.rank_should_record_loss:
                log_training_metrics(
                    step=self.step,
                    loss=loss_val if isinstance(loss_val, float) else loss_val.item(),
                    metrics=step_metrics,
                    world_size=self._size,
                    learning_rate=lr,
                    ntokens_seen=self.ntokens_seen * self._size,
                )

                if self.step % log_interval == 0 or self.step == 1:
                    print(
                        f"[Rank {self._rank}] Step {self.step}/{self.num_training_steps} | "
                        f"Loss: {loss_val:.4f} | "
                        f"Avg Loss: {step_metrics['running_avg_loss']:.4f} | "
                        f"Tokens/s: {step_metrics['global_tokens_per_second']:.0f} | "
                        f"Total Tokens: {self.ntokens_seen * self._size:,} | "
                        f"LR: {lr:.2e} | "
                        f"Step Time: {step_metrics['step_time_seconds']:.2f}s",
                        flush=True
                    )

            if self.validation_enabled and self.step % self.eval_every_n_steps == 0:
                await self.evaluate()

            self.checkpointer.save(
                curr_step=self.step,
                last_step=self.step == self.num_training_steps,
            )

            if self._rank == 0:
                await self.mlogger.flush.call_one(global_step=self.step)

        total_time = time.time() - self.metrics_tracker.training_start_time
        global_tokens = self.ntokens_seen * self._size
        print(
            f"[Rank {self._rank}] Training complete! "
            f"Total steps: {self.step} | "
            f"Total tokens: {global_tokens:,} | "
            f"Total time: {total_time:.1f}s | "
            f"Avg tokens/s: {global_tokens / total_time:.0f}",
            flush=True
        )

        if self.validation_enabled:
            logger.info("Running final evaluation at end of training...")
            await self.evaluate()

    @torch.no_grad()
    async def evaluate(self) -> None:
        for model_part in self.model_parts:
            model_part.eval()

        dp_mesh = None
        if self.parallel_dims is not None and self.parallel_dims.dp_enabled:
            dp_mesh = self.parallel_dims.world_mesh["dp"]

        maybe_no_grad = (
            contextlib.nullcontext()
            if self.parallel_dims.pp_enabled
            else torch.no_grad()
        )

        all_dataset_results = []

        for dataset_name, val_dataloader in self.val_dataloaders.items():
            print(f"[Rank {self._rank}] Evaluating dataset: {dataset_name}", flush=True)

            total_loss = torch.tensor(0.0, device=self.device)
            total_tokens = 0
            num_steps = 0

            batch_iter = StopAfterOneEpoch(
                iter=iter(val_dataloader),
                device=self.device,
                dp_mesh=dp_mesh,
            )
            eval_start_time = time.time()

            with maybe_no_grad:
                for batch in batch_iter:
                    if self.max_eval_steps is not None and num_steps >= self.max_eval_steps:
                        break

                    self._record_batch_metrics(batch.pop("metrics", []))

                    labels = batch.get("labels")
                    input_tensor = batch.get("input", batch.get("tokens"))

                    if input_tensor is None or labels is None:
                        continue

                    input_tensor = input_tensor.to(self.device)
                    labels = labels.to(self.device)

                    batch_size = labels.shape[0]
                    seq_len = labels.shape[1] if len(labels.shape) > 1 else self.job_config.training.seq_len
                    total_tokens += batch_size * seq_len

                    loss = self.forward_backward_step(input_tensor, labels, skip_backward=True)
                    total_loss += loss
                    num_steps += 1

            eval_time = time.time() - eval_start_time
            avg_loss = (total_loss / max(num_steps, 1)).item()
            tokens_per_second = total_tokens / eval_time if eval_time > 0 else 0

            result = {
                "dataset_name": dataset_name,
                "avg_loss": avg_loss,
                "num_steps": num_steps,
                "total_tokens": total_tokens,
                "eval_time": eval_time,
                "tokens_per_second": tokens_per_second,
            }
            all_dataset_results.append(result)

            print(
                f"[Rank {self._rank}] Eval {dataset_name} complete | "
                f"Steps: {num_steps} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Tokens: {total_tokens:,} | "
                f"Time: {eval_time:.1f}s | "
                f"Tokens/s: {tokens_per_second:.0f}",
                flush=True
            )

            if self.rank_should_record_loss:
                log_eval_metrics(
                    dataset_name=dataset_name,
                    avg_loss=avg_loss,
                    num_steps=num_steps,
                    total_tokens=total_tokens * self._size,
                    tokens_per_second=tokens_per_second * self._size,
                    eval_time=eval_time,
                )

        if self.rank_should_record_loss and len(all_dataset_results) > 1:
            losses = [r["avg_loss"] for r in all_dataset_results]
            steps = [r["num_steps"] for r in all_dataset_results]

            macro_avg_loss = sum(losses) / len(losses)
            record_metric("eval/macro_avg_loss", macro_avg_loss, Reduce.MEAN)

            total_steps = sum(steps)
            micro_avg_loss = sum(l * s for l, s in zip(losses, steps)) / total_steps
            record_metric("eval/micro_avg_loss", micro_avg_loss, Reduce.MEAN)

        for model_part in self.model_parts:
            model_part.train()

    @endpoint
    async def cleanup(self) -> None:
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if getattr(self, "mlogger", None):
            await self.mlogger.shutdown.call_one()

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict.get("ntokens_seen", 0)

    def __repr__(self) -> str:
        return "ForgeSFTRecipe"
import asyncio
import logging
import sys

import torch
import torchtitan.experiments.forge.train_spec as forge_train_spec
from forge.controller import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.observability import get_or_create_metric_logger
from forge.types import ProvisionerConfig, LauncherConfig, ServiceConfig, ProcessConfig
from forge.util.config import parse
from monarch.actor import current_rank, current_size, endpoint
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainingMetrics:
    """Helper class to track and compute training metrics."""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.reset()
    
    def reset(self):
        self.total_tokens_processed = 0
        self.total_samples_processed = 0
        self.step_start_time = None
        self.training_start_time = None
        self.step_tokens = 0
        self.step_samples = 0
        self.cumulative_loss = 0.0
        self.loss_count = 0
    
    def start_training(self):
        self.training_start_time = time.time()
    
    def start_step(self):
        self.step_start_time = time.time()
        self.step_tokens = 0
        self.step_samples = 0
    
    def record_batch(self, batch_size: int, seq_len: int, num_valid_tokens: int = None):
        if num_valid_tokens is None:
            num_valid_tokens = batch_size * seq_len
        self.step_tokens += num_valid_tokens
        self.step_samples += batch_size
    
    def record_loss(self, loss: float):
        self.cumulative_loss += loss
        self.loss_count += 1
    
    def end_step(self, current_step: int) -> dict:
        step_time = time.time() - self.step_start_time
        total_time = time.time() - self.training_start_time
        
        self.total_tokens_processed += self.step_tokens
        self.total_samples_processed += self.step_samples
        tokens_per_second = self.step_tokens / step_time if step_time > 0 else 0
        avg_tokens_per_second = self.total_tokens_processed / total_time if total_time > 0 else 0
        samples_per_second = self.step_samples / step_time if step_time > 0 else 0
        global_tokens_per_second = tokens_per_second * self.world_size
        global_avg_tokens_per_second = avg_tokens_per_second * self.world_size
        avg_loss = self.cumulative_loss / self.loss_count if self.loss_count > 0 else 0.0
        
        return {
            "step": current_step,
            "step_time_seconds": step_time,
            "total_time_seconds": total_time,
            "tokens_this_step": self.step_tokens,
            "total_tokens_processed": self.total_tokens_processed,
            "global_total_tokens": self.total_tokens_processed * self.world_size,
            "samples_this_step": self.step_samples,
            "total_samples_processed": self.total_samples_processed,
            "tokens_per_second": tokens_per_second,
            "global_tokens_per_second": global_tokens_per_second,
            "avg_tokens_per_second": avg_tokens_per_second,
            "global_avg_tokens_per_second": global_avg_tokens_per_second,
            "samples_per_second": samples_per_second,
            "running_avg_loss": avg_loss,
        }


class ForgeSFTRecipe(ForgeActor, ForgeEngine):
    job_config: ForgeJobConfig
    train_spec: forge_train_spec.ForgeTrainSpec
    parallel_dims: ParallelDims
    model: list[nn.Module]
    loss_fn: LossFunction
    optimizer: OptimizersContainer
    lr_scheduler: LRSchedulersContainer
    checkpointer: Checkpointer
    tokenizer: Tokenizer
    train_dataloader: Dataloader
    metric_logger: MetricLogger
    profiler: Profiler
    device: torch.device
    step: int

    def __init__(self, config: DictConfig):
        job_config = ForgeJobConfig().to_dict()
        job_config = OmegaConf.merge(job_config, config)

        self.current_step = 0 # todo: fix this, parse the checkpoint directory to find last saved step
        self.num_training_steps = job_config.training.steps
        self.gradient_accumulation_steps = 1
        self._rank = current_rank().rank
        self._size = math.prod(current_size().values())
        
        self.validation_enabled = False
        self.val_dataloaders = {}
        self.eval_every_n_steps = None
        self.max_eval_steps = None
        self.mlogger = None
        self.metrics_tracker = TrainingMetrics(self._rank, self._size)
        
        super().__init__(job_config)

    async def setup_metric_logger(self):
        mlogger = await get_or_create_metric_logger()
        return mlogger

    def record_batch_metrics(self, data_metrics: list):
        for metric in data_metrics:
            record_metric(metric.key, metric.value, metric.reduction)

    @endpoint
    async def setup(self):
        print(f"[Rank {self._rank}] Starting setup...", flush=True)
        
        if self.job_config.training.compile:
            raise ValueError(
                "training.compile=True is not currently supported. "
                "Compile is only supported with flex attention enabled, which requires PyTorch nightly. "
                "Please set training.compile=false in your config."
            )

        self.rank_should_record_loss = True
        if hasattr(self, "pp_has_last_stage") and not self.pp_has_last_stage:
            self.rank_should_record_loss = False

        print(f"[Rank {self._rank}] Setting up metric logger...", flush=True)
        self.mlogger = await self.setup_metric_logger()

        print(f"[Rank {self._rank}] Setting training datasets...", flush=True)
        logger.info("Setting training datasets")
        train_datasets_config = self.job_config.training.datasets

        self.train_dataloader = self.setup_data(train_datasets_config)

        print(f"[Rank {self._rank}] Setting eval config...", flush=True)
        eval_config = self.job_config["eval"]
        self.val_dataloaders = {}
        self.eval_every_n_steps = eval_config["eval_every_n_steps"]
        max_eval_steps = eval_config["max_eval_steps"]
        self.max_eval_steps = (
            max_eval_steps if max_eval_steps and max_eval_steps > 0 else None
        )
        self.validation_enabled = (
            self.eval_every_n_steps is not None and self.eval_every_n_steps > 0
        )
        
        if self.validation_enabled:
            print(f"[Rank {self._rank}] Setting eval datasets...", flush=True)
            logger.info("Setting eval datasets")
            self.eval_datasets_config = eval_config.datasets

            for i, dataset_config in enumerate(self.eval_datasets_config):
                ds_name = dataset_config.get("dataset_name", i)
                dataloader = self.setup_data([dataset_config])
                self.val_dataloaders[ds_name] = dataloader

        print(f"[Rank {self._rank}] Loading checkpoint...", flush=True)
        self.checkpointer.load(step=self.current_step)
        
        print(f"[Rank {self._rank}] Setup complete!", flush=True)

    def setup_data(self, dataset_configs: list[dict]) -> StatefulDataLoader:
        if len(dataset_configs) > 1:
            raise ValueError(
                f"Multiple training datasets not supported yet. "
                f"Got {len(dataset_configs)} datasets. "
            )

        dataset_config = dataset_configs[0]

        tokenizer = HuggingFaceModelTokenizer(
            tokenizer_json_path=os.path.join(
                self.job_config.model.hf_assets_path, "tokenizer.json"
            ),
            tokenizer_config_json_path=os.path.join(
                self.job_config.model.hf_assets_path, "tokenizer_config.json"
            ),
            generation_config_path=os.path.join(
                self.job_config.model.hf_assets_path, "generation_config.json"
            ),
            chat_template_path=(
                path
                if os.path.exists(
                    path := os.path.join(
                        self.job_config.model.hf_assets_path, "chat_template.jinja"
                    )
                )
                else None
            ),
            max_seq_len=self.job_config.training.seq_len,
        )

        dp_mesh = None
        if self.parallel_dims is not None and self.parallel_dims.dp_enabled:
            dp_mesh = self.parallel_dims.world_mesh.get_group("dp")

        dataset = sft_iterable_dataset(
            model_transform=tokenizer,
            message_transform=AlpacaToMessages(),
            dp_mesh=dp_mesh,
            **dataset_config,
        )

        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=self.job_config.training.local_batch_size,
            collate_fn=collate_padded,
        )

        return dataloader

    def forward_backward(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        skip_backward: bool = False,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs = input_dict["tokens"]
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(inputs, target=targets, losses=losses)
                else:
                    self.pp_schedule.step(target=targets, losses=losses)

            loss = (
                torch.sum(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor(-1.0, device=self.device)
            )

            if skip_backward:
                loss = loss.detach()

        else:
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                del pred

                if not skip_backward:
                    loss.backward()

        return loss

    def train_step(self, batch: dict, seq_len: int) -> float:
        # TODO
        # with GradientAccumulation(
        #     self.gradient_accumulation_steps,
        #     self.model,
        #     self.data_parallel_size,
        # ) as grad_acc:
        parallel_dims = self.parallel_dims
        labels = batch.pop("labels")
        batch_size = labels.shape[0]
        
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            num_valid_tokens = attention_mask.sum().item()
        else:
            num_valid_tokens = batch_size * seq_len
        
        self.metrics_tracker.record_batch(batch_size, seq_len, int(num_valid_tokens))
        
        loss = self.forward_backward(batch, labels)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=(
                parallel_dims.world_mesh["pp"] if parallel_dims.pp_enabled else None
            ),
            ep_enabled=parallel_dims.ep_enabled,
        )

        loss_val = loss.item()
        
        if self.rank_should_record_loss:
            self.metrics_tracker.record_loss(loss_val)

        self.optimizers.step()
        self.lr_schedulers.step()
        
        return loss_val

    def log_training_metrics(self, step: int, loss: float, metrics: dict):
        record_metric("train/step", step, Reduce.MAX)
        record_metric("train/loss", loss, Reduce.MEAN)
        record_metric("train/running_avg_loss", metrics["running_avg_loss"], Reduce.MEAN)
        record_metric("train/step_time_seconds", metrics["step_time_seconds"], Reduce.MEAN)
        record_metric("train/total_time_seconds", metrics["total_time_seconds"], Reduce.MAX)
        record_metric("train/tokens_this_step", metrics["tokens_this_step"], Reduce.SUM)
        record_metric("train/total_tokens_processed", metrics["global_total_tokens"], Reduce.MAX)
        record_metric("train/samples_this_step", metrics["samples_this_step"], Reduce.SUM)
        record_metric("train/total_samples_processed", metrics["total_samples_processed"] * self._size, Reduce.MAX)
        record_metric("train/tokens_per_second_per_gpu", metrics["tokens_per_second"], Reduce.MEAN)
        record_metric("train/global_tokens_per_second", metrics["global_tokens_per_second"], Reduce.MAX)
        record_metric("train/avg_tokens_per_second_per_gpu", metrics["avg_tokens_per_second"], Reduce.MEAN)
        record_metric("train/global_avg_tokens_per_second", metrics["global_avg_tokens_per_second"], Reduce.MAX)
        record_metric("train/samples_per_second", metrics["samples_per_second"], Reduce.MEAN)
        current_lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        record_metric("train/learning_rate", current_lr, Reduce.MEAN)
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            record_metric("system/gpu_memory_allocated_gb", gpu_mem_allocated, Reduce.MAX)
            record_metric("system/gpu_memory_reserved_gb", gpu_mem_reserved, Reduce.MAX)

    @endpoint
    async def train(self) -> None:
        print(f"[Rank {self._rank}] Starting training loop", flush=True)
        dataloader = iter(self.train_dataloader)
        self.optimizers.zero_grad()
        
        self.metrics_tracker.start_training()
        seq_len = self.job_config.training.seq_len
        log_interval = 10

        while self.current_step < self.num_training_steps:
            self.metrics_tracker.start_step()
            
            try:
                batch = next(dataloader)
            except StopIteration:
                print(f"[Rank {self._rank}] Dataloader exhausted, restarting...", flush=True)
                dataloader = iter(self.train_dataloader)
                batch = next(dataloader)

            self.record_batch_metrics(batch.pop("metrics", []))

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda")

            loss_val = self.train_step(batch, seq_len)
            
            self.current_step += 1
            step_metrics = self.metrics_tracker.end_step(self.current_step)

            if self.rank_should_record_loss:
                self.log_training_metrics(self.current_step, loss_val, step_metrics)
                
                if self.current_step % log_interval == 0 or self.current_step == 1:
                    print(
                        f"[Rank {self._rank}] Step {self.current_step}/{self.num_training_steps} | "
                        f"Loss: {loss_val:.4f} | "
                        f"Avg Loss: {step_metrics['running_avg_loss']:.4f} | "
                        f"Tokens/s: {step_metrics['global_tokens_per_second']:.0f} | "
                        f"Total Tokens: {step_metrics['global_total_tokens']:,} | "
                        f"Step Time: {step_metrics['step_time_seconds']:.2f}s | "
                        f"Total Time: {step_metrics['total_time_seconds']:.1f}s",
                        flush=True
                    )

            if (
                self.validation_enabled
                and self.current_step % self.eval_every_n_steps == 0
            ):
                await self.evaluate()

            self.checkpointer.save(
                curr_step=self.current_step,
                last_step=self.current_step == self.num_training_steps,
            )

            if self._rank == 0:
                await self.mlogger.flush.call_one(global_step=self.current_step)

        total_time = time.time() - self.metrics_tracker.training_start_time
        print(
            f"[Rank {self._rank}] Training complete! "
            f"Total steps: {self.current_step} | "
            f"Total tokens: {self.metrics_tracker.total_tokens_processed * self._size:,} | "
            f"Total time: {total_time:.1f}s | "
            f"Avg tokens/s: {self.metrics_tracker.total_tokens_processed * self._size / total_time:.0f}",
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
            dp_mesh = self.parallel_dims.world_mesh.get_group("dp")

        maybe_no_grad = (
            contextlib.nullcontext()
            if self.parallel_dims.pp_enabled
            else torch.no_grad()
        )

        all_dataset_losses = []
        all_dataset_steps = []
        
        for dataset_name, val_dataloader in self.val_dataloaders.items():
            print(f"[Rank {self._rank}] Evaluating dataset: {dataset_name}", flush=True)
            logger.info(f"=====Evaluating dataset: {dataset_name}=====")

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
                    if (
                        self.max_eval_steps is not None
                        and num_steps >= self.max_eval_steps
                    ):
                        logger.info(
                            f"[{dataset_name}] Reached max_eval_steps cap of {self.max_eval_steps}"
                        )
                        break

                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)

                    labels = batch.pop("labels")
                    batch_size = labels.shape[0]
                    seq_len = labels.shape[1] if len(labels.shape) > 1 else self.job_config.training.seq_len
                    
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        num_valid_tokens = attention_mask.sum().item()
                    else:
                        num_valid_tokens = batch_size * seq_len
                    
                    total_tokens += int(num_valid_tokens)
                    
                    loss = self.forward_backward(batch, labels, skip_backward=True)
                    total_loss += loss
                    num_steps += 1

            eval_time = time.time() - eval_start_time
            avg_loss = (total_loss / max(num_steps, 1)).item()
            tokens_per_second = total_tokens / eval_time if eval_time > 0 else 0
            
            all_dataset_losses.append(avg_loss)
            all_dataset_steps.append(num_steps)
            
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
                record_metric(f"eval/{dataset_name}/loss", avg_loss, Reduce.MEAN)
                record_metric(f"eval/{dataset_name}/num_steps", num_steps, Reduce.MAX)
                record_metric(f"eval/{dataset_name}/total_tokens", total_tokens * self._size, Reduce.MAX)
                record_metric(f"eval/{dataset_name}/tokens_per_second", tokens_per_second * self._size, Reduce.MAX)
                record_metric(f"eval/{dataset_name}/eval_time_seconds", eval_time, Reduce.MAX)

        if self.rank_should_record_loss and len(all_dataset_losses) > 1:
            macro_avg_loss = sum(all_dataset_losses) / len(all_dataset_losses)
            record_metric("eval/macro_avg_loss", macro_avg_loss, Reduce.MEAN)

            total_steps = sum(all_dataset_steps)
            micro_avg_loss = (
                sum(
                    loss * steps
                    for loss, steps in zip(all_dataset_losses, all_dataset_steps)
                )
                / total_steps
            )
            record_metric("eval/micro_avg_loss", micro_avg_loss, Reduce.MEAN)

        for model_part in self.model_parts:
            model_part.train()

        logger.info("==Evaluation complete==")

    @endpoint
    async def cleanup(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if getattr(self, "mlogger", None):
            await self.mlogger.shutdown.call_one()

    def __repr__(self) -> str:
        return "Trainer"


async def run(cfg: DictConfig) -> None:
    logger.info("Spawning recipe...")

    if cfg.get("provisioner", None) is not None:
        provisioner_dict = OmegaConf.to_container(cfg.provisioner, resolve=True)

        services_dict = {}
        if cfg.get("services"):
            for name, svc_cfg in cfg.services.items():
                svc_container = OmegaConf.to_container(svc_cfg, resolve=True)
                services_dict[name] = ServiceConfig(**svc_container)

        actors_dict = {}
        if cfg.get("actors"):
            for name, actor_cfg in cfg.actors.items():
                actor_container = OmegaConf.to_container(actor_cfg, resolve=True)
                actors_dict[name] = ProcessConfig(**actor_container)

        launcher_config = LauncherConfig(
            launcher=provisioner_dict.get("launcher"),
            job_name=provisioner_dict.get("job_name", ""),
            services=services_dict,
            actors=actors_dict,
            slurm_args=provisioner_dict.get("slurm_args", {}),
            cpus_per_task=provisioner_dict.get("cpus_per_task"),
            mem=provisioner_dict.get("mem"),
            gpus_per_node=provisioner_dict.get("gpus_per_node", 8),
        )

        await init_provisioner(
            ProvisionerConfig(launcher_config=launcher_config)
        )
    else:
        await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    actor_cfg = OmegaConf.to_container(cfg.actors.sft_trainer, resolve=True)
    recipe = await ForgeSFTRecipe.options(**actor_cfg).as_actor(cfg)

    logger.info("Created recipe, running setup.")
    await recipe.setup.call()

    logger.info("Recipe has been setup. Training now.")
    await recipe.train.call()

    logger.info("Done training. Clean up")
    await recipe.cleanup.call()

    await recipe.mesh.stop()
    await shutdown()
    logger.info("All done!")


@parse
def main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(main())
import time
from collections import deque
from typing import Optional

import torch
from forge.observability import record_metric, Reduce


class TrainingMetrics:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.reset()

    def reset(self):
        self.total_tokens_processed = 0
        self.total_samples_processed = 0
        self.step_start_time = 0.0
        self.training_start_time = 0.0
        self.step_tokens = 0
        self.step_samples = 0
        self.cumulative_loss = 0.0
        self.loss_count = 0
        self.ntokens_since_last_log = 0
        self.data_loading_times: deque[float] = deque(maxlen=100)

    def start_training(self):
        self.training_start_time = time.time()

    def start_step(self):
        self.step_start_time = time.time()
        self.step_tokens = 0
        self.step_samples = 0

    def record_batch(self, batch_size: int, seq_len: int, num_valid_tokens: Optional[int] = None):
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

        avg_data_load_time = (
            sum(self.data_loading_times) / len(self.data_loading_times)
            if self.data_loading_times
            else 0.0
        )

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
            "avg_data_load_time": avg_data_load_time,
        }


def log_training_metrics(
    step: int,
    loss: float,
    metrics: dict,
    world_size: int,
    learning_rate: float,
    ntokens_seen: int,
):
    record_metric("train/step", step, Reduce.MAX)
    record_metric("train/loss", loss, Reduce.MEAN)
    record_metric("train/running_avg_loss", metrics["running_avg_loss"], Reduce.MEAN)
    record_metric("train/step_time_seconds", metrics["step_time_seconds"], Reduce.MEAN)
    record_metric("train/total_time_seconds", metrics["total_time_seconds"], Reduce.MAX)
    record_metric("train/tokens_this_step", metrics["tokens_this_step"], Reduce.SUM)
    record_metric("train/ntokens_seen", ntokens_seen, Reduce.MAX)
    record_metric("train/samples_this_step", metrics["samples_this_step"], Reduce.SUM)
    record_metric("train/total_samples_processed", metrics["total_samples_processed"] * world_size, Reduce.MAX)
    record_metric("train/tokens_per_second_per_gpu", metrics["tokens_per_second"], Reduce.MEAN)
    record_metric("train/global_tokens_per_second", metrics["global_tokens_per_second"], Reduce.MAX)
    record_metric("train/avg_tokens_per_second_per_gpu", metrics["avg_tokens_per_second"], Reduce.MEAN)
    record_metric("train/global_avg_tokens_per_second", metrics["global_avg_tokens_per_second"], Reduce.MAX)
    record_metric("train/samples_per_second", metrics["samples_per_second"], Reduce.MEAN)
    record_metric("train/learning_rate", learning_rate, Reduce.MEAN)
    record_metric("train/avg_data_load_time", metrics["avg_data_load_time"], Reduce.MEAN)

    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        record_metric("system/gpu_memory_allocated_gb", gpu_mem_allocated, Reduce.MAX)
        record_metric("system/gpu_memory_reserved_gb", gpu_mem_reserved, Reduce.MAX)


def log_eval_metrics(
    dataset_name: str,
    avg_loss: float,
    num_steps: int,
    total_tokens: int,
    tokens_per_second: float,
    eval_time: float,
):
    record_metric(f"eval/{dataset_name}/loss", avg_loss, Reduce.MEAN)
    record_metric(f"eval/{dataset_name}/num_steps", num_steps, Reduce.MAX)
    record_metric(f"eval/{dataset_name}/total_tokens", total_tokens, Reduce.MAX)
    record_metric(f"eval/{dataset_name}/tokens_per_second", tokens_per_second, Reduce.MAX)
    record_metric(f"eval/{dataset_name}/eval_time_seconds", eval_time, Reduce.MAX)
    record_metric(f"eval/{dataset_name}/perplexity", torch.exp(torch.tensor(avg_loss)).item(), Reduce.MEAN)
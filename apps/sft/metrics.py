import time
from collections import deque
from dataclasses import dataclass, field

import torch
from forge.observability import record_metric, Reduce
from torchtitan.tools.utils import device_module, device_type, get_peak_flops


@dataclass
class MetricsTracker:
    """Tracks training metrics, logs to Forge's metric system, and prints to console."""
    rank: int
    world_size: int
    num_flops_per_token: int = 0
    log_freq: int = 10

    step_start_time: float = field(default=0.0, init=False)
    training_start_time: float = field(default=0.0, init=False)

    ntokens_since_last_log: int = field(default=0, init=False)
    cumulative_loss: float = field(default=0.0, init=False)
    loss_count: int = field(default=0, init=False)

    data_loading_times: deque = field(default_factory=lambda: deque(maxlen=1000), init=False)
    time_last_log: float = field(default_factory=time.perf_counter, init=False)

    _device_name: str = field(default="", init=False)
    _gpu_peak_flops: float = field(default=0, init=False)
    _device_capacity: int = field(default=0, init=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            device = torch.device(f"{device_type}:{device_module.current_device()}")  # type: ignore
            self._device_name = device_module.get_device_name(device) # type: ignore
            self._gpu_peak_flops = get_peak_flops(self._device_name)  # type: ignore
            self._device_capacity = device_module.get_device_properties(device).total_memory  # type: ignore
            device_module.reset_peak_memory_stats() # type: ignore

    def start_training(self):
        self.training_start_time = time.time()
        self.time_last_log = time.perf_counter()

    def start_step(self):
        self.step_start_time = time.time()

    def record_batch(self, ntokens: int, data_load_time: float):
        self.ntokens_since_last_log += ntokens
        self.data_loading_times.append(data_load_time)

    def record_loss(self, loss: float):
        self.cumulative_loss += loss
        self.loss_count += 1

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.log_freq == 0

    def get_step_metrics(self, step: int) -> dict:
        """Compute metrics for the current step."""
        step_time = time.time() - self.step_start_time if self.step_start_time else 0
        total_time = time.time() - self.training_start_time if self.training_start_time else 0
        time_delta = time.perf_counter() - self.time_last_log

        running_avg_loss = self.cumulative_loss / self.loss_count if self.loss_count > 0 else 0.0

        tps = self.ntokens_since_last_log / time_delta if time_delta > 0 else 0
        global_tps = tps * self.world_size

        if self.num_flops_per_token > 0 and self._gpu_peak_flops > 0:
            mfu = 100 * self.num_flops_per_token * tps / self._gpu_peak_flops
            tflops = self.num_flops_per_token * tps / 1e12
        else:
            mfu = 0.0
            tflops = 0.0

        avg_data_load_time = (
            sum(self.data_loading_times) / len(self.data_loading_times)
            if self.data_loading_times else 0.0
        )
        data_load_pct = (
            100 * sum(self.data_loading_times) / time_delta
            if time_delta > 0 and self.data_loading_times else 0.0
        )

        return {
            "step": step,
            "step_time_seconds": step_time,
            "total_time_seconds": total_time,
            "running_avg_loss": running_avg_loss,
            "tps": tps,
            "global_tps": global_tps,
            "tflops": tflops,
            "mfu": mfu,
            "avg_data_load_time": avg_data_load_time,
            "data_load_pct": data_load_pct,
        }

    def log_to_forge(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        ntokens_seen: int,
        metrics: dict,
    ):
        """Log metrics via Forge's record_metric for distributed reduction."""
        record_metric("train/loss", loss, Reduce.MEAN)
        record_metric("train/running_avg_loss", metrics["running_avg_loss"], Reduce.MEAN)
        record_metric("train/grad_norm", grad_norm, Reduce.MEAN)
        record_metric("train/learning_rate", learning_rate, Reduce.MEAN)
        record_metric("train/ntokens_seen", ntokens_seen, Reduce.MAX)
        record_metric("train/tps", metrics["tps"], Reduce.MEAN)
        record_metric("train/global_tps", metrics["global_tps"], Reduce.MAX)
        record_metric("train/tflops", metrics["tflops"], Reduce.MEAN)
        record_metric("train/mfu_pct", metrics["mfu"], Reduce.MEAN)
        record_metric("train/step_time_s", metrics["step_time_seconds"], Reduce.MEAN)
        record_metric("train/data_load_time_s", metrics["avg_data_load_time"], Reduce.MEAN)
        record_metric("train/data_load_pct", metrics["data_load_pct"], Reduce.MEAN)

        if torch.cuda.is_available():
            self._log_memory_to_forge("train")

    def log_to_console(
        self,
        step: int,
        num_steps: int,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        ntokens_seen: int,
        metrics: dict,
    ):
        """Print formatted metrics to console."""
        print(
            f"[Rank {self.rank}] Step {step}/{num_steps} | "
            f"Loss: {loss:.4f} | "
            f"Avg Loss: {metrics['running_avg_loss']:.4f} | "
            f"Grad Norm: {grad_norm:.4f} | "
            f"LR: {learning_rate:.2e} | "
            f"TPS: {metrics['global_tps']:,.0f} | "
            f"TFLOPS: {metrics['tflops']:.1f} | "
            f"MFU: {metrics['mfu']:.1f}% | "
            f"Tokens: {ntokens_seen:,} | "
            f"Step Time: {metrics['step_time_seconds']:.2f}s",
            flush=True,
        )

    def reset_logging_window(self):
        """Reset metrics that accumulate over the logging window."""
        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        if torch.cuda.is_available():
            device_module.reset_peak_memory_stats()

    def _log_memory_to_forge(self, prefix: str):
        device = torch.device(f"{device_type}:{device_module.current_device()}")
        mem_stats = device_module.memory_stats(device)

        max_active = mem_stats.get("active_bytes.all.peak", 0)
        max_reserved = mem_stats.get("reserved_bytes.all.peak", 0)
        num_retries = mem_stats.get("num_alloc_retries", 0)
        num_ooms = mem_stats.get("num_ooms", 0)

        max_active_gib = max_active / (1024 ** 3)
        max_reserved_gib = max_reserved / (1024 ** 3)

        if self._device_capacity > 0:
            max_active_pct = 100 * max_active / self._device_capacity
            max_reserved_pct = 100 * max_reserved / self._device_capacity
        else:
            max_active_pct = 0.0
            max_reserved_pct = 0.0

        record_metric(f"{prefix}/memory_active_gib", max_active_gib, Reduce.MAX)
        record_metric(f"{prefix}/memory_active_pct", max_active_pct, Reduce.MAX)
        record_metric(f"{prefix}/memory_reserved_gib", max_reserved_gib, Reduce.MAX)
        record_metric(f"{prefix}/memory_reserved_pct", max_reserved_pct, Reduce.MAX)
        record_metric(f"{prefix}/memory_alloc_retries", num_retries, Reduce.SUM)
        record_metric(f"{prefix}/memory_ooms", num_ooms, Reduce.SUM)

    def log_eval_to_forge(
        self,
        dataset_name: str,
        loss: float,
        ntokens: int,
        eval_time: float,
    ):
        """Log evaluation metrics via Forge."""
        tps = ntokens / eval_time if eval_time > 0 else 0
        perplexity = torch.exp(torch.tensor(loss)).item()

        record_metric(f"eval/{dataset_name}/loss", loss, Reduce.MEAN)
        record_metric(f"eval/{dataset_name}/perplexity", perplexity, Reduce.MEAN)
        record_metric(f"eval/{dataset_name}/tps", tps * self.world_size, Reduce.MAX)
        record_metric(f"eval/{dataset_name}/eval_time_s", eval_time, Reduce.MAX)
        record_metric(f"eval/{dataset_name}/ntokens", ntokens * self.world_size, Reduce.MAX)

        if torch.cuda.is_available():
            self._log_memory_to_forge(f"eval/{dataset_name}")

    def log_training_complete(self, ntokens_seen: int):
        """Print training completion summary."""
        total_time = time.time() - self.training_start_time if self.training_start_time else 0
        global_tokens = ntokens_seen * self.world_size
        avg_tps = global_tokens / total_time if total_time > 0 else 0

        print(
            f"[Rank {self.rank}] Training complete! "
            f"Total tokens: {global_tokens:,} | "
            f"Total time: {total_time:.1f}s | "
            f"Avg TPS: {avg_tps:,.0f}",
            flush=True,
        )
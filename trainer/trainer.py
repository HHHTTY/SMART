import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
try:
    from pytorch_lightning.loggers import TensorBoardLogger
except (ImportError, ModuleNotFoundError):
    TensorBoardLogger = None


class TrainingTelemetryCallback(Callback):
    """Print low-overhead progress and GPU memory telemetry to rjob logs."""

    def __init__(self, every_n_batches: int = 20):
        super().__init__()
        self.every_n_batches = max(1, int(every_n_batches))
        self._batch_count = 0
        self._last_report_count = 0
        self._last_report_time: Optional[float] = None

    def on_train_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        self._batch_count = 0
        self._last_report_count = 0
        self._last_report_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print("[telemetry] training_started", flush=True)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch
        self._batch_count += 1
        if self._batch_count != 1 and self._batch_count % self.every_n_batches != 0:
            return

        now = time.perf_counter()
        interval = now - self._last_report_time if self._last_report_time is not None else 0.0
        batches = self._batch_count - self._last_report_count
        batches_per_second = batches / interval if interval > 0 else 0.0
        self._last_report_time = now
        self._last_report_count = self._batch_count

        loss = None
        if torch.is_tensor(outputs):
            loss = outputs.detach().float().mean().item()
        elif isinstance(outputs, dict) and torch.is_tensor(outputs.get("loss")):
            loss = outputs["loss"].detach().float().mean().item()

        memory = "gpu=unavailable"
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            memory = (
                f"gpu_allocated_gib={allocated:.2f} "
                f"gpu_reserved_gib={reserved:.2f} "
                f"gpu_peak_allocated_gib={peak_allocated:.2f} "
                f"gpu_peak_reserved_gib={peak_reserved:.2f}"
            )

        loss_text = f"loss={loss:.6f}" if loss is not None else "loss=unavailable"
        print(
            f"[telemetry] epoch={trainer.current_epoch} batch_idx={batch_idx} "
            f"batch={self._batch_count} global_step={trainer.global_step} "
            f"{loss_text} batches_per_sec={batches_per_second:.3f} {memory}",
            flush=True,
        )


class ValidationMetricsCallback(Callback):
    """Persist validation metrics so remote runs can be audited reliably.

    Lightning's logger files are often worker-local on the training cluster.
    This callback writes a compact, rank-zero JSONL stream in the run
    directory, after the validation loop has finished and ``callback_metrics``
    contains the epoch aggregates.
    """

    DEFAULT_METRICS = (
        "val_loss",
        "val_token_acc",
        "val_molecular_accuracy",
        "val_top5",
        "val_top10",
    )

    def __init__(
        self,
        output_path: Optional[Union[str, Path]],
        metric_names: Optional[tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        self.output_path = Path(output_path) if output_path else None
        self.metric_names = metric_names or self.DEFAULT_METRICS
        self._written_keys: set[tuple[int, int]] = set()
        if self.output_path is not None and self.output_path.is_file():
            try:
                with self.output_path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        record = json.loads(line)
                        self._written_keys.add(
                            (int(record["epoch"]), int(record["global_step"]))
                        )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                # A truncated final line should not prevent a resumed run from
                # continuing; the next validation result will be appended.
                pass

    @staticmethod
    def _scalar(value: Any) -> Optional[float]:
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.detach().float().cpu().item()
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _is_global_zero(self, trainer: Trainer) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def on_validation_end(
        self, trainer: Trainer, pl_module: torch.nn.Module
    ) -> None:
        del pl_module
        # Lightning runs a short validation sanity check before epoch 0. It is
        # not a training result and must not be mistaken for one in the curve.
        if getattr(trainer, "sanity_checking", False):
            return
        if self.output_path is None or not self._is_global_zero(trainer):
            return

        metrics: Dict[str, float] = {}
        callback_metrics = getattr(trainer, "callback_metrics", {}) or {}
        for name in self.metric_names:
            value = self._scalar(callback_metrics.get(name))
            if value is not None:
                metrics[name] = value
        # Keep any additional scalar validation metrics (for example the MS
        # retrieval diagnostics) without making them required for generation
        # only jobs.
        for name, value in callback_metrics.items():
            if str(name).startswith("val_") and name not in metrics:
                scalar = self._scalar(value)
                if scalar is not None:
                    metrics[str(name)] = scalar
        if not metrics:
            return

        epoch = int(getattr(trainer, "current_epoch", 0))
        global_step = int(getattr(trainer, "global_step", 0))
        key = (epoch, global_step)
        if key in self._written_keys:
            return
        self._written_keys.add(key)

        record: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            **metrics,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        summary = " ".join(
            f"{name}={value:.6f}" for name, value in metrics.items()
        )
        print(
            f"[validation-summary] epoch={epoch} global_step={global_step} {summary}",
            flush=True,
        )


def build_trainer(
    model_type: str,
    log_dir: str,
    task: str,
    epochs: int,
    acc_batches: int = 8,
    clip_grad: float = 1.0,
    limit_train_batches: Optional[Union[int, float]] = None,
    limit_val_batches: float = 5.0,
    checkpoint_monitor: str = "val_molecular_accuracy",
    val_check_interval: Optional[Union[int, float]] = None,
    early_stopping_patience: Optional[int] = None,
    save_checkpoints: str = "best_5",
    early_stopping_delta: Optional[float] = None,
    early_stopping_len_set_sel: Optional[bool] = False,
    update_dataloaders: Optional[bool] = False,
    precision: Optional[str] = None,
    check_val_every_n_epoch: int = 1,
    checkpoint_dir: Optional[str] = None,
    deterministic: bool = True,
    benchmark: bool = False,
    matmul_precision: Optional[str] = None,
    telemetry_every_n_batches: int = 20,
    validation_metrics_path: Optional[str] = None,
) -> Trainer:
    if matmul_precision is not None:
        if matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError(
                "matmul_precision must be one of: highest, high, medium"
            )
        torch.set_float32_matmul_precision(matmul_precision)
        print(f"[trainer] float32_matmul_precision={matmul_precision}", flush=True)

    logger = None
    if TensorBoardLogger is not None:
        try:
            logger = TensorBoardLogger(log_dir, name=task)
        except (ImportError, ModuleNotFoundError):
            # TensorBoard is optional for training and is not installed in the
            # minimal AI Lab image.
            logger = None
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback: Optional[ModelCheckpoint] = None

    if save_checkpoints != "none" and model_type in [
        "BART",
        "BartForConditionalGeneration",
        "CustomBartForConditionalGeneration",
        "T5ForConditionalGeneration",
        "CustomModel"
    ]:
        if save_checkpoints == "best_5":
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir, monitor=checkpoint_monitor, save_last=False, save_top_k=5, mode="max" if "loss" not in checkpoint_monitor else "min"
            )
        elif save_checkpoints == "every_5_epochs":
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir, monitor=checkpoint_monitor, save_last=False, save_top_k=-1, every_n_epochs=5, mode="max" if "loss" not in checkpoint_monitor else "min"
            )
        elif save_checkpoints == "all":
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir, monitor=checkpoint_monitor, save_last=False, save_top_k=-1, mode="max" if "loss" not in checkpoint_monitor else "min"
            )
        if checkpoint_callback is not None:
            checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "_"
    elif save_checkpoints != "none" and model_type == "encoder":
        if "weather" in task:
            mode = "min"
        else:
            mode = "max"
        print(mode)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir, monitor="val_f1_score", save_last=False, save_top_k=5, mode=mode
        )

    callbacks = [
        lr_monitor,
        TrainingTelemetryCallback(every_n_batches=telemetry_every_n_batches),
    ]
    if validation_metrics_path is None:
        validation_metrics_path = str(Path(log_dir) / task / "validation_metrics.jsonl")
    callbacks.append(ValidationMetricsCallback(validation_metrics_path))
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)

    if early_stopping_patience:
        callbacks.append(
            EarlyStopping(
                monitor=checkpoint_monitor,
                min_delta=early_stopping_delta,
                patience=early_stopping_patience,
                mode="max" if "loss" not in checkpoint_monitor else "min",
            )
        )

    if early_stopping_len_set_sel:
        callbacks.append(
            EarlyStopping(
                monitor="len_set_sel",
                min_delta=0.,
                patience=3,
                mode="max",
            )
        )

    strategy = "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"

    trainer = Trainer(
        devices = -1 if torch.cuda.is_available() else 1,
        logger = logger,
        max_epochs = epochs,
        accumulate_grad_batches = acc_batches,
        gradient_clip_val = clip_grad,
        limit_train_batches = limit_train_batches,
        limit_val_batches = limit_val_batches,
        callbacks = callbacks,
        precision=precision or ("16-mixed" if torch.cuda.is_available() else "32-true"),
        strategy = strategy,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        deterministic=deterministic,
        benchmark=benchmark,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1 if update_dataloaders else 0
    )
    return trainer

"""Validation metrics for multilabel auxiliary tasks."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelRecall,
    MultilabelSpecificity,
)


class MultilabelValidationMetrics(nn.Module):
    """Accumulate per-label metrics and expose their macro averages."""

    def __init__(self, num_labels: int, threshold: float = 0.5) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive.")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between zero and one.")

        self.num_labels = num_labels
        self.threshold = threshold
        self.average_precision = MultilabelAveragePrecision(
            num_labels=num_labels,
            average=None,
        )
        self.f1 = MultilabelF1Score(
            num_labels=num_labels,
            average=None,
            threshold=threshold,
        )
        self.recall = MultilabelRecall(
            num_labels=num_labels,
            average=None,
            threshold=threshold,
        )
        self.specificity = MultilabelSpecificity(
            num_labels=num_labels,
            average=None,
            threshold=threshold,
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state from raw logits and binary targets."""
        if logits.ndim != 2 or logits.shape[1] != self.num_labels:
            raise ValueError(
                f"Expected logits with shape [batch, {self.num_labels}], got {logits.shape}."
            )
        if targets.shape != logits.shape:
            raise ValueError(
                f"Targets must match logits shape {logits.shape}, got {targets.shape}."
            )

        probabilities = torch.sigmoid(logits.detach())
        binary_targets = targets.detach().to(dtype=torch.long)
        self.average_precision.update(probabilities, binary_targets)
        self.f1.update(probabilities, binary_targets)
        self.recall.update(probabilities, binary_targets)
        self.specificity.update(probabilities, binary_targets)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Return per-bit metrics and macro averages."""
        per_bit_auprc = self.average_precision.compute()
        per_bit_f1 = self.f1.compute()
        per_bit_balanced_accuracy = (self.recall.compute() + self.specificity.compute()) / 2.0
        return {
            "macro_auprc": torch.nanmean(per_bit_auprc),
            "macro_f1": torch.nanmean(per_bit_f1),
            "macro_balanced_accuracy": torch.nanmean(per_bit_balanced_accuracy),
            "per_bit_auprc": per_bit_auprc,
            "per_bit_f1": per_bit_f1,
            "per_bit_balanced_accuracy": per_bit_balanced_accuracy,
        }

    def reset(self) -> None:
        """Clear accumulated predictions and confusion counts."""
        self.average_precision.reset()
        self.f1.reset()
        self.recall.reset()
        self.specificity.reset()

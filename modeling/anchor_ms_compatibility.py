"""Formula+CNMR conditioned compatibility scoring for MS.

This module deliberately contains no generator parameters.  The molecular
generator is used as a frozen feature extractor; only these two towers are
trained in the ACCR-MS Stage 1 experiment.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class AnchorMSCompatibility(nn.Module):
    """Dual-tower scorer for an anchor representation and an MS representation."""

    def __init__(
        self,
        anchor_dim: int = 1024,
        ms_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.anchor_tower = nn.Sequential(
            nn.Linear(anchor_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.ms_tower = nn.Sequential(
            nn.Linear(ms_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_anchor(self, anchor: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.anchor_tower(anchor), dim=-1)

    def encode_ms(self, ms: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.ms_tower(ms), dim=-1)

    def score_embeddings(
        self, anchor: torch.Tensor, ms: torch.Tensor
    ) -> torch.Tensor:
        """Return pairwise scores with shape ``[anchor_batch, ms_batch]``."""
        return self.encode_anchor(anchor) @ self.encode_ms(ms).transpose(0, 1) / self.temperature

    def score_pairs(
        self, anchor: torch.Tensor, ms: torch.Tensor
    ) -> torch.Tensor:
        """Return aligned pair scores with shape ``[batch]``."""
        return (self.encode_anchor(anchor) * self.encode_ms(ms)).sum(dim=-1) / self.temperature

    def forward(self, anchor: torch.Tensor, ms: torch.Tensor) -> torch.Tensor:
        return self.score_embeddings(anchor, ms)

    @staticmethod
    def info_nce_loss(scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError("InfoNCE scores must be a square [batch, batch] matrix")
        labels = torch.arange(scores.shape[0], device=scores.device)
        return F.cross_entropy(scores, labels)

    @staticmethod
    def margin_ranking_loss(
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float = 0.2,
    ) -> torch.Tensor:
        if positive.shape != negative.shape:
            raise ValueError("positive and negative scores must have the same shape")
        return F.relu(float(margin) - positive + negative).mean()


def make_hard_negative_indices(
    formulas: list[str],
    *,
    seed: int = 3247,
) -> torch.Tensor:
    """Choose same-formula negatives where possible, deterministically.

    The returned index is never the positive index when a batch has at least
    two rows.  Cross-batch hard-negative mining is intentionally left to the
    cache trainer, so the rule remains auditable and reproducible.
    """
    generator = torch.Generator().manual_seed(seed)
    groups: dict[str, list[int]] = {}
    for index, formula in enumerate(formulas):
        groups.setdefault(str(formula), []).append(index)
    negative: list[int] = []
    for index, formula in enumerate(formulas):
        candidates = [value for value in groups[str(formula)] if value != index]
        if not candidates:
            candidates = [value for value in range(len(formulas)) if value != index]
        if not candidates:
            candidates = [index]
        choice = torch.randint(len(candidates), (1,), generator=generator).item()
        negative.append(candidates[choice])
    return torch.tensor(negative, dtype=torch.long)


def score_summary(scores: torch.Tensor) -> Tuple[float, float, float]:
    values = scores.detach().float().cpu()
    return float(values.mean()), float(values.std()), float(values.max())

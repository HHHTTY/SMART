"""Small, explicit PCGrad utilities for modality-level TTT.

The functions in this module operate on an already selected parameter list.  They
never inspect or mutate model parameters, which makes the trainable scope easy to
audit in the Conflict-Aware Modality TTT runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


Gradient = torch.Tensor | None


@dataclass(frozen=True)
class GradientStats:
    cosine: float | None
    full_norm: float
    auxiliary_norm: float
    projected_norm: float
    projection_ratio: float
    conflict: bool
    shared_elements: int

    def as_dict(self) -> dict[str, float | bool | int | None]:
        return {
            "cosine": self.cosine,
            "full_norm": self.full_norm,
            "auxiliary_norm": self.auxiliary_norm,
            "projected_norm": self.projected_norm,
            "projection_ratio": self.projection_ratio,
            "conflict": self.conflict,
            "shared_elements": self.shared_elements,
        }


def _finite(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)


def _flat_pair(
    full: Sequence[Gradient], auxiliary: Sequence[Gradient]
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(full) != len(auxiliary):
        raise ValueError("gradient sequences must have equal length")
    full_parts: list[torch.Tensor] = []
    auxiliary_parts: list[torch.Tensor] = []
    for first, second in zip(full, auxiliary):
        if first is None or second is None:
            continue
        if first.shape != second.shape:
            raise ValueError("gradient shapes do not match")
        full_parts.append(_finite(first).reshape(-1))
        auxiliary_parts.append(_finite(second).reshape(-1))
    if not full_parts:
        device = next(
            (value.device for value in (*full, *auxiliary) if value is not None),
            torch.device("cpu"),
        )
        empty = torch.empty(0, dtype=torch.float32, device=device)
        return empty, empty
    return torch.cat(full_parts), torch.cat(auxiliary_parts)


def cosine_and_projection(
    full: Sequence[Gradient],
    auxiliary: Sequence[Gradient],
    *,
    conflict_threshold: float = 0.0,
    eps: float = 1e-12,
) -> tuple[list[Gradient], GradientStats]:
    """Return an auxiliary gradient, projected only for a negative cosine.

    The projection is applied to the complete parameter vector, then split back
    into the original parameter shapes.  Missing gradients stay missing.  All
    norm/cosine decisions happen in FP32 and non-finite values are treated as
    zero; this prevents a single bad bf16 element from silently poisoning an
    update.
    """
    full_flat, auxiliary_flat = _flat_pair(full, auxiliary)
    shared = int(full_flat.numel())
    if shared == 0:
        empty_stats = GradientStats(None, 0.0, 0.0, 0.0, 0.0, False, 0)
        return list(auxiliary), empty_stats
    full_norm = float(torch.linalg.vector_norm(full_flat).item())
    auxiliary_norm = float(torch.linalg.vector_norm(auxiliary_flat).item())
    denominator = max(full_norm * auxiliary_norm, eps)
    cosine_value = float(torch.dot(full_flat, auxiliary_flat).item() / denominator)
    should_project = cosine_value < -abs(float(conflict_threshold))
    projected_flat = auxiliary_flat
    if should_project and full_norm > eps:
        coefficient = torch.dot(auxiliary_flat, full_flat) / (
            torch.dot(full_flat, full_flat) + eps
        )
        projected_flat = auxiliary_flat - coefficient * full_flat
    projected_norm = float(torch.linalg.vector_norm(projected_flat).item())
    projection_ratio = projected_norm / max(auxiliary_norm, eps)

    result: list[Gradient] = []
    offset = 0
    for first, second in zip(full, auxiliary):
        if second is None:
            result.append(None)
            continue
        count = second.numel() if first is not None else 0
        if first is None or count == 0:
            result.append(_finite(second).to(dtype=second.dtype).reshape_as(second))
            continue
        part = projected_flat[offset : offset + count].reshape_as(second)
        result.append(part.to(dtype=second.dtype))
        offset += count
    return result, GradientStats(
        cosine=cosine_value,
        full_norm=full_norm,
        auxiliary_norm=auxiliary_norm,
        projected_norm=projected_norm,
        projection_ratio=projection_ratio,
        conflict=should_project,
        shared_elements=shared,
    )


def combine_gradients(
    parameters: Sequence[torch.nn.Parameter],
    full: Sequence[Gradient],
    auxiliaries: Sequence[tuple[float, Sequence[Gradient]]],
) -> tuple[torch.Tensor, int]:
    """Install ``full + sum(weight * auxiliary)`` and return norm/elements."""
    if len(parameters) != len(full):
        raise ValueError("parameter and full-gradient sequences have unequal length")
    for _weight, gradients in auxiliaries:
        if len(gradients) != len(parameters):
            raise ValueError("parameter and auxiliary-gradient sequences differ")
    squared = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    elements = 0
    for index, parameter in enumerate(parameters):
        value = None if full[index] is None else _finite(full[index])
        for weight, gradients in auxiliaries:
            current = gradients[index]
            if current is not None:
                addition = _finite(current).mul(float(weight))
                value = addition if value is None else value + addition
        if value is None:
            parameter.grad = None
            continue
        value = value.to(dtype=parameter.dtype).reshape_as(parameter)
        parameter.grad = value
        squared = squared + torch.sum(value.float() * value.float(), dtype=torch.float64)
        elements += value.numel()
    return squared.sqrt().to(dtype=torch.float32), elements


"""Deterministic sample-level PMGFA selection for CASP.

The selector in this module deliberately does not depend on a learned router,
router normalization, target labels, or target-batch statistics.  It consumes
the sample-invariant feature tensor produced by ``build_sample_invariant_router_features``
and selects one available spectroscopy modality.  Formula is handled by the
caller and is never part of the returned singleton action.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import torch

from .modality_subset_router import ROUTED_MODALITIES, RouterFeatureBatch


DIRECT_SELECTOR_NAMES = (
    "A0_pmgfa_argmin",
    "A1_sample_source_distance_argmin",
    "A2_standardized_mean_argmin",
    "A3_pmgfa_formula_tiebreak",
    "A4_fixed_formula_cnmr",
    "A5_random_singleton",
)


@dataclass(frozen=True)
class DirectSelection:
    """Selection output for a batch of rows."""

    selector: str
    selected: torch.Tensor
    scores: torch.Tensor
    pmgfa: torch.Tensor
    sample_source_distance: torch.Tensor
    formula_compatibility: torch.Tensor
    margin: torch.Tensor

    def validate(self) -> None:
        if self.selected.ndim != 1:
            raise ValueError("selected actions must have shape [batch]")
        batch = self.selected.shape[0]
        for name, value in (
            ("scores", self.scores),
            ("pmgfa", self.pmgfa),
            ("sample_source_distance", self.sample_source_distance),
            ("formula_compatibility", self.formula_compatibility),
        ):
            if value.shape != (batch, len(ROUTED_MODALITIES)):
                raise ValueError(f"{name} must have shape [batch, 4]")
        if self.margin.shape != (batch,):
            raise ValueError("margin must have shape [batch]")
        if not torch.isfinite(self.selected.float()).all():
            raise ValueError("selected actions contain non-finite values")


def _finite_mask(features: RouterFeatureBatch) -> torch.Tensor:
    available = features.availability.bool()
    if available.ndim != 2 or available.shape[1] != len(ROUTED_MODALITIES):
        raise ValueError("availability must have shape [batch, 4]")
    if not available.any(dim=1).all():
        raise ValueError("every sample needs at least one available spectrum")
    return available


def _masked_values(values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    if values.shape != available.shape:
        raise ValueError("feature and availability shapes differ")
    return values.float().masked_fill(~available, float("inf"))


def _row_standardize(values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """Standardize one metric within each row, without using batch moments."""
    valid = available.float()
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values.masked_fill(~available, 0.0) * valid).sum(dim=1, keepdim=True) / count
    centered = values - mean
    variance = (centered.masked_fill(~available, 0.0).square() * valid).sum(
        dim=1, keepdim=True
    ) / count
    standardized = centered / variance.sqrt().clamp_min(1e-6)
    return standardized.masked_fill(~available, float("inf"))


def _margin(scores: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    ordered = scores.masked_fill(~available, float("inf")).sort(dim=1).values
    finite = torch.isfinite(ordered)
    first = ordered[:, 0]
    second = torch.where(finite[:, 1], ordered[:, 1], first)
    return (second - first).nan_to_num(0.0, posinf=0.0, neginf=0.0)


def _argmin_with_formula_tie(
    scores: torch.Tensor,
    compatibility: torch.Tensor,
    available: torch.Tensor,
    *,
    tolerance: float,
) -> torch.Tensor:
    minimum = scores.masked_fill(~available, float("inf")).min(dim=1, keepdim=True).values
    ties = available & (scores - minimum).abs().le(float(tolerance))
    compatible = compatibility.masked_fill(~ties, float("-inf"))
    chosen = compatible.argmax(dim=1)
    no_tie = ~ties.any(dim=1)
    if no_tie.any():
        chosen[no_tie] = scores[no_tie].masked_fill(~available[no_tie], float("inf")).argmin(dim=1)
    return chosen


def select_singleton(
    features: RouterFeatureBatch,
    *,
    selector: str,
    generator: torch.Generator | None = None,
    tie_tolerance: float = 1e-8,
) -> DirectSelection:
    """Select one spectrum per row using A0-A5.

    A2 uses row-wise z-scores of the two fixed observable distances.  This is
    intentionally sample-local: it cannot change when neighbouring rows are
    regrouped into a different batch.  A3 only consults Formula compatibility
    when the PMGFA minima are tied within ``tie_tolerance``.
    """
    if selector not in DIRECT_SELECTOR_NAMES:
        raise ValueError(f"unknown direct selector: {selector}")
    features.validate()
    available = _finite_mask(features)
    pmgfa = features.modality[:, :, 0].float()
    distance = features.modality[:, :, 1].float()
    compatibility = features.modality[:, :, 2].float()
    if not torch.isfinite(pmgfa[available]).all() or not torch.isfinite(distance[available]).all():
        raise ValueError("PMGFA and source-distance features must be finite when available")

    if selector == "A0_pmgfa_argmin":
        scores = _masked_values(pmgfa, available)
        selected = scores.argmin(dim=1)
    elif selector == "A1_sample_source_distance_argmin":
        scores = _masked_values(distance, available)
        selected = scores.argmin(dim=1)
    elif selector == "A2_standardized_mean_argmin":
        pmgfa_z = _row_standardize(pmgfa, available)
        distance_z = _row_standardize(distance, available)
        scores = ((pmgfa_z + distance_z) * 0.5).masked_fill(~available, float("inf"))
        selected = scores.argmin(dim=1)
    elif selector == "A3_pmgfa_formula_tiebreak":
        scores = _masked_values(pmgfa, available)
        selected = _argmin_with_formula_tie(
            pmgfa, compatibility, available, tolerance=tie_tolerance
        )
    elif selector == "A4_fixed_formula_cnmr":
        cnmr = ROUTED_MODALITIES.index("CNMR")
        if not available[:, cnmr].all():
            raise ValueError("A4 requires CNMR to be available for every row")
        scores = _masked_values(pmgfa, available)
        selected = torch.full_like(scores[:, 0], cnmr, dtype=torch.long)
    else:
        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(0)
        selected_rows = []
        for row in available.detach().cpu():
            choices = row.nonzero(as_tuple=False).flatten()
            index = int(torch.randint(len(choices), (), generator=generator).item())
            selected_rows.append(int(choices[index]))
        selected = torch.tensor(selected_rows, dtype=torch.long, device=available.device)
        scores = _masked_values(pmgfa, available)

    selected = selected.to(dtype=torch.long)
    margin_scores = scores if selector != "A4_fixed_formula_cnmr" else _masked_values(pmgfa, available)
    result = DirectSelection(
        selector=selector,
        selected=selected,
        scores=scores,
        pmgfa=pmgfa,
        sample_source_distance=distance,
        formula_compatibility=compatibility,
        margin=_margin(margin_scores, available),
    )
    result.validate()
    return result


def singleton_name(index: int) -> str:
    if not 0 <= int(index) < len(ROUTED_MODALITIES):
        raise ValueError("singleton index out of range")
    return ROUTED_MODALITIES[int(index)]


def singleton_subset(index: int) -> tuple[str, ...]:
    return (singleton_name(index),)


def route_entropy(selected: torch.Tensor, modality_count: int = 4) -> float:
    counts = torch.bincount(selected.detach().cpu(), minlength=modality_count).float()
    probabilities = counts / counts.sum().clamp_min(1.0)
    return float(-(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum())


def source_statistics_hash(source: Any) -> str:
    """Hash source statistics by values, independent of pickle ordering."""
    if hasattr(source, "sample_count") and hasattr(source, "means") and hasattr(source, "stds"):
        payload: dict[str, Any] = {"sample_count": int(source.sample_count), "modalities": {}}
        for name in sorted(source.means):
            mean = source.means[name].detach().cpu().contiguous()
            std = source.stds[name].detach().cpu().contiguous()
            payload["modalities"][name] = {
                "mean_dtype": str(mean.dtype),
                "mean_shape": list(mean.shape),
                "mean": mean.numpy().tobytes().hex(),
                "std_dtype": str(std.dtype),
                "std_shape": list(std.shape),
                "std": std.numpy().tobytes().hex(),
            }
    elif isinstance(source, Mapping):
        payload = source
    else:
        raise TypeError("source must be SourceShiftStatistics or a mapping")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ComplementDropoutViews:
    full: tuple[str, ...]
    single_dropout: tuple[str, ...] | None
    double_dropout: tuple[str, ...] | None
    dropped_single: tuple[str, ...]
    dropped_double: tuple[str, ...]


class ComplementDropoutScheduler:
    """Deterministic block-balanced one/two-modality complement dropout."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._random = random.Random(self.seed)
        self._consumed = 0

    @property
    def consumed(self) -> int:
        return self._consumed

    def next(self, complement: Sequence[str]) -> ComplementDropoutViews:
        values = tuple(str(name) for name in complement)
        unknown = set(values).difference(ROUTED_MODALITIES)
        if len(set(values)) != len(values) or unknown:
            raise ValueError("complement contains duplicate or unknown modalities")
        shuffled = list(values)
        self._random.shuffle(shuffled)
        single_drop = tuple(shuffled[:1]) if len(shuffled) >= 2 else ()
        double_drop = tuple(shuffled[:2]) if len(shuffled) >= 3 else ()
        single_keep = tuple(name for name in values if name not in single_drop) or None
        double_keep = tuple(name for name in values if name not in double_drop) or None
        output = ComplementDropoutViews(
            full=values,
            single_dropout=single_keep,
            double_dropout=double_keep,
            dropped_single=single_drop,
            dropped_double=double_drop,
        )
        self._consumed += 1
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "consumed": self._consumed,
            "random_state": self._random.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("dropout scheduler seed differs from saved state")
        self._consumed = int(state["consumed"])
        self._random.setstate(state["random_state"])


"""Metrics for batch-context versus sample-context router parity audits."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import torch

from .modality_subset_router import ACTION_SUBSETS


def _flat(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().reshape(value.shape[0], -1)


def feature_delta_statistics(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    if left.shape != right.shape:
        raise ValueError(f"feature shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}")
    delta = (_flat(left) - _flat(right)).abs()
    return {
        "shape": list(left.shape),
        "mean_abs_difference": delta.mean(dim=0).tolist(),
        "std_abs_difference": delta.std(dim=0, unbiased=False).tolist(),
        "max_abs_difference": delta.max(dim=0).values.tolist(),
        "row_mean_abs_difference": delta.mean(dim=1).tolist(),
    }


def _rowwise_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = _flat(left)
    right = _flat(right)
    return torch.nn.functional.cosine_similarity(left, right, dim=1, eps=1e-12)


def _rowwise_kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float().clamp_min(1e-12)
    right = right.float().clamp_min(1e-12)
    return (left * (left.log() - right.log())).sum(dim=1)


def _top_margin(probabilities: torch.Tensor) -> torch.Tensor:
    top = probabilities.topk(2, dim=1).values
    return top[:, 0] - top[:, 1]


def route_frequency(actions: torch.Tensor) -> dict[str, int]:
    return dict(
        Counter(
            "+".join(ACTION_SUBSETS[int(index)])
            for index in actions.detach().cpu().tolist()
            if int(index) >= 0
        )
    )


def parity_summary(
    batch_features: Mapping[str, torch.Tensor],
    sample_features: Mapping[str, torch.Tensor],
    batch_logits: torch.Tensor,
    sample_logits: torch.Tensor,
    batch_probabilities: torch.Tensor,
    sample_probabilities: torch.Tensor,
    batch_actions: torch.Tensor,
    sample_actions: torch.Tensor,
) -> dict[str, object]:
    if batch_actions.shape != sample_actions.shape:
        raise ValueError("batch and sample action arrays must have equal length")
    feature_stats = {
        name: feature_delta_statistics(batch_features[name], sample_features[name])
        for name in ("modality", "pair", "global_features", "availability")
    }
    agreement = batch_actions.eq(sample_actions)
    logits_cosine = _rowwise_cosine(batch_logits, sample_logits)
    probability_kl = _rowwise_kl(batch_probabilities, sample_probabilities)
    batch_margin = _top_margin(batch_probabilities)
    sample_margin = _top_margin(sample_probabilities)
    return {
        "rows": int(batch_actions.numel()),
        "feature_delta": feature_stats,
        "action_agreement": float(agreement.float().mean()),
        "action_agreement_count": int(agreement.sum()),
        "logit_cosine_mean": float(logits_cosine.mean()),
        "logit_cosine_std": float(logits_cosine.std(unbiased=False)),
        "probability_kl_batch_to_sample_mean": float(probability_kl.mean()),
        "probability_kl_batch_to_sample_p95": float(torch.quantile(probability_kl, 0.95)),
        "top1_margin_batch_mean": float(batch_margin.mean()),
        "top1_margin_sample_mean": float(sample_margin.mean()),
        "top1_margin_difference_mean": float((batch_margin - sample_margin).mean()),
        "top1_margin_difference_abs_mean": float((batch_margin - sample_margin).abs().mean()),
        "batch_route_frequency": route_frequency(batch_actions),
        "sample_route_frequency": route_frequency(sample_actions),
    }


def permutation_sensitivity(
    reference_actions: torch.Tensor,
    permuted_actions: torch.Tensor,
) -> dict[str, object]:
    if reference_actions.shape != permuted_actions.shape:
        raise ValueError("permutation action arrays must have equal length")
    changed = reference_actions.ne(permuted_actions)
    return {
        "rows": int(changed.numel()),
        "changed_count": int(changed.sum()),
        "changed_fraction": float(changed.float().mean()),
        "reference_route_frequency": route_frequency(reference_actions),
        "permuted_route_frequency": route_frequency(permuted_actions),
    }


def alignment_rows(
    source_row_index: list[int],
    batch_actions: torch.Tensor,
    sample_actions: torch.Tensor,
    batch_probabilities: torch.Tensor,
    sample_probabilities: torch.Tensor,
    batch_logits: torch.Tensor,
    sample_logits: torch.Tensor,
) -> list[dict[str, object]]:
    if len(source_row_index) != len(batch_actions):
        raise ValueError("source row indices and route arrays have different lengths")
    batch_margin = _top_margin(batch_probabilities)
    sample_margin = _top_margin(sample_probabilities)
    rows = []
    for row, source_index in enumerate(source_row_index):
        rows.append(
            {
                "source_row_index": int(source_index),
                "batch_action": int(batch_actions[row]),
                "sample_action": int(sample_actions[row]),
                "action_agreement": bool(batch_actions[row] == sample_actions[row]),
                "batch_probability": float(batch_probabilities[row, batch_actions[row]]),
                "sample_probability": float(sample_probabilities[row, sample_actions[row]]),
                "batch_margin": float(batch_margin[row]),
                "sample_margin": float(sample_margin[row]),
                "logit_cosine": float(_rowwise_cosine(batch_logits[row:row + 1], sample_logits[row:row + 1])[0]),
                "probability_kl_batch_to_sample": float(_rowwise_kl(batch_probabilities[row:row + 1], sample_probabilities[row:row + 1])[0]),
            }
        )
    return rows


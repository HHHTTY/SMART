"""Source statistics and observable shift features for modality routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .modality_subset_router import (
    PAIR_INDICES,
    ROUTED_MODALITIES,
    ROUTER_FEATURE_SCHEMA_SAMPLE_INVARIANT,
    ROUTER_FEATURE_SCHEMA_SOURCE_REFERENCE_FREE,
    ROUTER_FEATURE_SCHEMA_NO_BATCH_PMGFA,
    NO_BATCH_PMGFA_GLOBAL_FEATURE_NAMES,
    NO_BATCH_PMGFA_MODALITY_FEATURE_NAMES,
    SAMPLE_INVARIANT_GLOBAL_FEATURE_NAMES,
    SAMPLE_INVARIANT_MODALITY_FEATURE_NAMES,
    RouterFeatureBatch,
)


def masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("hidden/mask must have shapes [batch, tokens, dim]/[batch, tokens]")
    weights = mask.to(hidden.dtype)
    return (hidden * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)


@dataclass(frozen=True)
class SourceShiftStatistics:
    sample_count: int
    means: Mapping[str, torch.Tensor]
    stds: Mapping[str, torch.Tensor]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "modalities": {
                name: {"mean": self.means[name].cpu(), "std": self.stds[name].cpu()}
                for name in self.means
            },
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceShiftStatistics":
        modalities = payload["modalities"]
        return cls(
            sample_count=int(payload["sample_count"]),
            means={str(name): value["mean"].float() for name, value in modalities.items()},
            stds={str(name): value["std"].float() for name, value in modalities.items()},
        )


def fit_source_statistics(
    batches: Iterable[Mapping[str, Sequence[torch.Tensor]]],
) -> SourceShiftStatistics:
    rows: dict[str, list[list[torch.Tensor]]] = {}
    sample_count = 0
    for batch in batches:
        current_size = None
        for name, layers in batch.items():
            detached = [value.detach().float().cpu() for value in layers]
            if not detached or any(value.ndim != 2 for value in detached):
                raise ValueError("each modality must contain [batch, dim] layer features")
            current_size = detached[0].shape[0] if current_size is None else current_size
            if any(value.shape[0] != current_size for value in detached):
                raise ValueError("inconsistent batch size in source feature batch")
            rows.setdefault(str(name), []).append(detached)
        sample_count += int(current_size or 0)
    if sample_count < 1 or not rows:
        raise ValueError("source statistics require at least one feature batch")
    means: dict[str, torch.Tensor] = {}
    stds: dict[str, torch.Tensor] = {}
    for name, modality_batches in rows.items():
        layer_count = len(modality_batches[0])
        if any(len(value) != layer_count for value in modality_batches):
            raise ValueError("source layer counts are inconsistent")
        layer_rows = [
            torch.cat([value[layer] for value in modality_batches], dim=0)
            for layer in range(layer_count)
        ]
        means[name] = torch.stack([value.mean(dim=0) for value in layer_rows])
        stds[name] = torch.stack(
            [value.std(dim=0, unbiased=value.shape[0] > 1).clamp_min(1e-6) for value in layer_rows]
        )
    return SourceShiftStatistics(sample_count, means, stds)


def extract_layerwise_modality_features(
    model: Any,
    batch: Mapping[str, Any],
    *,
    modalities: Sequence[str] = ("Formula", *ROUTED_MODALITIES),
) -> tuple[dict[str, list[torch.Tensor]], dict[str, torch.Tensor]]:
    """Run each modality independently through the frozen shared encoder.

    Separating segments avoids treating already-fused features as an
    independent modality quality score.  This function reads encoder inputs
    and masks only; decoder targets are neither needed nor inspected.
    """
    input_ids, attention_mask, embeddings = model._prepare_generation_inputs(dict(batch))
    encoder = model.hf_model.model.encoder if hasattr(model.hf_model, "model") else model.hf_model.encoder
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise ValueError("PMGFA extraction requires an encoder with .layers")
    requested = set(modalities)
    features: dict[str, list[torch.Tensor]] = {}
    masks: dict[str, torch.Tensor] = {}
    offset = 0
    for name, value in input_ids.items():
        length = int(model._input_sequence_length(value))
        mask = attention_mask[:, offset : offset + length].bool()
        hidden = embeddings[:, offset : offset + length]
        offset += length
        if name not in requested:
            continue
        modality_layers = []
        for layer in layers:
            hidden = layer(hidden, src_key_padding_mask=~mask)
            modality_layers.append(masked_pool(hidden, mask))
        features[str(name)] = modality_layers
        masks[str(name)] = mask
    return features, masks


def pmgfa_discrepancy(
    features: Mapping[str, Sequence[torch.Tensor]],
    source: SourceShiftStatistics,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    values: dict[str, torch.Tensor] = {}
    for name, layers in features.items():
        if name not in source.means:
            continue
        stacked = torch.stack([value.float() for value in layers], dim=1)
        target_mean = stacked.mean(dim=0)
        target_std = stacked.std(dim=0, unbiased=stacked.shape[0] > 1)
        source_mean = source.means[name].to(stacked.device)
        source_std = source.stds[name].to(stacked.device)
        if target_mean.shape != source_mean.shape:
            raise ValueError(f"source statistics shape mismatch for {name}")
        scale = math.sqrt(target_mean.shape[-1])
        values[name] = (
            torch.linalg.vector_norm(target_mean - source_mean, dim=-1)
            + torch.linalg.vector_norm(target_std - source_std, dim=-1)
        ).mean() / scale
    if not values:
        raise ValueError("no target modality matches source PMGFA statistics")
    return torch.stack(tuple(values.values())).mean(), values


def sample_source_distance(
    layers: Sequence[torch.Tensor],
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
) -> torch.Tensor:
    stacked = torch.stack([value.float() for value in layers], dim=1)
    mean = source_mean.to(stacked.device)
    std = source_std.to(stacked.device).clamp_min(1e-4)
    if stacked.shape[1:] != mean.shape:
        raise ValueError("sample/source feature shapes do not match")
    standardized = (stacked - mean.unsqueeze(0)) / std.unsqueeze(0)
    return standardized.square().mean(dim=(1, 2)).sqrt()


def build_router_features(
    features: Mapping[str, Sequence[torch.Tensor]],
    token_masks: Mapping[str, torch.Tensor],
    source: SourceShiftStatistics,
) -> tuple[RouterFeatureBatch, dict[str, torch.Tensor]]:
    """Build observable shift, anchor-compatibility and exposure features.

    The per-modality columns are PMGFA, sample-to-source distance, cosine
    compatibility with Formula, and log token count. Domain-adversarial
    features are intentionally excluded from this router.
    """
    if "Formula" not in features:
        raise ValueError("Formula features are required as the structural anchor")
    batch_size = features["Formula"][0].shape[0]
    anchor = F.normalize(features["Formula"][-1].float(), dim=-1)
    modality_rows = []
    pooled: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, torch.Tensor] = {}
    availability_rows = []
    counts = []
    for name in ROUTED_MODALITIES:
        if name not in features or name not in token_masks or name not in source.means:
            modality_rows.append(anchor.new_zeros((batch_size, 4)))
            availability_rows.append(torch.zeros(batch_size, dtype=torch.bool, device=anchor.device))
            counts.append(anchor.new_zeros(batch_size))
            pooled[name] = anchor.new_zeros(anchor.shape)
            diagnostics[f"batch_pmgfa/{name}"] = anchor.new_tensor(float("nan"))
            diagnostics[f"sample_distance/{name}"] = anchor.new_zeros(batch_size)
            continue
        layers = features[name]
        current = F.normalize(layers[-1].float(), dim=-1)
        pooled[name] = current
        distance = sample_source_distance(layers, source.means[name], source.stds[name])
        stacked = torch.stack([value.float() for value in layers], dim=1)
        target_mean = stacked.mean(dim=0)
        target_std = stacked.std(dim=0, unbiased=batch_size > 1)
        source_mean = source.means[name].to(anchor.device)
        source_std = source.stds[name].to(anchor.device)
        pmgfa = (
            torch.linalg.vector_norm(target_mean - source_mean, dim=-1)
            + torch.linalg.vector_norm(target_std - source_std, dim=-1)
        ).mean() / math.sqrt(target_mean.shape[-1])
        mask = token_masks[name].bool()
        available = mask.any(dim=1)
        count = mask.sum(dim=1).float()
        compatibility = (current * anchor).sum(dim=-1)
        modality_rows.append(
            torch.stack(
                (
                    pmgfa.expand(batch_size),
                    distance,
                    compatibility,
                    torch.log1p(count),
                ),
                dim=-1,
            )
        )
        availability_rows.append(available)
        counts.append(count)
        diagnostics[f"batch_pmgfa/{name}"] = pmgfa.detach()
        diagnostics[f"sample_distance/{name}"] = distance.detach()
    modality_tensor = torch.stack(modality_rows, dim=1)
    availability = torch.stack(availability_rows, dim=1)
    count_tensor = torch.stack(counts, dim=1)
    pair_rows = []
    for left, right in PAIR_INDICES:
        left_name = ROUTED_MODALITIES[left]
        right_name = ROUTED_MODALITIES[right]
        final_cosine = (pooled[left_name] * pooled[right_name]).sum(dim=-1)
        left_layers = features.get(left_name)
        right_layers = features.get(right_name)
        if left_layers is None or right_layers is None:
            mean_cosine = torch.zeros_like(final_cosine)
        else:
            layer_cosines = [
                (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(dim=-1)
                for a, b in zip(left_layers, right_layers)
            ]
            mean_cosine = torch.stack(layer_cosines, dim=-1).mean(dim=-1)
        pair_rows.append(torch.stack((final_cosine, mean_cosine), dim=-1))
    pair_tensor = torch.stack(pair_rows, dim=1)
    total_count = count_tensor.sum(dim=1).clamp_min(1.0)
    global_features = torch.stack(
        (
            availability.float().sum(dim=1) / 4.0,
            torch.log1p(total_count),
            modality_tensor[:, :, 0].nan_to_num().mean(dim=1),
            modality_tensor[:, :, 1].mean(dim=1),
        ),
        dim=-1,
    )
    result = RouterFeatureBatch(modality_tensor, pair_tensor, global_features, availability)
    result.validate()
    return result, diagnostics


def build_source_reference_free_router_features(
    features: Mapping[str, Sequence[torch.Tensor]],
    token_masks: Mapping[str, torch.Tensor],
) -> tuple[RouterFeatureBatch, dict[str, torch.Tensor]]:
    """Build router features without PMGFA or fitted source statistics."""
    if "Formula" not in features:
        raise ValueError("Formula features are required as the structural anchor")
    batch_size = features["Formula"][0].shape[0]
    anchor = F.normalize(features["Formula"][-1].float(), dim=-1)
    modality_rows = []
    pooled: dict[str, torch.Tensor] = {}
    availability_rows = []
    counts = []
    diagnostics: dict[str, torch.Tensor] = {}
    for name in ROUTED_MODALITIES:
        if name not in features or name not in token_masks:
            modality_rows.append(anchor.new_zeros((batch_size, 2)))
            availability_rows.append(
                torch.zeros(batch_size, dtype=torch.bool, device=anchor.device)
            )
            counts.append(anchor.new_zeros(batch_size))
            pooled[name] = anchor.new_zeros(anchor.shape)
            continue
        layers = features[name]
        current = F.normalize(layers[-1].float(), dim=-1)
        pooled[name] = current
        mask = token_masks[name].bool()
        available = mask.any(dim=1)
        count = mask.sum(dim=1).float()
        compatibility = (current * anchor).sum(dim=-1)
        modality_rows.append(
            torch.stack((compatibility, torch.log1p(count)), dim=-1)
        )
        availability_rows.append(available)
        counts.append(count)
        diagnostics[f"formula_compatibility/{name}"] = compatibility.detach()

    modality_tensor = torch.stack(modality_rows, dim=1)
    availability = torch.stack(availability_rows, dim=1)
    count_tensor = torch.stack(counts, dim=1)
    pair_rows = []
    for left, right in PAIR_INDICES:
        left_name = ROUTED_MODALITIES[left]
        right_name = ROUTED_MODALITIES[right]
        final_cosine = (pooled[left_name] * pooled[right_name]).sum(dim=-1)
        left_layers = features.get(left_name)
        right_layers = features.get(right_name)
        if left_layers is None or right_layers is None:
            mean_cosine = torch.zeros_like(final_cosine)
        else:
            layer_cosines = [
                (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(dim=-1)
                for a, b in zip(left_layers, right_layers)
            ]
            mean_cosine = torch.stack(layer_cosines, dim=-1).mean(dim=-1)
        pair_rows.append(torch.stack((final_cosine, mean_cosine), dim=-1))
    pair_tensor = torch.stack(pair_rows, dim=1)
    global_features = torch.stack(
        (
            availability.float().sum(dim=1) / 4.0,
            torch.log1p(count_tensor.sum(dim=1).clamp_min(1.0)),
        ),
        dim=-1,
    )
    result = RouterFeatureBatch(
        modality_tensor,
        pair_tensor,
        global_features,
        availability,
        feature_schema=ROUTER_FEATURE_SCHEMA_SOURCE_REFERENCE_FREE,
    )
    result.validate()
    return result, diagnostics


def build_no_batch_pmgfa_router_features(
    features: Mapping[str, Sequence[torch.Tensor]],
    token_masks: Mapping[str, torch.Tensor],
    source: SourceShiftStatistics,
) -> tuple[RouterFeatureBatch, dict[str, torch.Tensor]]:
    """Build the original router inputs with only batch PMGFA removed."""
    if "Formula" not in features:
        raise ValueError("Formula features are required as the structural anchor")
    batch_size = features["Formula"][0].shape[0]
    anchor = F.normalize(features["Formula"][-1].float(), dim=-1)
    modality_rows = []
    pooled: dict[str, torch.Tensor] = {}
    availability_rows = []
    counts = []
    distances = []
    diagnostics: dict[str, torch.Tensor] = {}
    for name in ROUTED_MODALITIES:
        if name not in features or name not in token_masks or name not in source.means:
            modality_rows.append(anchor.new_zeros((batch_size, 3)))
            availability_rows.append(
                torch.zeros(batch_size, dtype=torch.bool, device=anchor.device)
            )
            counts.append(anchor.new_zeros(batch_size))
            distances.append(anchor.new_zeros(batch_size))
            pooled[name] = anchor.new_zeros(anchor.shape)
            continue
        layers = features[name]
        current = F.normalize(layers[-1].float(), dim=-1)
        pooled[name] = current
        distance = sample_source_distance(layers, source.means[name], source.stds[name])
        mask = token_masks[name].bool()
        available = mask.any(dim=1)
        count = mask.sum(dim=1).float()
        compatibility = (current * anchor).sum(dim=-1)
        modality_rows.append(
            torch.stack((distance, compatibility, torch.log1p(count)), dim=-1)
        )
        availability_rows.append(available)
        counts.append(count)
        distances.append(distance)
        diagnostics[f"sample_distance/{name}"] = distance.detach()

    modality_tensor = torch.stack(modality_rows, dim=1)
    availability = torch.stack(availability_rows, dim=1)
    count_tensor = torch.stack(counts, dim=1)
    distance_tensor = torch.stack(distances, dim=1)
    pair_rows = []
    for left, right in PAIR_INDICES:
        left_name = ROUTED_MODALITIES[left]
        right_name = ROUTED_MODALITIES[right]
        final_cosine = (pooled[left_name] * pooled[right_name]).sum(dim=-1)
        left_layers = features.get(left_name)
        right_layers = features.get(right_name)
        if left_layers is None or right_layers is None:
            mean_cosine = torch.zeros_like(final_cosine)
        else:
            layer_cosines = [
                (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(dim=-1)
                for a, b in zip(left_layers, right_layers)
            ]
            mean_cosine = torch.stack(layer_cosines, dim=-1).mean(dim=-1)
        pair_rows.append(torch.stack((final_cosine, mean_cosine), dim=-1))
    pair_tensor = torch.stack(pair_rows, dim=1)
    valid = availability.float()
    available_count = valid.sum(dim=1).clamp_min(1.0)
    global_features = torch.stack(
        (
            valid.sum(dim=1) / 4.0,
            torch.log1p(count_tensor.sum(dim=1).clamp_min(1.0)),
            (distance_tensor * valid).sum(dim=1) / available_count,
        ),
        dim=-1,
    )
    result = RouterFeatureBatch(
        modality_tensor,
        pair_tensor,
        global_features,
        availability,
        feature_schema=ROUTER_FEATURE_SCHEMA_NO_BATCH_PMGFA,
    )
    result.validate()
    return result, diagnostics


def sample_pmgfa_distance(
    layers: Sequence[torch.Tensor],
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
) -> torch.Tensor:
    """Return a per-sample PMGFA-like distance with fixed source statistics.

    The old PMGFA column compares a batch mean and batch standard deviation to
    source statistics.  This replacement compares each sample's layerwise
    hidden vectors to the fixed source mean and therefore does not depend on
    neighboring rows.  ``sample_source_distance`` remains the standardized RMS
    distance; keeping both columns gives the router magnitude and normalized
    views of the same observable shift.
    """
    stacked = torch.stack([value.float() for value in layers], dim=1)
    mean = source_mean.to(stacked.device)
    std = source_std.to(stacked.device).clamp_min(1e-4)
    if stacked.shape[1:] != mean.shape:
        raise ValueError("source statistics shape mismatch for sample PMGFA")
    centered = stacked - mean.unsqueeze(0)
    return centered.square().mean(dim=-1).sqrt().mean(dim=1)


def build_sample_invariant_router_features(
    features: Mapping[str, Sequence[torch.Tensor]],
    token_masks: Mapping[str, torch.Tensor],
    source: SourceShiftStatistics,
) -> tuple[RouterFeatureBatch, dict[str, torch.Tensor]]:
    """Build router features whose values are invariant to batch grouping.

    All modality and pair columns are sample-level.  Global columns summarize
    the same row rather than the current batch.  Source normalization is fixed
    by ``source`` and no target-batch mean or variance is computed.
    """
    if "Formula" not in features:
        raise ValueError("Formula features are required as the structural anchor")
    batch_size = features["Formula"][0].shape[0]
    anchor = F.normalize(features["Formula"][-1].float(), dim=-1)
    modality_rows = []
    pooled: dict[str, torch.Tensor] = {}
    availability_rows = []
    counts = []
    diagnostics: dict[str, torch.Tensor] = {}
    for name in ROUTED_MODALITIES:
        if name not in features or name not in token_masks or name not in source.means:
            modality_rows.append(anchor.new_zeros((batch_size, 4)))
            availability_rows.append(torch.zeros(batch_size, dtype=torch.bool, device=anchor.device))
            counts.append(anchor.new_zeros(batch_size))
            pooled[name] = anchor.new_zeros(anchor.shape)
            diagnostics[f"sample_pmgfa/{name}"] = anchor.new_zeros(batch_size)
            continue
        layers = features[name]
        current = F.normalize(layers[-1].float(), dim=-1)
        pooled[name] = current
        mask = token_masks[name].bool()
        available = mask.any(dim=1)
        count = mask.sum(dim=1).float()
        pmgfa = sample_pmgfa_distance(layers, source.means[name], source.stds[name])
        distance = sample_source_distance(layers, source.means[name], source.stds[name])
        compatibility = (current * anchor).sum(dim=-1)
        modality_rows.append(
            torch.stack((pmgfa, distance, compatibility, torch.log1p(count)), dim=-1)
        )
        availability_rows.append(available)
        counts.append(count)
        diagnostics[f"sample_pmgfa/{name}"] = pmgfa.detach()
        diagnostics[f"sample_distance/{name}"] = distance.detach()

    modality_tensor = torch.stack(modality_rows, dim=1)
    availability = torch.stack(availability_rows, dim=1)
    count_tensor = torch.stack(counts, dim=1)
    pair_rows = []
    for left, right in PAIR_INDICES:
        left_name = ROUTED_MODALITIES[left]
        right_name = ROUTED_MODALITIES[right]
        final_cosine = (pooled[left_name] * pooled[right_name]).sum(dim=-1)
        left_layers = features.get(left_name)
        right_layers = features.get(right_name)
        if left_layers is None or right_layers is None:
            mean_cosine = torch.zeros_like(final_cosine)
        else:
            layer_cosines = [
                (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(dim=-1)
                for a, b in zip(left_layers, right_layers)
            ]
            mean_cosine = torch.stack(layer_cosines, dim=-1).mean(dim=-1)
        pair_rows.append(torch.stack((final_cosine, mean_cosine), dim=-1))
    pair_tensor = torch.stack(pair_rows, dim=1)
    available_count = availability.float().sum(dim=1).clamp_min(1.0)
    valid = availability.float()
    global_features = torch.stack(
        (
            availability.float().sum(dim=1) / 4.0,
            torch.log1p(count_tensor.sum(dim=1).clamp_min(1.0)),
            (modality_tensor[:, :, 0] * valid).sum(dim=1) / available_count,
            (modality_tensor[:, :, 1] * valid).sum(dim=1) / available_count,
        ),
        dim=-1,
    )
    result = RouterFeatureBatch(
        modality_tensor,
        pair_tensor,
        global_features,
        availability,
        feature_schema=ROUTER_FEATURE_SCHEMA_SAMPLE_INVARIANT,
    )
    result.validate()
    return result, diagnostics

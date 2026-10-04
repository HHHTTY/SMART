"""Structured router over all non-empty spectroscopy modality subsets.

Formula is deliberately outside the action space and is always retained by
the caller.  The four routed modalities are encoded by a four-bit mask, which
makes the 15 actions stable across dataset construction, training and TTA.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn


ROUTED_MODALITIES = ("HNMR", "CNMR", "MSMS", "IR")
ROUTER_FEATURE_SCHEMA_BATCH = "batch_context_v1"
ROUTER_FEATURE_SCHEMA_SAMPLE_INVARIANT = "sample_invariant_v1"
ROUTER_FEATURE_SCHEMA_SOURCE_REFERENCE_FREE = "source_reference_free_v1"
ROUTER_FEATURE_SCHEMA_NO_BATCH_PMGFA = "no_batch_pmgfa_v1"
ROUTER_FEATURE_SCHEMAS = frozenset(
    (
        ROUTER_FEATURE_SCHEMA_BATCH,
        ROUTER_FEATURE_SCHEMA_SAMPLE_INVARIANT,
        ROUTER_FEATURE_SCHEMA_SOURCE_REFERENCE_FREE,
        ROUTER_FEATURE_SCHEMA_NO_BATCH_PMGFA,
    )
)
PAIR_INDICES = tuple(combinations(range(len(ROUTED_MODALITIES)), 2))
ACTION_BITS = tuple(range(1, 1 << len(ROUTED_MODALITIES)))
MODALITY_FEATURE_NAMES = (
    "batch_pmgfa",
    "sample_source_distance",
    "formula_compatibility",
    "log_token_count",
)
SAMPLE_INVARIANT_MODALITY_FEATURE_NAMES = (
    "sample_pmgfa",
    "sample_source_distance",
    "formula_compatibility",
    "log_token_count",
)
SOURCE_REFERENCE_FREE_MODALITY_FEATURE_NAMES = (
    "formula_compatibility",
    "log_token_count",
)
NO_BATCH_PMGFA_MODALITY_FEATURE_NAMES = (
    "sample_source_distance",
    "formula_compatibility",
    "log_token_count",
)
PAIR_FEATURE_NAMES = ("final_layer_cosine", "mean_layer_cosine")
GLOBAL_FEATURE_NAMES = (
    "available_modality_fraction",
    "log_total_token_count",
    "mean_batch_pmgfa",
    "mean_sample_source_distance",
)
SAMPLE_INVARIANT_GLOBAL_FEATURE_NAMES = (
    "available_modality_fraction",
    "log_total_token_count",
    "sample_pmgfa_mean",
    "sample_source_distance_mean",
)
SOURCE_REFERENCE_FREE_GLOBAL_FEATURE_NAMES = (
    "available_modality_fraction",
    "log_total_token_count",
)
NO_BATCH_PMGFA_GLOBAL_FEATURE_NAMES = (
    "available_modality_fraction",
    "log_total_token_count",
    "mean_sample_source_distance",
)


def router_feature_names(
    feature_schema: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if feature_schema == ROUTER_FEATURE_SCHEMA_BATCH:
        return MODALITY_FEATURE_NAMES, PAIR_FEATURE_NAMES, GLOBAL_FEATURE_NAMES
    if feature_schema == ROUTER_FEATURE_SCHEMA_SAMPLE_INVARIANT:
        return (
            SAMPLE_INVARIANT_MODALITY_FEATURE_NAMES,
            PAIR_FEATURE_NAMES,
            SAMPLE_INVARIANT_GLOBAL_FEATURE_NAMES,
        )
    if feature_schema == ROUTER_FEATURE_SCHEMA_SOURCE_REFERENCE_FREE:
        return (
            SOURCE_REFERENCE_FREE_MODALITY_FEATURE_NAMES,
            PAIR_FEATURE_NAMES,
            SOURCE_REFERENCE_FREE_GLOBAL_FEATURE_NAMES,
        )
    if feature_schema == ROUTER_FEATURE_SCHEMA_NO_BATCH_PMGFA:
        return (
            NO_BATCH_PMGFA_MODALITY_FEATURE_NAMES,
            PAIR_FEATURE_NAMES,
            NO_BATCH_PMGFA_GLOBAL_FEATURE_NAMES,
        )
    raise ValueError(f"unknown router feature schema: {feature_schema}")


def subset_from_bits(bits: int) -> tuple[str, ...]:
    if bits not in ACTION_BITS:
        raise ValueError("a routed subset must be a non-empty four-bit action")
    return tuple(
        name for index, name in enumerate(ROUTED_MODALITIES) if bits & (1 << index)
    )


ACTION_SUBSETS = tuple(subset_from_bits(bits) for bits in ACTION_BITS)


def bits_from_subset(subset: Iterable[str]) -> int:
    values = frozenset(str(value) for value in subset)
    unknown = values.difference(ROUTED_MODALITIES)
    if unknown:
        raise ValueError(f"unknown routed modalities: {sorted(unknown)}")
    if not values:
        raise ValueError("the empty routed subset is not a valid action")
    return sum(1 << index for index, name in enumerate(ROUTED_MODALITIES) if name in values)


def action_index(subset: Iterable[str]) -> int:
    return ACTION_BITS.index(bits_from_subset(subset))


def retained_input_modalities(subset: Iterable[str]) -> frozenset[str]:
    """Return the exact encoder modalities for an action, always including Formula."""
    values = subset_from_bits(bits_from_subset(subset))
    return frozenset(("Formula", *values))


def action_matrix(*, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(
        [
            [bool(bits & (1 << index)) for index in range(len(ROUTED_MODALITIES))]
            for bits in ACTION_BITS
        ],
        dtype=torch.bool,
        device=device,
    )


@dataclass(frozen=True)
class RouterFeatureBatch:
    """Leakage-free observable features consumed by the source-trained router."""

    modality: torch.Tensor
    pair: torch.Tensor
    global_features: torch.Tensor
    availability: torch.Tensor
    feature_schema: str = ROUTER_FEATURE_SCHEMA_BATCH

    def validate(self) -> None:
        if self.feature_schema not in ROUTER_FEATURE_SCHEMAS:
            raise ValueError(f"unknown router feature schema: {self.feature_schema}")
        if self.modality.ndim != 3 or self.modality.shape[1] != 4:
            raise ValueError("modality features must have shape [batch, 4, feature_dim]")
        if self.pair.ndim != 3 or self.pair.shape[1] != len(PAIR_INDICES):
            raise ValueError("pair features must have shape [batch, 6, feature_dim]")
        if self.global_features.ndim != 2:
            raise ValueError("global features must have shape [batch, feature_dim]")
        if self.availability.shape != self.modality.shape[:2]:
            raise ValueError("availability must have shape [batch, 4]")
        batch_size = self.modality.shape[0]
        if self.pair.shape[0] != batch_size or self.global_features.shape[0] != batch_size:
            raise ValueError("all router features must use the same batch dimension")
        if not self.availability.bool().any(dim=1).all():
            raise ValueError("every sample needs at least one available spectral modality")

    def to(self, device: torch.device | str) -> "RouterFeatureBatch":
        return RouterFeatureBatch(
            modality=self.modality.to(device),
            pair=self.pair.to(device),
            global_features=self.global_features.to(device),
            availability=self.availability.to(device),
            feature_schema=self.feature_schema,
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "modality": self.modality,
            "pair": self.pair,
            "global_features": self.global_features,
            "availability": self.availability,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, torch.Tensor]) -> "RouterFeatureBatch":
        return cls(
            modality=values["modality"],
            pair=values["pair"],
            global_features=values["global_features"],
            availability=values["availability"],
            feature_schema=str(values.get("feature_schema", ROUTER_FEATURE_SCHEMA_BATCH)),
        )


def pmgfa_support_mask(
    features: RouterFeatureBatch, upper_thresholds: torch.Tensor
) -> torch.Tensor:
    """Return rows whose per-modality PMGFA stays inside source support."""
    thresholds = torch.as_tensor(
        upper_thresholds,
        dtype=features.modality.dtype,
        device=features.modality.device,
    )
    if thresholds.shape != (len(ROUTED_MODALITIES),):
        raise ValueError("PMGFA support thresholds must have one value per modality")
    if features.feature_schema in {
        ROUTER_FEATURE_SCHEMA_SOURCE_REFERENCE_FREE,
        ROUTER_FEATURE_SCHEMA_NO_BATCH_PMGFA,
    }:
        raise ValueError("PMGFA support is undefined for source-reference-free features")
    pmgfa = features.modality[:, :, 0]
    supported = torch.isfinite(pmgfa) & pmgfa.le(thresholds[None])
    return (supported | ~features.availability.bool()).all(dim=1)


class ModalitySubsetRouter(nn.Module):
    """Score subsets as size bias + modality utilities + pair interactions."""

    def __init__(
        self,
        modality_feature_dim: int,
        pair_feature_dim: int,
        global_feature_dim: int,
        *,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        feature_schema: str = ROUTER_FEATURE_SCHEMA_BATCH,
    ) -> None:
        super().__init__()
        if min(modality_feature_dim, pair_feature_dim) < 1 or global_feature_dim < 0:
            raise ValueError("router feature dimensions are invalid")
        if temperature <= 0:
            raise ValueError("router temperature must be positive")
        if feature_schema not in ROUTER_FEATURE_SCHEMAS:
            raise ValueError(f"unknown router feature schema: {feature_schema}")
        self.modality_feature_dim = int(modality_feature_dim)
        self.pair_feature_dim = int(pair_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.feature_schema = str(feature_schema)
        self.modality_identity = nn.Parameter(torch.empty(4, hidden_dim))
        self.pair_identity = nn.Parameter(torch.empty(len(PAIR_INDICES), hidden_dim))
        self.modality_net = nn.Sequential(
            nn.Linear(modality_feature_dim + global_feature_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_net = nn.Sequential(
            nn.Linear(pair_feature_dim + global_feature_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.size_net = nn.Sequential(
            nn.Linear(max(1, global_feature_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.normal_(self.modality_identity, std=0.02)
        nn.init.normal_(self.pair_identity, std=0.02)
        self.register_buffer("actions", action_matrix(), persistent=True)
        pair_actions = []
        for action in self.actions:
            pair_actions.append([bool(action[left] and action[right]) for left, right in PAIR_INDICES])
        self.register_buffer("pair_actions", torch.tensor(pair_actions, dtype=torch.bool), persistent=True)
        self.register_buffer("action_sizes", self.actions.sum(dim=1).long(), persistent=True)
        self.register_buffer("modality_mean", torch.zeros(1, 4, modality_feature_dim))
        self.register_buffer("modality_std", torch.ones(1, 4, modality_feature_dim))
        self.register_buffer("pair_mean", torch.zeros(1, len(PAIR_INDICES), pair_feature_dim))
        self.register_buffer("pair_std", torch.ones(1, len(PAIR_INDICES), pair_feature_dim))
        self.register_buffer("global_mean", torch.zeros(1, global_feature_dim))
        self.register_buffer("global_std", torch.ones(1, global_feature_dim))

    @torch.no_grad()
    def set_feature_normalization(self, features: RouterFeatureBatch) -> None:
        features.validate()
        self.modality_mean.copy_(features.modality.mean(dim=0, keepdim=True))
        self.modality_std.copy_(
            features.modality.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        )
        self.pair_mean.copy_(features.pair.mean(dim=0, keepdim=True))
        self.pair_std.copy_(
            features.pair.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        )
        if self.global_feature_dim:
            self.global_mean.copy_(features.global_features.mean(dim=0, keepdim=True))
            self.global_std.copy_(
                features.global_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            )

    def _validate_dimensions(self, features: RouterFeatureBatch) -> None:
        features.validate()
        if features.feature_schema != self.feature_schema:
            raise ValueError(
                "router feature schema does not match checkpoint: "
                f"{features.feature_schema!r} != {self.feature_schema!r}"
            )
        if features.modality.shape[-1] != self.modality_feature_dim:
            raise ValueError("modality feature dimension does not match router checkpoint")
        if features.pair.shape[-1] != self.pair_feature_dim:
            raise ValueError("pair feature dimension does not match router checkpoint")
        if features.global_features.shape[-1] != self.global_feature_dim:
            raise ValueError("global feature dimension does not match router checkpoint")

    def logits(self, features: RouterFeatureBatch) -> torch.Tensor:
        self._validate_dimensions(features)
        batch_size = features.modality.shape[0]
        modality_features = (features.modality - self.modality_mean) / self.modality_std
        pair_features = (features.pair - self.pair_mean) / self.pair_std
        available = features.availability.bool()
        modality_features = modality_features.masked_fill(
            ~available.unsqueeze(-1), 0.0
        )
        for pair_index, (left, right) in enumerate(PAIR_INDICES):
            pair_available = available[:, left] & available[:, right]
            pair_features[:, pair_index] = pair_features[:, pair_index].masked_fill(
                ~pair_available.unsqueeze(-1), 0.0
            )

        # A near-constant source feature carries no learned signal. Keep it at
        # its training mean on target domains instead of extrapolating far
        # outside the source distribution (e.g. availability 1.0 -> 0.75).
        stable_global = self.global_std >= 1e-4
        stable_global_std = self.global_std.masked_fill(~stable_global, 1.0)
        global_features = (features.global_features - self.global_mean) / stable_global_std
        global_features = global_features.masked_fill(~stable_global, 0.0)
        global_modality = global_features[:, None, :].expand(-1, 4, -1)
        modality_identity = self.modality_identity[None].expand(batch_size, -1, -1)
        utilities = self.modality_net(
            torch.cat((modality_features, global_modality, modality_identity), dim=-1)
        ).squeeze(-1)

        global_pair = global_features[:, None, :].expand(-1, len(PAIR_INDICES), -1)
        pair_identity = self.pair_identity[None].expand(batch_size, -1, -1)
        interactions = self.pair_net(
            torch.cat((pair_features, global_pair, pair_identity), dim=-1)
        ).squeeze(-1)

        size_input = global_features
        if self.global_feature_dim == 0:
            size_input = utilities.new_zeros((batch_size, 1))
        size_bias = self.size_net(size_input)
        scores = torch.einsum("bm,am->ba", utilities, self.actions.to(utilities.dtype))
        scores = scores + torch.einsum(
            "bp,ap->ba", interactions, self.pair_actions.to(interactions.dtype)
        )
        scores = scores + size_bias[:, self.action_sizes - 1]

        valid_actions = (~self.actions[None] | available[:, None, :]).all(dim=-1)
        if not valid_actions.any(dim=1).all():
            raise ValueError("no valid non-empty router action for at least one sample")
        return scores.masked_fill(~valid_actions, float("-inf")) / self.temperature

    def forward(self, features: RouterFeatureBatch) -> torch.Tensor:
        return torch.softmax(self.logits(features), dim=-1)

    @torch.no_grad()
    def select(self, features: RouterFeatureBatch) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self(features)
        return probabilities.argmax(dim=-1), probabilities

    def checkpoint_payload(self) -> dict[str, object]:
        modality_feature_names, pair_feature_names, global_feature_names = (
            router_feature_names(self.feature_schema)
        )
        return {
            "state_dict": self.state_dict(),
            "modality_feature_dim": self.modality_feature_dim,
            "pair_feature_dim": self.pair_feature_dim,
            "global_feature_dim": self.global_feature_dim,
            "hidden_dim": self.hidden_dim,
            "temperature": self.temperature,
            "feature_schema": self.feature_schema,
            "modalities": ROUTED_MODALITIES,
            "action_subsets": ACTION_SUBSETS,
            "modality_feature_names": modality_feature_names,
            "pair_feature_names": pair_feature_names,
            "global_feature_names": global_feature_names,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, object]) -> "ModalitySubsetRouter":
        if tuple(payload.get("modalities", ())) != ROUTED_MODALITIES:
            raise ValueError("router checkpoint modality order is incompatible")
        feature_schema = str(payload.get("feature_schema", ROUTER_FEATURE_SCHEMA_BATCH))
        modality_names, pair_names, global_names = router_feature_names(feature_schema)
        expected_schema = {
            "modality_feature_names": modality_names,
            "pair_feature_names": pair_names,
            "global_feature_names": global_names,
        }
        for key, expected in expected_schema.items():
            if tuple(payload.get(key, ())) != expected:
                raise ValueError(f"router checkpoint {key} is incompatible")
        model = cls(
            int(payload["modality_feature_dim"]),
            int(payload["pair_feature_dim"]),
            int(payload["global_feature_dim"]),
            hidden_dim=int(payload.get("hidden_dim", 64)),
            temperature=float(payload.get("temperature", 1.0)),
            feature_schema=feature_schema,
        )
        model.load_state_dict(payload["state_dict"], strict=True)  # type: ignore[arg-type]
        return model

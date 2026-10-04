"""Anchor-Preserving Progressive Modality Bridging (APPMB).

The block is deliberately independent of :class:`HFWrapper`: it operates on
the already concatenated multimodal embeddings and therefore leaves the
source checkpoint, shared encoder and decoder untouched.  Formula and CNMR
are structural anchors.  HNMR, MSMS and IR receive separate low-rank residual
bridges and sample-wise gates.

No target labels are consumed by this module.  The gate features are computed
from observed embeddings only: anchor compatibility, two-view stability and
the valid-token fraction.  Padding and excluded modalities are exact identity
paths, and token positions are never compacted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


Span = tuple[str, int, int]


@dataclass
class BridgeDiagnostics:
    modalities: tuple[str, ...]
    gates: torch.Tensor
    anchor_shift: torch.Tensor
    instability: torch.Tensor
    valid_fraction: torch.Tensor
    correction_ratio: torch.Tensor
    availability: torch.Tensor
    token_gates: Mapping[str, torch.Tensor]

    def detached(self) -> dict[str, torch.Tensor | tuple[str, ...]]:
        return {
            "modalities": self.modalities,
            "gates": self.gates.detach(),
            "anchor_shift": self.anchor_shift.detach(),
            "instability": self.instability.detach(),
            "valid_fraction": self.valid_fraction.detach(),
            "correction_ratio": self.correction_ratio.detach(),
            "availability": self.availability.detach(),
            "token_gates": {
                name: value.detach() for name, value in self.token_gates.items()
            },
        }


class _ModalityResidualBridge(nn.Module):
    """One zero-output low-rank residual bridge plus a reliability gate."""

    def __init__(self, d_model: int, rank: int, gate_hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.anchor_norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.anchor_down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=True)
        self.affine_log_scale = nn.Parameter(torch.zeros(d_model))
        self.affine_bias = nn.Parameter(torch.zeros(d_model))
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model + 3),
            nn.Linear(2 * d_model + 3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        # The source checkpoint is reproduced exactly at construction time.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def residual(self, tokens: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tokens)
        anchor_feature = self.anchor_down(self.anchor_norm(anchor))[:, None, :]
        low_rank = self.up(F.gelu(self.down(normalized) + anchor_feature))
        scale = 0.10 * torch.tanh(self.affine_log_scale)
        bias = 0.10 * torch.tanh(self.affine_bias)
        return low_rank + tokens * scale + bias


class AnchorPreservingProgressiveBridge(nn.Module):
    """Modality-local residual repair controlled by an observed-data gate."""

    def __init__(
        self,
        d_model: int = 512,
        rank: int = 16,
        gate_hidden: int = 64,
        modalities: Sequence[str] = ("HNMR", "MSMS", "IR"),
        anchor_modalities: Sequence[str] = ("Formula", "CNMR"),
        max_delta_ratio: float = 0.20,
        modality_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__()
        if rank <= 0 or gate_hidden <= 0:
            raise ValueError("rank and gate_hidden must be positive")
        if not 0.0 < max_delta_ratio <= 1.0:
            raise ValueError("max_delta_ratio must lie in (0, 1]")
        self.d_model = int(d_model)
        self.modalities = tuple(str(value) for value in modalities)
        self.anchor_modalities = tuple(str(value) for value in anchor_modalities)
        self.max_delta_ratio = float(max_delta_ratio)
        self.modality_map = dict(modality_map or {})
        self.bridges = nn.ModuleDict(
            {
                name: _ModalityResidualBridge(self.d_model, rank, gate_hidden)
                for name in self.modalities
            }
        )

    def canonical_name(self, data_name: str) -> str:
        return str(self.modality_map.get(str(data_name), str(data_name)))

    @staticmethod
    def _validate_spans(spans: Sequence[Span], sequence_length: int) -> tuple[Span, ...]:
        result = tuple((str(name), int(start), int(end)) for name, start, end in spans)
        if not result:
            raise ValueError("at least one modality span is required")
        cursor = 0
        for _name, start, end in result:
            if start != cursor or end < start:
                raise ValueError("modality spans must be contiguous and ordered")
            cursor = end
        if cursor != sequence_length:
            raise ValueError("modality spans do not cover the input sequence")
        return result

    @staticmethod
    def _masked_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _masks(self, attention_mask: torch.Tensor, spans: Sequence[Span]) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        for data_name, start, end in spans:
            canonical = self.canonical_name(data_name)
            current = torch.zeros_like(attention_mask, dtype=torch.bool)
            current[:, start:end] = attention_mask[:, start:end].bool()
            masks[canonical] = masks.get(canonical, torch.zeros_like(current)) | current
        return masks

    def _anchor(
        self,
        embeddings: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
        *,
        anchor_permutation: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = []
        available = []
        for name in self.anchor_modalities:
            mask = masks.get(name)
            if mask is None:
                continue
            pooled.append(self._masked_pool(embeddings, mask))
            available.append(mask.any(dim=1))
        if not pooled:
            return embeddings.new_zeros(embeddings.shape[0], embeddings.shape[-1]), torch.zeros(
                embeddings.shape[0], dtype=torch.bool, device=embeddings.device
            )
        values = torch.stack(pooled, dim=0)
        observed = torch.stack(available, dim=0)
        weights = observed.to(embeddings.dtype).unsqueeze(-1)
        anchor = (values * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1.0)
        anchor_available = observed.any(dim=0)
        if anchor_permutation is not None:
            anchor = anchor[anchor_permutation]
            anchor_available = anchor_available[anchor_permutation]
        return anchor, anchor_available

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_spans: Sequence[Span],
        *,
        stability_embeddings: Optional[torch.Tensor] = None,
        stability_attention_mask: Optional[torch.Tensor] = None,
        enabled_modalities: Optional[Sequence[str]] = None,
        fixed_gate: Optional[float] = None,
        anchor_permutation: Optional[torch.Tensor] = None,
        detach_gate_features: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, BridgeDiagnostics]:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.d_model:
            raise ValueError("embeddings must have shape [batch, sequence, d_model]")
        if attention_mask.shape != embeddings.shape[:2]:
            raise ValueError("attention_mask shape does not match embeddings")
        spans = self._validate_spans(modality_spans, embeddings.shape[1])
        masks = self._masks(attention_mask, spans)
        stability_embeddings = embeddings if stability_embeddings is None else stability_embeddings
        stability_attention_mask = (
            attention_mask if stability_attention_mask is None else stability_attention_mask
        )
        if stability_embeddings.shape != embeddings.shape:
            raise ValueError("stability_embeddings must match embeddings")
        if stability_attention_mask.shape != attention_mask.shape:
            raise ValueError("stability_attention_mask must match attention_mask")
        stability_masks = self._masks(stability_attention_mask, spans)
        anchor, anchor_available = self._anchor(
            embeddings, masks, anchor_permutation=anchor_permutation
        )
        enabled = set(self.modalities if enabled_modalities is None else enabled_modalities)
        unknown = enabled.difference(self.modalities)
        if unknown:
            raise ValueError(f"unknown bridge modalities: {sorted(unknown)}")

        output = embeddings.clone()
        batch = embeddings.shape[0]
        scalar_shape = (batch, len(self.modalities))
        gates = embeddings.new_zeros(scalar_shape)
        shifts = embeddings.new_zeros(scalar_shape)
        instabilities = embeddings.new_zeros(scalar_shape)
        fractions = embeddings.new_zeros(scalar_shape)
        correction_ratios = embeddings.new_zeros(scalar_shape)
        availability = torch.zeros(scalar_shape, dtype=torch.bool, device=embeddings.device)
        token_gates: dict[str, torch.Tensor] = {}

        for index, name in enumerate(self.modalities):
            mask = masks.get(name)
            if mask is None:
                continue
            observed = mask.any(dim=1) & anchor_available
            availability[:, index] = observed
            weak = self._masked_pool(embeddings, mask)
            stable_mask = stability_masks.get(name, mask)
            stable = self._masked_pool(stability_embeddings, stable_mask)
            shift = 1.0 - F.cosine_similarity(weak, anchor, dim=-1, eps=1e-6)
            instability = 1.0 - F.cosine_similarity(weak, stable, dim=-1, eps=1e-6)
            fraction = mask.float().sum(dim=1) / mask.shape[1]
            if name == "IR":
                anchor_tokens = anchor[:, None, :].expand_as(embeddings)
                token_shift = 1.0 - F.cosine_similarity(
                    embeddings, anchor_tokens, dim=-1, eps=1e-6
                )
                token_instability = 1.0 - F.cosine_similarity(
                    embeddings, stability_embeddings, dim=-1, eps=1e-6
                )
                token_fraction = fraction[:, None].expand_as(token_shift)
                feature = torch.cat(
                    [
                        anchor_tokens,
                        embeddings,
                        token_shift[:, :, None],
                        token_instability[:, :, None],
                        token_fraction[:, :, None],
                    ],
                    dim=-1,
                )
                if detach_gate_features:
                    feature = feature.detach()
                learned_token_gate = torch.sigmoid(
                    self.bridges[name].gate(feature)
                ).squeeze(-1)
                token_gate = (
                    learned_token_gate
                    if fixed_gate is None
                    else learned_token_gate.new_full(
                        learned_token_gate.shape, float(fixed_gate)
                    )
                )
                token_gate = (
                    token_gate
                    * mask.to(token_gate.dtype)
                    * observed[:, None].to(token_gate.dtype)
                )
                gate = token_gate.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                token_gates[name] = token_gate.masked_fill(~mask, float("nan"))
            else:
                feature = torch.cat(
                    [anchor, weak, shift[:, None], instability[:, None], fraction[:, None]],
                    dim=-1,
                )
                if detach_gate_features:
                    feature = feature.detach()
                learned_gate = torch.sigmoid(self.bridges[name].gate(feature)).squeeze(-1)
                gate = (
                    learned_gate
                    if fixed_gate is None
                    else learned_gate.new_full(learned_gate.shape, float(fixed_gate))
                )
                gate = gate * observed.to(gate.dtype)
                token_gate = gate[:, None].expand(mask.shape) * mask.to(gate.dtype)
                token_gates[name] = token_gate.masked_fill(~mask, float("nan"))
            shifts[:, index] = shift
            instabilities[:, index] = instability
            fractions[:, index] = fraction
            gates[:, index] = gate
            if name not in enabled:
                continue

            raw_delta = self.bridges[name].residual(embeddings, anchor)
            raw_delta = raw_delta * mask.unsqueeze(-1).to(raw_delta.dtype)
            reference = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
            # The epsilon is inside sqrt so the exact zero-initialized branch
            # has a finite derivative on its first optimization step.
            delta_norm = (raw_delta.float().pow(2).sum(dim=(1, 2)) + 1e-12).sqrt()
            reference_norm = (
                reference.float().pow(2).sum(dim=(1, 2)) + 1e-12
            ).sqrt().clamp_min(1e-6)
            bound = self.max_delta_ratio * reference_norm
            scale = (bound / delta_norm.clamp_min(1e-6)).clamp(max=1.0)
            bounded = raw_delta * scale[:, None, None].to(raw_delta.dtype)
            correction = bounded * token_gate[:, :, None]
            output = output + correction
            correction_ratios[:, index] = (
                (correction.float().pow(2).sum(dim=(1, 2)) + 1e-12).sqrt()
                / reference_norm
            )

        diagnostics = BridgeDiagnostics(
            modalities=self.modalities,
            gates=gates,
            anchor_shift=shifts,
            instability=instabilities,
            valid_fraction=fractions,
            correction_ratio=correction_ratios,
            availability=availability,
            token_gates=token_gates,
        )
        return (output, diagnostics) if return_diagnostics else output

    def set_trainable_modalities(
        self,
        modalities: Sequence[str],
        *,
        gate_and_affine_only: bool = False,
        gate_only: bool = False,
    ) -> None:
        """Apply dropped-modality-only trainability for source phases or TTA."""
        if gate_and_affine_only and gate_only:
            raise ValueError("gate_and_affine_only and gate_only are mutually exclusive")
        selected = set(modalities)
        unknown = selected.difference(self.modalities)
        if unknown:
            raise ValueError(f"unknown bridge modalities: {sorted(unknown)}")
        for name, bridge in self.bridges.items():
            active = name in selected
            for parameter in bridge.parameters():
                parameter.requires_grad_(active)
            if active and gate_and_affine_only:
                for parameter in bridge.norm.parameters():
                    parameter.requires_grad_(False)
                for parameter in bridge.anchor_norm.parameters():
                    parameter.requires_grad_(False)
                for parameter in bridge.down.parameters():
                    parameter.requires_grad_(False)
                for parameter in bridge.anchor_down.parameters():
                    parameter.requires_grad_(False)
                for parameter in bridge.up.parameters():
                    parameter.requires_grad_(False)
            if active and gate_only:
                for parameter in bridge.parameters():
                    parameter.requires_grad_(False)
                for parameter in bridge.gate.parameters():
                    parameter.requires_grad_(True)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

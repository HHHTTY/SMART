"""Shared-encoder adaptation primitives for the TAHCD transfer experiment.

The paper assumes one encoder per modality.  This implementation keeps the
multitask-ms backbone unchanged and applies the equivalent operations to
contiguous modality spans in the *shared* encoder output:

* Formula/CNMR provide a per-sample anchor representation;
* HNMR/MSMS/IR receive a slack-bounded repair towards that anchor;
* only a scalar fast coefficient per modality is updated at test time.

This is an explicitly named *TAHCD-inspired shared-encoder transfer*.  It is
not a claim that the independent-encoder ASSA/SACA implementation from the
paper has been reproduced verbatim.

The block is intentionally stateless across samples.  ``fast_state`` is passed
by the caller and is never written into module parameters during inference.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


Span = tuple[str, int, int]


class TAHCDSharedBlock(nn.Module):
    """TAHCD-style slack repair for a shared multimodal encoder."""

    def __init__(
        self,
        d_model: int,
        *,
        modalities: Sequence[str] = ("HNMR", "MSMS", "IR"),
        anchor_modalities: Sequence[str] = ("Formula", "CNMR"),
        modality_map: Optional[Mapping[str, str]] = None,
        slack: float = 0.10,
        max_alpha: float = 0.50,
        min_alpha: Optional[float] = None,
        distance_temperature: float = 0.10,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = int(d_model)
        self.modalities = tuple(str(name) for name in modalities)
        self.anchor_modalities = tuple(str(name) for name in anchor_modalities)
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be non-empty and unique")
        if not self.anchor_modalities:
            raise ValueError("anchor_modalities must be non-empty")
        if float(slack) < 0.0:
            raise ValueError("slack must be non-negative")
        if float(max_alpha) <= 0.0:
            raise ValueError("max_alpha must be positive")
        if min_alpha is None:
            min_alpha = -float(max_alpha)
        if float(min_alpha) > 0.0 or float(min_alpha) >= float(max_alpha):
            raise ValueError("min_alpha must be <= 0 and smaller than max_alpha")
        if float(distance_temperature) <= 0.0:
            raise ValueError("distance_temperature must be positive")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.modality_map = {
            str(key): str(value) for key, value in (modality_map or {}).items()
        }
        self.slack = float(slack)
        self.max_alpha = float(max_alpha)
        self.min_alpha = float(min_alpha)
        self.distance_temperature = float(distance_temperature)
        self.eps = float(eps)
        # This is a slow initialization only.  The default is identity; the
        # source backbone therefore remains an exact no-update baseline.
        self.a_meta = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in self.modalities}
        )

    def _canonical_name(self, data_name: str) -> str:
        name = str(data_name)
        if name in self.modality_map:
            return self.modality_map[name]
        for canonical, mapped_name in self.modality_map.items():
            if mapped_name == name:
                return canonical
        return name

    @staticmethod
    def _validate_spans(spans: Sequence[Span], sequence_length: int) -> tuple[Span, ...]:
        normalised = tuple((str(name), int(start), int(end)) for name, start, end in spans)
        previous = 0
        for _name, start, end in normalised:
            if start != previous or start < 0 or end <= start or end > sequence_length:
                raise ValueError("modality spans must be contiguous and cover the sequence")
            previous = end
        if previous != sequence_length:
            raise ValueError("modality spans must cover the complete sequence")
        return normalised

    def initial_fast_state(
        self,
        batch_size: int,
        *,
        detach: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, torch.Tensor]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        state: dict[str, torch.Tensor] = {}
        for name in self.modalities:
            value = self.a_meta[name]
            if detach:
                value = value.detach()
            value = value.to(
                device=device if device is not None else value.device,
                dtype=dtype if dtype is not None else value.dtype,
            )
            value = value.reshape(1, 1).expand(int(batch_size), 1)
            state[name] = value.clone() if detach else value
        return state

    def _state_for(
        self,
        fast_state: Optional[Mapping[str, torch.Tensor]],
        name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = self.a_meta[name] if fast_state is None else fast_state.get(name, self.a_meta[name])
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"fast_state[{name!r}] must be a tensor")
        value = value.to(device=device, dtype=dtype)
        if value.ndim == 0 or value.shape == (1,):
            return value.reshape(1, 1).expand(batch_size, 1)
        if value.ndim == 1 and value.shape[0] == batch_size:
            return value.reshape(batch_size, 1)
        if value.shape == (batch_size, 1):
            return value
        raise ValueError(
            f"fast_state[{name!r}] has shape {tuple(value.shape)}; expected scalar or {(batch_size, 1)}"
        )

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _weighted_anchor(
        pooled: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        names: Sequence[str],
        hidden: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a per-sample anchor without averaging missing modalities.

        A padded/missing span is represented by an all-zero pooled vector.  It
        must not count as an observed anchor, otherwise a sample with only one
        anchor is silently shrunk by a factor of two.  When no configured
        anchor is observed in a row, the valid sequence pool is used as the
        structural fallback and the row remains marked unavailable for repair.
        """
        values: list[torch.Tensor] = []
        availability: list[torch.Tensor] = []
        for name in names:
            if name in pooled and name in masks:
                values.append(pooled[name])
                availability.append(masks[name].any(dim=1))
        if not values:
            # No configured anchor span is present at all (for example after
            # an explicit modality exclusion).  Keep a finite fallback vector
            # for diagnostics, but disable every repair row.
            return TAHCDSharedBlock._pool(hidden, valid), torch.zeros(
                hidden.shape[0], dtype=torch.bool, device=hidden.device
            )
        value_stack = torch.stack(values, dim=0)
        available_stack = torch.stack(availability, dim=0)
        weights = available_stack.to(dtype=hidden.dtype).unsqueeze(-1)
        count = weights.sum(dim=0)
        anchor = (value_stack * weights).sum(dim=0) / count.clamp_min(1.0)
        # The fallback is only for numerical stability/diagnostics.  It is not
        # used to repair a sample with no observed configured anchor.
        fallback = TAHCDSharedBlock._pool(hidden, valid)
        anchor = torch.where((count > 0.0), anchor, fallback)
        return anchor, count.squeeze(-1).gt(0.0)

    def _span_masks(
        self,
        attention_mask: torch.Tensor,
        spans: Sequence[Span],
    ) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        for data_name, start, end in spans:
            canonical = self._canonical_name(data_name)
            masks[canonical] = attention_mask[:, start:end].bool()
        return masks

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_spans: Sequence[Span],
        *,
        fast_state: Optional[Mapping[str, torch.Tensor]] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        if hidden.shape[-1] != self.d_model:
            raise ValueError(f"hidden last dimension is {hidden.shape[-1]}, expected {self.d_model}")
        if attention_mask.shape != hidden.shape[:2]:
            raise ValueError("attention_mask must match hidden's first two dimensions")
        spans = self._validate_spans(modality_spans, hidden.shape[1])
        valid = attention_mask.to(device=hidden.device).bool()
        batch_size = hidden.shape[0]
        masks: dict[str, torch.Tensor] = {}
        pooled: dict[str, torch.Tensor] = {}
        for data_name, start, end in spans:
            canonical = self._canonical_name(data_name)
            mask = torch.zeros_like(valid)
            mask[:, start:end] = valid[:, start:end]
            masks[canonical] = mask
            pooled[canonical] = self._pool(hidden, mask)

        anchor, anchor_available = self._weighted_anchor(
            pooled,
            masks,
            self.anchor_modalities,
            hidden,
            valid,
        )

        output = hidden.clone()
        distances = hidden.new_zeros(batch_size, len(self.modalities))
        gates = hidden.new_zeros(batch_size, len(self.modalities))
        alphas = hidden.new_zeros(batch_size, len(self.modalities))
        correction_ratios = hidden.new_zeros(batch_size, len(self.modalities))
        for index, name in enumerate(self.modalities):
            if name not in masks:
                continue
            mask = masks[name]
            z = pooled[name]
            distance = 1.0 - F.cosine_similarity(z, anchor, dim=-1, eps=self.eps)
            gate = torch.sigmoid((distance - self.slack) / self.distance_temperature)
            state = self._state_for(fast_state, name, batch_size, hidden.device, hidden.dtype)
            alpha = state.squeeze(-1).clamp(self.min_alpha, self.max_alpha)
            # Anchor availability is not a target label; it is a structural
            # validity mask produced solely from observed modality tokens.
            modality_available = mask.any(dim=1)
            strength = (
                alpha
                * gate
                * anchor_available.to(dtype=hidden.dtype)
                * modality_available.to(dtype=hidden.dtype)
            )
            correction = (anchor.unsqueeze(1) - hidden) * strength[:, None, None]
            reference = hidden * mask.unsqueeze(-1).to(hidden.dtype)
            delta = correction * mask.unsqueeze(-1).to(hidden.dtype)
            delta_norm = (delta.float().pow(2).sum(dim=(1, 2)) + self.eps**2).sqrt()
            ref_norm = reference.float().pow(2).sum(dim=(1, 2)).sqrt().clamp_min(self.eps)
            scale = (self.max_alpha * ref_norm / delta_norm.clamp_min(self.eps)).clamp(max=1.0)
            delta = delta * torch.where(delta_norm > self.eps, scale, torch.ones_like(scale))[:, None, None].to(delta.dtype)
            output = output + delta
            distances[:, index] = distance.detach()
            gates[:, index] = gate.detach()
            alphas[:, index] = alpha.detach()
            correction_ratios[:, index] = (
                delta.float().pow(2).sum(dim=(1, 2)) + self.eps**2
            ).sqrt() / ref_norm

        if not return_diagnostics:
            return output
        return output, {
            "distances": distances,
            "gates": gates,
            "alphas": alphas,
            "correction_ratios": correction_ratios,
            "anchor_available": anchor_available.detach(),
            "anchor": anchor.detach(),
            "modality_available": torch.stack(
                [masks.get(name, torch.zeros_like(valid)).any(dim=1) for name in self.modalities],
                dim=1,
            ).detach(),
        }

    def consistency_loss(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_spans: Sequence[Span],
        *,
        fast_state: Mapping[str, torch.Tensor],
        prior_weight: float = 0.01,
    ) -> torch.Tensor:
        """Compute the label-free TTCE objective for a temporary state."""
        adapted, diagnostics = self(
            hidden,
            attention_mask,
            modality_spans,
            fast_state=fast_state,
            return_diagnostics=True,
        )
        spans = self._validate_spans(modality_spans, hidden.shape[1])
        valid = attention_mask.bool()
        pooled: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for data_name, start, end in spans:
            canonical = self._canonical_name(data_name)
            mask = torch.zeros_like(valid)
            mask[:, start:end] = valid[:, start:end]
            masks[canonical] = mask
            pooled[canonical] = self._pool(adapted, mask)
        anchor, anchor_available = self._weighted_anchor(
            pooled,
            masks,
            self.anchor_modalities,
            adapted,
            valid,
        )
        losses = []
        loss_weights = []
        for name in self.modalities:
            if name not in pooled:
                continue
            distance = 1.0 - F.cosine_similarity(pooled[name], anchor.detach(), dim=-1, eps=self.eps)
            available = masks[name].any(dim=1) & anchor_available
            losses.append(F.relu(distance - self.slack).pow(2) * available.to(distance.dtype))
            loss_weights.append(available.to(distance.dtype))
        if not losses:
            return adapted.sum() * 0.0
        loss_values = torch.stack(losses, dim=0)
        weight_values = torch.stack(loss_weights, dim=0)
        loss = loss_values.sum() / weight_values.sum().clamp_min(1.0)
        if prior_weight:
            # ``diagnostics["alphas"]`` is intentionally detached for logging.
            # Use the original fast-state tensors here so the prior really
            # contributes a gradient to the temporary adaptation variables.
            prior_terms = []
            prior_weights = []
            for name in self.modalities:
                state = self._state_for(
                    fast_state,
                    name,
                    hidden.shape[0],
                    hidden.device,
                    hidden.dtype,
                )
                available = masks.get(name, torch.zeros_like(valid)).any(dim=1)
                prior_terms.append(state.squeeze(-1).pow(2) * available.to(state.dtype))
                prior_weights.append(available.to(state.dtype))
            if prior_terms:
                prior = torch.stack(prior_terms, dim=0)
                prior = prior.sum() / torch.stack(prior_weights, dim=0).sum().clamp_min(1.0)
                loss = loss + float(prior_weight) * prior
        return loss

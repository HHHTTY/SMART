"""ExposureTTT: pre-fusion low-rank fast-weight adaptation.

The module is deliberately small and sequence-length preserving.  Slow
weights learn one correction basis per spectroscopy modality; the per-sample
fast state only selects a point in that basis during an inner update.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


Span = tuple[str, int, int]


class ExposureTTTBlock(nn.Module):
    """Low-rank modality correction with an episodic fast state.

    Formula and unknown modalities are copied byte-for-byte into the output.
    Only the configured spectroscopy modalities can receive a correction.
    ``a_meta`` is a slow, learned initialization; callers pass a temporary
    ``fast_state`` to evaluate an adapted molecule without mutating parameters.
    """

    def __init__(
        self,
        d_model: int,
        *,
        rank: int = 8,
        modalities: Sequence[str] = ("HNMR", "CNMR", "MSMS", "IR"),
        modality_map: Optional[Mapping[str, str]] = None,
        bound_ratio: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError("d_model must be positive")
        if int(rank) <= 0 or int(rank) > int(d_model):
            raise ValueError("rank must be in [1, d_model]")
        if not 0.0 < float(bound_ratio):
            raise ValueError("bound_ratio must be positive")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.d_model = int(d_model)
        self.rank = int(rank)
        self.modalities = tuple(str(name) for name in modalities)
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be a non-empty sequence of unique names")
        self.modality_index = {name: index for index, name in enumerate(self.modalities)}
        self.modality_map = {
            str(key): str(value) for key, value in (modality_map or {}).items()
        }
        self.bound_ratio = float(bound_ratio)
        self.eps = float(eps)

        # U_m:[D,r], V_m:[r,D], and a_meta_m:[r].  ParameterDict keys are
        # stable and therefore make checkpoints unambiguous across runs.
        self.slow_u = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(self.d_model, self.rank)) for name in self.modalities}
        )
        self.slow_v = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(self.rank, self.d_model)) for name in self.modalities}
        )
        self.a_meta = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(self.rank)) for name in self.modalities}
        )
        self.input_norm = nn.LayerNorm(self.d_model)
        for name in self.modalities:
            nn.init.normal_(self.slow_u[name], mean=0.0, std=0.02)
            nn.init.normal_(self.slow_v[name], mean=0.0, std=0.02)

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
                raise ValueError("modality spans must be contiguous and cover the input sequence")
            previous = end
        if previous != sequence_length:
            raise ValueError("modality spans must cover the complete input sequence")
        return normalised

    def initial_fast_state(
        self,
        batch_size: int,
        *,
        detach: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, torch.Tensor]:
        """Return a fresh per-batch state; no in-place parameter mutation occurs."""
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
            state[name] = value.unsqueeze(0).expand(int(batch_size), -1)
            if detach:
                state[name] = state[name].clone()
        return state

    def _state_for(
        self,
        fast_state: Optional[Mapping[str, torch.Tensor]],
        name: str,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = self.a_meta[name] if fast_state is None else fast_state.get(name, self.a_meta[name])
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"fast_state[{name!r}] must be a tensor")
        value = value.to(device=device, dtype=dtype)
        if value.ndim == 1 and value.shape[0] == self.rank:
            return value.unsqueeze(0).expand(batch_size, -1)
        if value.shape == (batch_size, self.rank):
            return value
        raise ValueError(
            f"fast_state[{name!r}] has shape {tuple(value.shape)}; "
            f"expected {(self.rank,)} or {(batch_size, self.rank)}"
        )

    def _bound_delta(
        self,
        delta: torch.Tensor,
        reference: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a per-sample Frobenius trust region and return its ratio."""
        valid_f = valid.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
        delta = delta * valid_f
        reference = reference * valid_f
        # ``sqrt(sum(delta**2))`` has an undefined derivative at the exact
        # zero initialization used by ``a_meta``.  Adding eps inside the
        # square root keeps the identity path numerically finite while still
        # enforcing the same trust-region bound for non-zero corrections.
        delta_norm = (
            delta.float().pow(2).sum(dim=(1, 2), keepdim=True) + self.eps**2
        ).sqrt()
        reference_norm = reference.float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
        allowed = reference_norm * self.bound_ratio
        scale = torch.where(
            delta_norm > self.eps,
            (allowed / delta_norm.clamp_min(self.eps)).clamp(max=1.0),
            torch.ones_like(delta_norm),
        )
        bounded = delta * scale.to(dtype=delta.dtype)
        ratio = bounded.float().pow(2).sum(dim=(1, 2)).sqrt() / reference_norm.squeeze(
            -1
        ).squeeze(-1).clamp_min(self.eps)
        return bounded, ratio

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_spans: Sequence[Span],
        *,
        fast_state: Optional[Mapping[str, torch.Tensor]] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply ``U_m diag(a_m) V_m LN(Z_m)`` to spectroscopy spans."""
        if inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [batch, sequence, d_model]")
        if inputs_embeds.shape[-1] != self.d_model:
            raise ValueError(
                f"inputs_embeds last dimension is {inputs_embeds.shape[-1]}, expected {self.d_model}"
            )
        if attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("attention_mask must match the first two input dimensions")
        spans = self._validate_spans(modality_spans, inputs_embeds.shape[1])
        valid = attention_mask.to(device=inputs_embeds.device).bool()
        batch_size = inputs_embeds.shape[0]
        output = inputs_embeds.clone()
        ratios = inputs_embeds.new_zeros(batch_size, len(self.modalities))
        fast_norms = inputs_embeds.new_zeros(batch_size, len(self.modalities))

        for data_name, start, end in spans:
            canonical = self._canonical_name(data_name)
            if canonical not in self.modality_index:
                # Formula and any non-spectroscopic side channel are exact anchors.
                continue
            index = self.modality_index[canonical]
            segment = inputs_embeds[:, start:end]
            segment_valid = valid[:, start:end]
            state = self._state_for(
                fast_state,
                canonical,
                batch_size,
                device=segment.device,
                dtype=segment.dtype,
            )
            normalised = self.input_norm(segment.float())
            projected = torch.einsum(
                "btd,rd->btr", normalised, self.slow_v[canonical].float()
            )
            projected = projected * state.float().unsqueeze(1)
            correction = torch.einsum(
                "btr,dr->btd", projected, self.slow_u[canonical].float()
            ).to(dtype=segment.dtype)
            correction, ratio = self._bound_delta(correction, segment, segment_valid)
            updated = segment + correction
            output[:, start:end] = torch.where(
                segment_valid.unsqueeze(-1), updated, segment
            )
            ratios[:, index] = ratio.to(dtype=ratios.dtype)
            fast_norms[:, index] = state.float().norm(dim=-1).to(dtype=fast_norms.dtype)

        if not return_diagnostics:
            return output
        return output, {
            "correction_ratio": ratios.detach(),
            "fast_state_norm": fast_norms.detach(),
        }

    def build_exposure_masks(
        self,
        attention_mask: torch.Tensor,
        modality_spans: Sequence[Span],
        modality: str,
        ratios: Sequence[float] = (0.25, 0.5, 1.0),
        *,
        deterministic: bool = False,
    ) -> list[tuple[str, torch.Tensor]]:
        """Create subset masks for one modality while preserving shape.

        Training uses a random valid-token subset.  Validation can request a
        deterministic, evenly spaced subset so repeated evaluations measure
        model changes rather than a new random mask on every pass.
        """
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        spans = self._validate_spans(modality_spans, attention_mask.shape[1])
        target = self._canonical_name(modality)
        ratios_float = tuple(float(value) for value in ratios)
        if not ratios_float or any(value <= 0.0 or value > 1.0 for value in ratios_float):
            raise ValueError("exposure ratios must lie in (0, 1]")
        full = attention_mask.bool()
        views: list[tuple[str, torch.Tensor]] = []
        for ratio in ratios_float:
            view = full.clone()
            found = False
            for data_name, start, end in spans:
                if self._canonical_name(data_name) != target:
                    continue
                found = True
                source = full[:, start:end]
                limited = torch.zeros_like(source)
                for row in range(source.shape[0]):
                    positions = source[row].nonzero(as_tuple=False).flatten()
                    if positions.numel() == 0:
                        continue
                    if ratio >= 1.0:
                        chosen = positions
                    else:
                        count = max(1, int(math.ceil(ratio * positions.numel())))
                        if deterministic:
                            # ``floor(i*n/count)`` is unique for count <= n
                            # and gives a stable coverage of the valid span.
                            indices = (
                                torch.arange(count, device=positions.device)
                                * positions.numel()
                                // count
                            )
                            chosen = positions[indices]
                        else:
                            chosen = positions[
                                torch.randperm(
                                    positions.numel(), device=positions.device
                                )[:count]
                            ]
                    limited[row, chosen] = True
                view[:, start:end] = limited
            if found:
                views.append((f"exposure_{ratio:g}", view))
        return views

    def build_drop_mask(
        self,
        attention_mask: torch.Tensor,
        modality_spans: Sequence[Span],
        modality: str,
    ) -> Optional[torch.Tensor]:
        """Drop one modality; return ``None`` if it would leave a row empty."""
        spans = self._validate_spans(modality_spans, attention_mask.shape[1])
        target = self._canonical_name(modality)
        view = attention_mask.bool().clone()
        found = False
        for data_name, start, end in spans:
            if self._canonical_name(data_name) == target:
                view[:, start:end] = False
                found = True
        if not found or not view.any(dim=1).all():
            return None
        return view

    def shuffle_modality_view(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_spans: Sequence[Span],
        modality: str,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace one modality's evidence with another batch item's evidence.

        This is a validation-only negative control for the exposure inner task.
        Formula and every non-selected modality remain attached to the original
        molecule.  Both the selected embeddings and its valid-token mask are
        moved together, so the control cannot accidentally expose a length
        pattern from one molecule with the content from another.
        """
        if inputs_embeds.ndim != 3 or attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("inputs_embeds and attention_mask shapes are incompatible")
        spans = self._validate_spans(modality_spans, inputs_embeds.shape[1])
        target = self._canonical_name(modality)
        output = inputs_embeds.clone()
        shuffled_mask = attention_mask.bool().clone()
        batch_size = inputs_embeds.shape[0]
        if batch_size < 2:
            return output, shuffled_mask
        if deterministic:
            permutation = torch.roll(
                torch.arange(batch_size, device=inputs_embeds.device), shifts=1
            )
        else:
            permutation = torch.randperm(batch_size, device=inputs_embeds.device)
            if torch.equal(permutation, torch.arange(batch_size, device=inputs_embeds.device)):
                permutation = torch.roll(permutation, shifts=1)
        found = False
        for data_name, start, end in spans:
            if self._canonical_name(data_name) != target:
                continue
            found = True
            output[:, start:end] = inputs_embeds[permutation, start:end]
            shuffled_mask[:, start:end] = attention_mask[permutation, start:end].bool()
        if not found:
            raise ValueError(f"modality {modality!r} is not present in modality_spans")
        return output, shuffled_mask

    def apply_shift_augmentation(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_spans: Sequence[Span],
        modality: str,
        *,
        enabled: bool = False,
        scale_jitter: float = 0.0,
        noise_std: float = 0.0,
        token_dropout: float = 0.0,
        baseline_shift: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a conservative modality-local shift to an exposure view.

        The data pipeline exposes token ids rather than physical peak arrays at
        this point.  The augmentation therefore acts in embedding space: a
        small per-example scale/jitter and optional token dropout model the
        source-to-target representation shift without touching Formula or any
        other modality.  The returned mask is updated for dropped tokens.
        """
        if inputs_embeds.ndim != 3 or attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("inputs_embeds and attention_mask shapes are incompatible")
        if not enabled:
            return inputs_embeds, attention_mask.bool()
        if min(float(scale_jitter), float(noise_std), float(token_dropout), float(baseline_shift)) < 0:
            raise ValueError("shift augmentation magnitudes must be non-negative")
        if float(token_dropout) > 1.0:
            raise ValueError("token_dropout must be in [0, 1]")

        spans = self._validate_spans(modality_spans, inputs_embeds.shape[1])
        target = self._canonical_name(modality)
        output = inputs_embeds.clone()
        shifted_mask = attention_mask.bool().clone()
        batch_size = inputs_embeds.shape[0]
        # Validation controls must be repeatable.  Stochastic corruption is a
        # training-only proxy; deterministic validation uses the uncorrupted
        # exposure mask while still exercising the same code path.
        if deterministic:
            token_dropout = 0.0
            scale_jitter = 0.0
            noise_std = 0.0
            baseline_shift = 0.0

        for data_name, start, end in spans:
            if self._canonical_name(data_name) != target:
                continue
            segment = output[:, start:end]
            valid = shifted_mask[:, start:end]
            if not valid.any():
                continue

            # Keep at least one token per row when stochastic dropout is used.
            if token_dropout > 0.0:
                drop = (torch.rand(valid.shape, device=valid.device) < float(token_dropout)) & valid
                for row in range(batch_size):
                    positions = valid[row].nonzero(as_tuple=False).flatten()
                    if positions.numel() and (~drop[row, positions]).sum() == 0:
                        keep = positions[torch.randint(positions.numel(), (1,), device=positions.device)]
                        drop[row, keep] = False
                valid = valid & ~drop
                shifted_mask[:, start:end] = valid
                segment = torch.where(valid.unsqueeze(-1), segment, torch.zeros_like(segment))

            token_scale = 1.0
            if scale_jitter > 0.0:
                token_scale = 1.0 + torch.randn(
                    (batch_size, 1, 1), device=segment.device, dtype=segment.dtype
                ) * float(scale_jitter)
            shifted = segment * token_scale
            if noise_std > 0.0:
                magnitude = segment.detach().float().std(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
                noise = torch.randn_like(segment) * (float(noise_std) * magnitude.to(segment.dtype))
                shifted = shifted + noise
            if baseline_shift > 0.0:
                magnitude = segment.detach().float().std(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
                offset = torch.randn(
                    (batch_size, 1, 1), device=segment.device, dtype=segment.dtype
                ) * (float(baseline_shift) * magnitude.to(segment.dtype))
                shifted = shifted + offset
            output[:, start:end] = torch.where(valid.unsqueeze(-1), shifted, segment)
        return output, shifted_mask

    @staticmethod
    def masked_pool(hidden: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
            raise ValueError("hidden/mask shapes are incompatible")
        weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(eps)

    @staticmethod
    def consistency_loss(representations: Sequence[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
        """Mean cosine distance to a stop-gradient mean representation."""
        if not representations:
            raise ValueError("at least one representation is required")
        reference = torch.stack(
            [F.normalize(value, dim=-1, eps=eps) for value in representations], dim=0
        ).mean(dim=0).detach()
        return torch.stack(
            [1.0 - F.cosine_similarity(value, reference, dim=-1, eps=eps).mean() for value in representations]
        ).mean()

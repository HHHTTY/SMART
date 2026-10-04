"""MS-only residual adaptation and domain-adversarial building blocks.

The module is deliberately independent from the training runner.  It only
knows how to identify m/z and intensity tokens, cap a residual correction,
summarise an MS sequence, and reverse the domain gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def gradient_reverse(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Return ``value`` unchanged while reversing its upstream gradient."""

    return _GradientReversal.apply(value, float(scale))


def build_ms_role_masks(
    token_ids: torch.Tensor,
    *,
    numeric_id_lookup: torch.Tensor,
    header_id_lookup: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split numeric MS tokens into alternating m/z and intensity roles.

    ``token_ids`` is batch-first.  Every energy header resets the alternation,
    and a headerless spectrum starts with an m/z token.  Specials and padding
    neither receive a role nor advance the alternation.
    """

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    if numeric_id_lookup.ndim != 1 or header_id_lookup.ndim != 1:
        raise ValueError("token lookup tensors must be one-dimensional")
    if numeric_id_lookup.device != token_ids.device:
        numeric_id_lookup = numeric_id_lookup.to(token_ids.device)
    if header_id_lookup.device != token_ids.device:
        header_id_lookup = header_id_lookup.to(token_ids.device)

    valid_ids = token_ids.ge(0) & token_ids.lt(numeric_id_lookup.numel())
    safe_ids = token_ids.clamp(min=0, max=max(0, numeric_id_lookup.numel() - 1))
    numeric = valid_ids & numeric_id_lookup[safe_ids]
    headers = valid_ids & header_id_lookup[safe_ids]

    mz = torch.zeros_like(numeric)
    intensity = torch.zeros_like(numeric)
    next_is_mz = torch.ones(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)
    for position in range(token_ids.shape[1]):
        next_is_mz = torch.where(
            headers[:, position], torch.ones_like(next_is_mz), next_is_mz
        )
        current_numeric = numeric[:, position]
        mz[:, position] = current_numeric & next_is_mz
        intensity[:, position] = current_numeric & ~next_is_mz
        next_is_mz = torch.where(current_numeric, ~next_is_mz, next_is_mz)
    return mz, intensity, numeric


@dataclass
class AdapterDiagnostics:
    numeric_tokens: torch.Tensor
    mean_correction_ratio: torch.Tensor
    clipped_fraction: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "numeric_tokens": float(self.numeric_tokens.detach().cpu()),
            "mean_correction_ratio": float(
                self.mean_correction_ratio.detach().cpu()
            ),
            "clipped_fraction": float(self.clipped_fraction.detach().cpu()),
        }


@dataclass
class MSAdapterOutput:
    embeddings: torch.Tensor
    raw_summary: torch.Tensor
    adapted_summary: torch.Tensor
    diagnostics: AdapterDiagnostics


def _masked_mean_std(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("values/mask must have shapes [B,L,D] and [B,L]")
    weights = mask.unsqueeze(-1).to(values.dtype)
    count = weights.sum(dim=1)
    denominator = count.clamp_min(1.0)
    mean = (values * weights).sum(dim=1) / denominator
    variance = ((values - mean.unsqueeze(1)).square() * weights).sum(dim=1)
    variance = variance / denominator
    std = torch.sqrt(variance.clamp_min(0.0) + eps)
    available = count.gt(0)
    mean = torch.where(available, mean, torch.zeros_like(mean))
    std = torch.where(available, std, torch.zeros_like(std))
    return mean, std


def ms_distribution_summary(
    ms_embeddings: torch.Tensor,
    mz_mask: torch.Tensor,
    intensity_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-spectrum [mean_mz, std_mz, mean_int, std_int]."""

    mz_mean, mz_std = _masked_mean_std(ms_embeddings, mz_mask)
    int_mean, int_std = _masked_mean_std(ms_embeddings, intensity_mask)
    return torch.cat((mz_mean, mz_std, int_mean, int_std), dim=-1)


class MSResidualAdapter(nn.Module):
    """Zero-initialised bottleneck adapter with a per-token residual cap."""

    def __init__(
        self,
        d_model: int = 512,
        bottleneck: int = 64,
        max_residual_ratio: float = 0.20,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if d_model <= 0 or bottleneck <= 0:
            raise ValueError("d_model and bottleneck must be positive")
        if not 0.0 < max_residual_ratio <= 1.0:
            raise ValueError("max_residual_ratio must be in (0, 1]")
        self.d_model = int(d_model)
        self.bottleneck = int(bottleneck)
        self.max_residual_ratio = float(max_residual_ratio)
        self.eps = float(eps)
        self.norm = nn.LayerNorm(self.d_model)
        self.down = nn.Linear(self.d_model, self.bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(self.bottleneck, self.d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        *,
        ms_span: Optional[tuple[int, int]],
        mz_mask: Optional[torch.Tensor],
        intensity_mask: Optional[torch.Tensor],
    ) -> MSAdapterOutput:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.d_model:
            raise ValueError(
                f"embeddings must have shape [B,L,{self.d_model}]"
            )
        if ms_span is None:
            zeros = embeddings.new_zeros((embeddings.shape[0], 4 * self.d_model))
            scalar = embeddings.new_zeros(())
            return MSAdapterOutput(
                embeddings=embeddings,
                raw_summary=zeros,
                adapted_summary=zeros,
                diagnostics=AdapterDiagnostics(scalar, scalar, scalar),
            )
        start, end = ms_span
        if not 0 <= start < end <= embeddings.shape[1]:
            raise ValueError("ms_span is outside the embedding sequence")
        if mz_mask is None or intensity_mask is None:
            raise ValueError("MS role masks are required when an MS span is present")
        expected_shape = (embeddings.shape[0], end - start)
        if mz_mask.shape != expected_shape or intensity_mask.shape != expected_shape:
            raise ValueError(
                f"MS role masks must have shape {expected_shape}, got "
                f"{tuple(mz_mask.shape)} and {tuple(intensity_mask.shape)}"
            )

        segment = embeddings[:, start:end]
        numeric_mask = mz_mask | intensity_mask
        raw_summary = ms_distribution_summary(segment, mz_mask, intensity_mask)
        raw_delta = self.up(self.activation(self.down(self.norm(segment))))
        original_norm = segment.norm(dim=-1)
        raw_delta_norm = raw_delta.norm(dim=-1)
        cap = self.max_residual_ratio * original_norm
        scale = torch.minimum(
            torch.ones_like(raw_delta_norm),
            cap / raw_delta_norm.clamp_min(self.eps),
        )
        capped_delta = raw_delta * scale.unsqueeze(-1)
        adapted_segment = torch.where(
            numeric_mask.unsqueeze(-1), segment + capped_delta, segment
        )
        adapted = torch.cat(
            (embeddings[:, :start], adapted_segment, embeddings[:, end:]), dim=1
        )
        adapted_summary = ms_distribution_summary(
            adapted_segment, mz_mask, intensity_mask
        )

        numeric_count = numeric_mask.sum()
        safe_count = numeric_count.clamp_min(1).to(embeddings.dtype)
        correction_ratio = capped_delta.norm(dim=-1) / original_norm.clamp_min(
            self.eps
        )
        clipped = raw_delta_norm > (cap + self.eps)
        diagnostics = AdapterDiagnostics(
            numeric_tokens=numeric_count.to(embeddings.dtype),
            mean_correction_ratio=(
                correction_ratio * numeric_mask.to(correction_ratio.dtype)
            ).sum()
            / safe_count,
            clipped_fraction=(clipped & numeric_mask).sum().to(embeddings.dtype)
            / safe_count,
        )
        return MSAdapterOutput(
            embeddings=adapted,
            raw_summary=raw_summary,
            adapted_summary=adapted_summary,
            diagnostics=diagnostics,
        )


class ModalityResidualAdapter(MSResidualAdapter):
    """Adapt valid tokens in one modality with mean/std domain features."""

    def __init__(
        self,
        modality: str,
        special_token_ids: tuple[int, ...] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.modality = modality
        self.special_token_ids = tuple(int(value) for value in special_token_ids)

    def forward(
        self,
        embeddings: torch.Tensor,
        *,
        modality_span: Optional[tuple[int, int]],
        valid_mask: Optional[torch.Tensor],
    ) -> MSAdapterOutput:
        empty = None if valid_mask is None else torch.zeros_like(valid_mask)
        output = super().forward(
            embeddings,
            ms_span=modality_span,
            mz_mask=valid_mask,
            intensity_mask=empty,
        )
        summary_width = 2 * self.d_model
        return MSAdapterOutput(
            embeddings=output.embeddings,
            raw_summary=output.raw_summary[:, :summary_width],
            adapted_summary=output.adapted_summary[:, :summary_width],
            diagnostics=output.diagnostics,
        )


class MSDomainDiscriminator(nn.Module):
    """Two-layer discriminator over a 4*d_model MS distribution summary."""

    def __init__(
        self, d_model: int = 512, hidden_dim: int = 256, summary_features: int = 4
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(summary_features) * int(d_model), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.network(summary).squeeze(-1)


class ChemistryConditionedMSDomainDiscriminator(nn.Module):
    """Condition the MS domain boundary on observable Formula chemistry.

    Formula features only gate MS-derived hidden features.  There is no
    condition-only logit path, which prevents the discriminator from solving
    the task solely from different source/target Formula frequencies.
    """

    def __init__(
        self,
        d_model: int = 512,
        condition_dim: int = 28,
        hidden_dim: int = 256,
        summary_features: int = 4,
    ) -> None:
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        self.condition_dim = int(condition_dim)
        self.summary_projection = nn.Linear(
            int(summary_features) * int(d_model), int(hidden_dim), bias=False
        )
        self.condition_gate = nn.Sequential(
            nn.Linear(self.condition_dim, int(hidden_dim)),
            nn.Tanh(),
        )
        self.activation = nn.GELU()
        self.classifier = nn.Linear(int(hidden_dim), 1)

    def forward(
        self, summary: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[0] != summary.shape[0]:
            raise ValueError("condition must have shape [batch, condition_dim]")
        if condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"expected condition_dim={self.condition_dim}, got {condition.shape[1]}"
            )
        hidden = self.activation(self.summary_projection(summary))
        gate = 1.0 + self.condition_gate(condition.to(summary.dtype))
        return self.classifier(hidden * gate).squeeze(-1)

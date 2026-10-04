"""Late-layer anchor bridges for frozen multimodal encoders.

The bridge is attached to selected encoder layers with forward hooks.  At
each selected layer, Formula and CNMR hidden states are pooled into a local
structural anchor.  Only valid HNMR, MSMS, and IR tokens receive bounded,
modality-specific residual corrections.  The encoder and decoder remain
frozen and token positions are never compacted.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


Span = tuple[str, int, int]
AnchorState = tuple[torch.Tensor, torch.Tensor]
ANCHOR_MODES = ("none", "internal", "independent_fc")


@dataclass
class LateBridgeRun:
    """Per-forward state populated by the selected encoder-layer hooks."""

    attention_mask: torch.Tensor
    modality_spans: tuple[Span, ...]
    enabled_modalities: frozenset[str]
    enabled_layers: frozenset[int]
    anchor_permutation: Optional[torch.Tensor] = None
    external_anchors: Optional[Mapping[int, AnchorState]] = None
    capture_only: bool = False
    hidden_states: dict[int, torch.Tensor] = field(default_factory=dict)
    correction_ratios: dict[int, torch.Tensor] = field(default_factory=dict)
    anchor_available: dict[int, torch.Tensor] = field(default_factory=dict)


class _LayerModalityBridge(nn.Module):
    """Zero-output, anchor-conditioned low-rank residual adapter."""

    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(d_model)
        self.anchor_norm = nn.LayerNorm(d_model)
        self.token_down = nn.Linear(d_model, rank, bias=False)
        self.anchor_down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=True)
        self.affine_log_scale = nn.Parameter(torch.zeros(d_model))
        self.affine_bias = nn.Parameter(torch.zeros(d_model))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self, tokens: torch.Tensor, anchor: torch.Tensor, *, use_anchor: bool = True
    ) -> torch.Tensor:
        token_feature = self.token_down(self.token_norm(tokens))
        anchor_input = anchor if use_anchor else torch.zeros_like(anchor)
        anchor_feature = self.anchor_down(self.anchor_norm(anchor_input))[:, None, :]
        low_rank = self.up(F.gelu(token_feature + anchor_feature))
        scale = 0.10 * torch.tanh(self.affine_log_scale)
        bias = 0.10 * torch.tanh(self.affine_bias)
        return low_rank + tokens * scale + bias


class LateLayerAnchorBridge(nn.Module):
    """Apply anchor-conditioned bridges after selected encoder layers.

    Layer indices are zero-based.  ``layer_indices=(2, 3, 4, 5)`` therefore
    means encoder layers 3--6 in one-based reporting.
    """

    def __init__(
        self,
        d_model: int = 512,
        rank: int = 16,
        layer_indices: Sequence[int] = (2, 3, 4, 5),
        modalities: Sequence[str] = ("HNMR", "MSMS", "IR"),
        anchor_modalities: Sequence[str] = ("Formula", "CNMR"),
        max_delta_ratio: float = 0.10,
        modality_map: Optional[Mapping[str, str]] = None,
        anchor_mode: str = "internal",
    ) -> None:
        super().__init__()
        indices = tuple(int(value) for value in layer_indices)
        if not indices or any(value < 0 for value in indices):
            raise ValueError("layer_indices must contain non-negative values")
        if len(set(indices)) != len(indices):
            raise ValueError("layer_indices must be unique")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 < max_delta_ratio <= 1.0:
            raise ValueError("max_delta_ratio must lie in (0, 1]")
        if anchor_mode not in ANCHOR_MODES:
            raise ValueError(
                f"anchor_mode must be one of {ANCHOR_MODES}, got {anchor_mode!r}"
            )
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.layer_indices = indices
        self.modalities = tuple(str(value) for value in modalities)
        self.anchor_modalities = tuple(str(value) for value in anchor_modalities)
        self.max_delta_ratio = float(max_delta_ratio)
        self.modality_map = dict(modality_map or {})
        self.anchor_mode = str(anchor_mode)
        self.layer_bridges = nn.ModuleDict(
            {
                str(layer_index): nn.ModuleDict(
                    {
                        name: _LayerModalityBridge(self.d_model, self.rank)
                        for name in self.modalities
                    }
                )
                for layer_index in self.layer_indices
            }
        )
        self._handles: list[Any] = []
        self._run: Optional[LateBridgeRun] = None

    def canonical_name(self, name: str) -> str:
        return str(self.modality_map.get(str(name), str(name)))

    @staticmethod
    def _validate_spans(spans: Sequence[Span], length: int) -> tuple[Span, ...]:
        result = tuple((str(name), int(start), int(end)) for name, start, end in spans)
        cursor = 0
        for _name, start, end in result:
            if start != cursor or end < start:
                raise ValueError("modality spans must be contiguous and ordered")
            cursor = end
        if not result or cursor != length:
            raise ValueError("modality spans do not cover the hidden sequence")
        return result

    def _masks(
        self, attention_mask: torch.Tensor, spans: Sequence[Span]
    ) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        for raw_name, start, end in spans:
            name = self.canonical_name(raw_name)
            current = torch.zeros_like(attention_mask, dtype=torch.bool)
            current[:, start:end] = attention_mask[:, start:end].bool()
            masks[name] = masks.get(name, torch.zeros_like(current)) | current
        return masks

    @staticmethod
    def _masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _anchor(
        self,
        hidden: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
        permutation: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = []
        available = []
        for name in self.anchor_modalities:
            mask = masks.get(name)
            if mask is None:
                continue
            pooled.append(self._masked_pool(hidden, mask))
            available.append(mask.any(dim=1))
        if not pooled:
            return hidden.new_zeros(hidden.shape[0], hidden.shape[-1]), torch.zeros(
                hidden.shape[0], dtype=torch.bool, device=hidden.device
            )
        stacked = torch.stack(pooled, dim=0)
        observed = torch.stack(available, dim=0)
        weights = observed.to(hidden.dtype).unsqueeze(-1)
        anchor = (stacked * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1.0)
        anchor_available = observed.any(dim=0)
        if permutation is not None:
            anchor = anchor[permutation]
            anchor_available = anchor_available[permutation]
        return anchor, anchor_available

    def anchors_from_run(self, run: LateBridgeRun) -> dict[int, AnchorState]:
        """Pool per-layer F/C anchors captured by an independent encoder pass."""

        masks = self._masks(run.attention_mask, run.modality_spans)
        anchors: dict[int, AnchorState] = {}
        for layer_index in self.layer_indices:
            if layer_index not in run.hidden_states:
                raise RuntimeError(f"missing captured hidden state for layer {layer_index}")
            anchor, available = self._anchor(
                run.hidden_states[layer_index], masks, permutation=None
            )
            anchors[layer_index] = (anchor.detach(), available.detach())
        return anchors

    def _resolve_anchor(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
        run: LateBridgeRun,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.anchor_mode == "none":
            return hidden.new_zeros(hidden.shape[0], hidden.shape[-1]), torch.ones(
                hidden.shape[0], dtype=torch.bool, device=hidden.device
            )
        if self.anchor_mode == "internal":
            return self._anchor(hidden, masks, run.anchor_permutation)
        if run.external_anchors is None or layer_index not in run.external_anchors:
            raise RuntimeError(
                "independent_fc mode requires an external F/C anchor for every active layer"
            )
        anchor, available = run.external_anchors[layer_index]
        if anchor.shape != (hidden.shape[0], hidden.shape[-1]):
            raise ValueError("external anchor shape does not match encoder hidden state")
        if available.shape != (hidden.shape[0],):
            raise ValueError("external anchor availability must have shape [batch]")
        anchor = anchor.to(device=hidden.device, dtype=hidden.dtype)
        available = available.to(device=hidden.device, dtype=torch.bool)
        if run.anchor_permutation is not None:
            anchor = anchor[run.anchor_permutation]
            available = available[run.anchor_permutation]
        return anchor, available

    @staticmethod
    def _hidden_and_rebuild(output: Any) -> tuple[torch.Tensor, Any]:
        if isinstance(output, torch.Tensor):
            return output, lambda value: value
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0], lambda value: (value, *output[1:])
        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            return output[0], lambda value: [value, *output[1:]]
        raise TypeError(f"unsupported encoder layer output type: {type(output)!r}")

    def _apply_layer(
        self, layer_index: int, hidden: torch.Tensor, run: LateBridgeRun
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("encoder hidden state must have shape [batch, sequence, d_model]")
        spans = self._validate_spans(run.modality_spans, hidden.shape[1])
        if run.attention_mask.shape != hidden.shape[:2]:
            raise ValueError("attention mask does not match encoder hidden state")
        masks = self._masks(run.attention_mask, spans)
        anchor, anchor_available = self._resolve_anchor(
            layer_index, hidden, masks, run
        )
        output = hidden
        ratios = hidden.new_zeros(hidden.shape[0], len(self.modalities))
        for modality_index, name in enumerate(self.modalities):
            if name not in run.enabled_modalities:
                continue
            mask = masks.get(name)
            if mask is None:
                continue
            observed = mask.any(dim=1) & anchor_available
            effective_mask = mask & observed[:, None]
            raw_delta = self.layer_bridges[str(layer_index)][name](
                hidden, anchor, use_anchor=self.anchor_mode != "none"
            )
            raw_delta = raw_delta * effective_mask.unsqueeze(-1).to(raw_delta.dtype)
            reference = hidden * effective_mask.unsqueeze(-1).to(hidden.dtype)
            delta_norm = (raw_delta.float().pow(2).sum(dim=(1, 2)) + 1e-12).sqrt()
            reference_norm = (
                reference.float().pow(2).sum(dim=(1, 2)) + 1e-12
            ).sqrt().clamp_min(1e-6)
            bound = self.max_delta_ratio * reference_norm
            scale = (bound / delta_norm.clamp_min(1e-6)).clamp(max=1.0)
            correction = raw_delta * scale[:, None, None].to(raw_delta.dtype)
            output = output + correction
            ratios[:, modality_index] = (
                (correction.float().pow(2).sum(dim=(1, 2)) + 1e-12).sqrt()
                / reference_norm
            )
        run.correction_ratios[layer_index] = ratios
        run.anchor_available[layer_index] = anchor_available
        return output

    def _hook(self, layer_index: int):
        def apply(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            run = self._run
            if run is None:
                return output
            hidden, rebuild = self._hidden_and_rebuild(output)
            if run.capture_only or layer_index not in run.enabled_layers:
                run.hidden_states[layer_index] = hidden
                return output
            repaired = self._apply_layer(layer_index, hidden, run)
            run.hidden_states[layer_index] = repaired
            return rebuild(repaired)

        return apply

    def attach(self, encoder: nn.Module) -> None:
        """Attach to an encoder with a ``layers`` sequence."""
        self.detach()
        layers = getattr(encoder, "layers", None)
        if layers is None:
            raise TypeError("encoder must expose a layers sequence")
        depth = len(layers)
        invalid = [index for index in self.layer_indices if index >= depth]
        if invalid:
            raise ValueError(f"layer indices {invalid} exceed encoder depth {depth}")
        for layer_index in self.layer_indices:
            self._handles.append(
                layers[layer_index].register_forward_hook(self._hook(layer_index))
            )

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._run = None

    @contextmanager
    def activate(
        self,
        attention_mask: torch.Tensor,
        modality_spans: Sequence[Span],
        *,
        enabled_modalities: Optional[Sequence[str]] = None,
        enabled_layers: Optional[Sequence[int]] = None,
        anchor_permutation: Optional[torch.Tensor] = None,
        external_anchors: Optional[Mapping[int, AnchorState]] = None,
        capture_only: bool = False,
    ) -> Iterator[LateBridgeRun]:
        """Activate hooks for exactly one encoder forward."""
        if not self._handles:
            raise RuntimeError("bridge must be attached before activation")
        if self._run is not None:
            raise RuntimeError("nested LateLayerAnchorBridge activation is unsupported")
        enabled = frozenset(
            self.modalities if enabled_modalities is None else enabled_modalities
        )
        unknown = enabled.difference(self.modalities)
        if unknown:
            raise ValueError(f"unknown bridge modalities: {sorted(unknown)}")
        selected_layers = frozenset(
            self.layer_indices if enabled_layers is None else (int(value) for value in enabled_layers)
        )
        unknown_layers = selected_layers.difference(self.layer_indices)
        if unknown_layers:
            raise ValueError(f"unknown bridge layers: {sorted(unknown_layers)}")
        run = LateBridgeRun(
            attention_mask=attention_mask,
            modality_spans=tuple(modality_spans),
            enabled_modalities=enabled,
            enabled_layers=selected_layers,
            anchor_permutation=anchor_permutation,
            external_anchors=external_anchors,
            capture_only=bool(capture_only),
        )
        self._run = run
        try:
            yield run
        finally:
            self._run = None

    def set_trainable_modalities(self, modalities: Sequence[str]) -> None:
        selected = set(modalities)
        unknown = selected.difference(self.modalities)
        if unknown:
            raise ValueError(f"unknown bridge modalities: {sorted(unknown)}")
        for layer in self.layer_bridges.values():
            for name, bridge in layer.items():
                for parameter in bridge.parameters():
                    parameter.requires_grad_(name in selected)

    def correction_matrix(self, run: LateBridgeRun) -> torch.Tensor:
        """Return diagnostics as [batch, selected_layer, modality]."""
        zero = torch.zeros(
            run.attention_mask.shape[0],
            len(self.modalities),
            dtype=torch.float32,
            device=run.attention_mask.device,
        )
        values = [run.correction_ratios.get(index, zero) for index in self.layer_indices]
        if not values:
            return run.attention_mask.new_zeros(
                (run.attention_mask.shape[0], 0, len(self.modalities)),
                dtype=torch.float32,
            )
        return torch.stack(values, dim=1)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

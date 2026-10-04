"""Hierarchical spectral repair for multimodal generation.

This module is the implementation of the design in
``TAHCD_INSPIRED_SPECTRAL_NEGATIVE_TRANSFER_PLAN_20260916_CN.md``.  It is
deliberately separate from the older ``TAHCDSharedBlock`` so that experiments
can distinguish the two methods.

The block has three independent pieces:

* a fixed source-domain subspace plus a zero-initialised expression
  calibrator;
* a local reliability network whose output is injected as an additive key
  bias (``log(g)``) in encoder self-attention and decoder cross-attention;
* an independent per-modality relation head and a Formula/CNMR reference.

The fast state is a small per-sample vector and is never written into module
parameters.  The default statistics are conservative identity statistics; a
source-statistics file should be loaded for a real Stage-1 run.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


Span = tuple[str, int, int]


class ContextualMaskPredictor(nn.Module):
    """Predict held-out units from modality-local context and an FC anchor."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.input_projection = nn.Linear(d_model, hidden_dim)
        self.anchor_projection = nn.Linear(d_model, hidden_dim, bias=False)
        self.context = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_projection = nn.Linear(hidden_dim, d_model)

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor: torch.Tensor,
        context_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if values.ndim != 3 or valid_mask.shape != values.shape[:2]:
            raise ValueError("context predictor values/mask shapes are incompatible")
        if anchor.shape != (values.shape[0], values.shape[2]):
            raise ValueError("context predictor anchor must have shape [batch, d_model]")
        if context_gate is not None and context_gate.shape != values.shape[:2]:
            raise ValueError("context_gate must have shape [batch, sequence]")
        valid = valid_mask.to(values.dtype).unsqueeze(-1)
        hidden = self.input_projection(self.input_norm(values))
        if context_gate is not None:
            # Gate after per-token normalisation/projection. Applying it before
            # LayerNorm makes positive scalar gates nearly scale-invariant.
            hidden = hidden * context_gate.to(hidden.dtype).unsqueeze(-1)
        hidden = hidden + self.anchor_projection(anchor).unsqueeze(1)
        hidden = hidden * valid
        hidden = self.context(hidden.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(hidden) * valid


class SpectralRepairBlock(nn.Module):
    """Global calibration, local attention gating and conditional slack."""

    def __init__(
        self,
        d_model: int,
        *,
        modalities: Sequence[str] = ("HNMR", "MSMS", "IR"),
        anchor_modalities: Sequence[str] = ("Formula", "CNMR"),
        modality_map: Optional[Mapping[str, str]] = None,
        unit_structure: Optional[Mapping[str, Mapping[str, Any]]] = None,
        rank: int = 32,
        relation_dim: int = 32,
        fast_dim: int = 8,
        gate_hidden: int = 64,
        min_gate: float = 0.05,
        max_fast_scale: float = 0.05,
        instance_conditioned_gate: bool = False,
        gate_init_logit: float = 8.0,
        max_fast_gate_logit: float = 0.05,
        max_calibration_ratio: float = 0.10,
        slack_weight: float = 1.0,
        clean_weight: float = 0.05,
        trust_weight: float = 0.01,
        enable_calibration: bool = True,
        enable_gate: bool = True,
        enable_relation: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model <= 0 or rank <= 0 or relation_dim <= 0 or fast_dim <= 0:
            raise ValueError("d_model/rank/relation_dim/fast_dim must be positive")
        if not 0.0 < min_gate < 1.0:
            raise ValueError("min_gate must lie in (0, 1)")
        if (
            max_fast_scale < 0.0
            or max_fast_gate_logit < 0.0
            or max_calibration_ratio < 0.0
        ):
            raise ValueError("repair scales must be non-negative")
        self.d_model = int(d_model)
        self.rank = int(min(rank, d_model))
        self.relation_dim = int(relation_dim)
        self.fast_dim = int(fast_dim)
        self.modalities = tuple(str(value) for value in modalities)
        self.anchor_modalities = tuple(str(value) for value in anchor_modalities)
        self.modality_map = {
            str(key): str(value) for key, value in (modality_map or {}).items()
        }
        self.unit_structure = {
            str(name): dict(config) for name, config in (unit_structure or {}).items()
        }
        self.min_gate = float(min_gate)
        self.max_fast_scale = float(max_fast_scale)
        self.instance_conditioned_gate = bool(instance_conditioned_gate)
        self.gate_init_logit = float(gate_init_logit)
        self.max_fast_gate_logit = float(max_fast_gate_logit)
        self.max_calibration_ratio = float(max_calibration_ratio)
        self.slack_weight = float(slack_weight)
        self.clean_weight = float(clean_weight)
        self.trust_weight = float(trust_weight)
        self.enable_calibration = bool(enable_calibration)
        self.enable_gate = bool(enable_gate)
        self.enable_relation = bool(enable_relation)
        self.eps = float(eps)
        self.hard_gate_strategy: Optional[str] = None
        self.hard_keep_ratio: Optional[float] = None
        self.hard_gate_modalities: tuple[str, ...] = ()
        self.hard_gate_seed = 3247
        self._hard_gate_calls = 0

        # Source statistics.  The identity basis is only a safe construction
        # default; source-statistics generation is provided separately.
        self.register_buffer("means", torch.zeros(len(self.modalities), d_model))
        basis = torch.zeros(len(self.modalities), d_model, self.rank)
        basis[:, : self.rank, :] = torch.eye(self.rank).unsqueeze(0)
        self.register_buffer("bases", basis)
        self.register_buffer(
            "relation_means", torch.zeros(len(self.modalities), relation_dim)
        )
        self.register_buffer(
            "relation_precision",
            torch.eye(relation_dim).expand(len(self.modalities), -1, -1).clone(),
        )
        self.register_buffer(
            "relation_tau", torch.full((len(self.modalities),), float(relation_dim))
        )

        # Slow calibration parameters are exactly identity at initialisation.
        self.log_scale = nn.Parameter(torch.zeros(len(self.modalities), self.rank))
        self.shift = nn.Parameter(torch.zeros(len(self.modalities), self.rank))

        self.gates = nn.ModuleDict()
        self.fast_to_scale = nn.ModuleDict()
        self.fast_to_shift = nn.ModuleDict()
        self.fast_to_gate = nn.ModuleDict()
        self.fast_gate_directions = nn.ModuleDict()
        self.relation_heads = nn.ModuleDict()
        self.anchor_heads = nn.ModuleDict()
        self.relation_decoders = nn.ModuleDict()
        self.mask_predictors = nn.ModuleDict()
        for name in self.modalities:
            # Token-local fallback is intentional for MS/HNMR when a robust
            # physical peak parser is unavailable. IR tokens are existing 75
            # point patches and therefore use the same local interface.
            self.gates[name] = nn.Sequential(
                nn.Linear(2 * d_model + 1, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.zeros_(self.gates[name][-1].weight)
            nn.init.constant_(self.gates[name][-1].bias, self.gate_init_logit)
            self.fast_to_scale[name] = nn.Linear(fast_dim, self.rank, bias=False)
            self.fast_to_shift[name] = nn.Linear(fast_dim, self.rank, bias=False)
            if self.instance_conditioned_gate:
                self.fast_gate_directions[name] = nn.Linear(
                    gate_hidden,
                    fast_dim,
                    bias=False,
                )
            else:
                self.fast_to_gate[name] = nn.Linear(fast_dim, 1, bias=False)
            self.relation_heads[name] = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, relation_dim),
            )
            self.anchor_heads[name] = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, relation_dim),
            )
            self.relation_decoders[name] = nn.Sequential(
                nn.LayerNorm(relation_dim),
                nn.Linear(relation_dim, d_model),
            )
            self.mask_predictors[name] = ContextualMaskPredictor(
                d_model,
                gate_hidden,
            )
            # Small fast-state maps let the TTA variables move the slow
            # directions without creating one free variable per token.
            fast_layers = [
                self.fast_to_scale[name],
                self.fast_to_shift[name],
            ]
            if self.instance_conditioned_gate:
                fast_layers.append(self.fast_gate_directions[name])
            else:
                fast_layers.append(self.fast_to_gate[name])
            for layer in fast_layers:
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)

    def canonical_name(self, name: str) -> str:
        name = str(name)
        if name in self.modality_map:
            return self.modality_map[name]
        for canonical, mapped in self.modality_map.items():
            if mapped == name:
                return canonical
        return name

    @staticmethod
    def validate_spans(spans: Sequence[Span], length: int) -> tuple[Span, ...]:
        normalised = tuple((str(name), int(start), int(end)) for name, start, end in spans)
        previous = 0
        for _name, start, end in normalised:
            if start != previous or start < 0 or end <= start or end > length:
                raise ValueError("modality spans must be contiguous and cover the sequence")
            previous = end
        if previous != length:
            raise ValueError("modality spans must cover the complete sequence")
        return normalised

    def initial_fast_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        state: dict[str, torch.Tensor] = {}
        for name in self.modalities:
            value = torch.zeros(batch_size, self.fast_dim, device=device, dtype=dtype)
            state[name] = value.requires_grad_(requires_grad)
        return state

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    @staticmethod
    def _reduce_per_sample(
        values: torch.Tensor,
        available: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        if values.ndim != 1 or available.shape != values.shape:
            raise ValueError("per-sample values/availability shapes are incompatible")
        if reduction == "none":
            return values
        if reduction == "sum":
            return values.sum()
        if reduction == "mean":
            selected = values.masked_select(available.bool())
            return selected.mean() if selected.numel() else values.sum() * 0.0
        raise ValueError("reduction must be one of: none, sum, mean")

    def _relation_project(
        self,
        head: nn.Module,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Project relation features without a trainable scale ambiguity."""
        projected = head(values)
        normalised = F.layer_norm(projected.float(), (self.relation_dim,))
        return normalised.to(projected.dtype)

    def _masks(
        self,
        attention_mask: torch.Tensor,
        spans: Sequence[Span],
    ) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        for data_name, start, end in spans:
            canonical = self.canonical_name(data_name)
            mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            mask[:, start:end] = attention_mask[:, start:end].bool()
            masks[canonical] = mask
        return masks

    def _embedding_anchor(
        self,
        inputs: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = []
        availability = []
        for name in self.anchor_modalities:
            mask = masks.get(name)
            if mask is None:
                continue
            values.append(self._pool(inputs, mask))
            availability.append(mask.any(dim=1))
        if not values:
            return self._pool(inputs, torch.ones_like(inputs[..., 0], dtype=torch.bool)), torch.zeros(
                inputs.shape[0], dtype=torch.bool, device=inputs.device
            )
        stack = torch.stack(values)
        avail = torch.stack(availability).to(inputs.dtype).unsqueeze(-1)
        count = avail.sum(0)
        anchor = (stack * avail).sum(0) / count.clamp_min(1.0)
        return anchor.detach(), count.squeeze(-1).gt(0.0)

    def _resolve_anchor(
        self,
        inputs: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
        anchor: Optional[torch.Tensor],
        anchor_available: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use an independently encoded F+C reference when one is supplied."""
        if anchor is None:
            return self._embedding_anchor(inputs, masks)
        expected = (inputs.shape[0], self.d_model)
        if anchor.shape != expected:
            raise ValueError(
                f"independent anchor has shape {tuple(anchor.shape)}, expected {expected}"
            )
        if anchor_available is None:
            available = torch.ones(
                inputs.shape[0], dtype=torch.bool, device=inputs.device
            )
        else:
            available = anchor_available.to(device=inputs.device, dtype=torch.bool)
            if available.shape != (inputs.shape[0],):
                raise ValueError("anchor_available must have shape [batch]")
        return anchor.to(device=inputs.device, dtype=inputs.dtype).detach(), available

    def _state(self, state: Optional[Mapping[str, torch.Tensor]], name: str, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if state is None or name not in state:
            return torch.zeros(batch, self.fast_dim, device=device, dtype=dtype)
        value = state[name].to(device=device, dtype=dtype)
        if value.shape != (batch, self.fast_dim):
            raise ValueError(
                f"fast state {name} has shape {tuple(value.shape)}, expected {(batch, self.fast_dim)}"
            )
        return value

    def _calibrate(
        self,
        values: torch.Tensor,
        index: int,
        name: str,
        state: torch.Tensor,
    ) -> torch.Tensor:
        mean = self.means[index].to(values)
        basis = self.bases[index].to(values)
        coordinates = (values - mean) @ basis
        slow_scale = torch.tanh(self.log_scale[index])
        fast_scale = self.fast_to_scale[name](state).tanh().unsqueeze(1) * self.max_fast_scale
        fast_shift = self.fast_to_shift[name](state).tanh().unsqueeze(1) * self.max_fast_scale
        correction_coordinates = (
            coordinates * (slow_scale + fast_scale)
            + self.shift[index]
            + fast_shift
        )
        correction = correction_coordinates @ basis.transpose(0, 1)
        ratio = correction.norm(dim=-1, keepdim=True) / values.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        correction = correction * (self.max_calibration_ratio / ratio.clamp_min(self.max_calibration_ratio)).clamp(max=1.0)
        return values + correction

    def _local_gate(
        self,
        values: torch.Tensor,
        name: str,
        state: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.shape[1] == 1:
            local = values
        else:
            local = F.avg_pool1d(
                values.transpose(1, 2), kernel_size=3, stride=1, padding=1
            ).transpose(1, 2)
        position = torch.linspace(
            0.0, 1.0, values.shape[1], device=values.device, dtype=values.dtype
        ).view(1, -1, 1).expand(values.shape[0], -1, -1)
        features = torch.cat((values, local, position), dim=-1)
        hidden = self.gates[name][1](self.gates[name][0](features))
        base_logits = self.gates[name][2](hidden).squeeze(-1)
        if self.instance_conditioned_gate:
            directions = F.normalize(
                self.fast_gate_directions[name](hidden).float(),
                dim=-1,
                eps=self.eps,
            ).to(hidden.dtype)
            raw_delta = torch.einsum("btd,bd->bt", directions, state)
            fast_delta = self.max_fast_gate_logit * torch.tanh(raw_delta)
        else:
            fast_delta = (
                self.fast_to_gate[name](state).tanh() * self.max_fast_scale
            ).expand_as(base_logits)
        logits = base_logits + fast_delta
        if unit_ids is None:
            gate = self.min_gate + (1.0 - self.min_gate) * torch.sigmoid(logits)
            effective_delta = fast_delta
        else:
            if unit_ids.shape != logits.shape:
                raise ValueError("unit_ids must match modality-local gate logits")
            gate = torch.ones_like(logits)
            valid_units = unit_ids.ge(0)
            if valid_units.any():
                units_per_row = int(unit_ids[valid_units].max()) + 1
                row_offsets = (
                    torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
                    * units_per_row
                )
                flat_ids = (unit_ids.clamp_min(0) + row_offsets).reshape(-1)
                flat_valid = valid_units.reshape(-1)
                sums = logits.new_zeros(logits.shape[0] * units_per_row)
                counts = logits.new_zeros(logits.shape[0] * units_per_row)
                sums.scatter_add_(0, flat_ids[flat_valid], logits.reshape(-1)[flat_valid])
                counts.scatter_add_(
                    0,
                    flat_ids[flat_valid],
                    torch.ones_like(logits.reshape(-1)[flat_valid]),
                )
                group_logits = sums / counts.clamp_min(1.0)
                delta_sums = logits.new_zeros(logits.shape[0] * units_per_row)
                delta_sums.scatter_add_(
                    0,
                    flat_ids[flat_valid],
                    fast_delta.reshape(-1)[flat_valid],
                )
                group_delta = delta_sums / counts.clamp_min(1.0)
                group_gates = self.min_gate + (
                    1.0 - self.min_gate
                ) * torch.sigmoid(group_logits)
                gate = torch.where(
                    valid_units,
                    group_gates[flat_ids].reshape_as(logits),
                    gate,
                )
                effective_delta = torch.where(
                    valid_units,
                    group_delta[flat_ids].reshape_as(logits),
                    torch.zeros_like(logits),
                )
            else:
                effective_delta = torch.zeros_like(logits)
        return gate, effective_delta

    def build_unit_ids(
        self,
        token_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """Build physical-record unit IDs from tokenizer structure metadata."""
        if token_ids.shape != valid_mask.shape:
            raise ValueError("token_ids and valid_mask must have matching shapes")
        name = self.canonical_name(modality)
        output = torch.full_like(token_ids, -1, dtype=torch.long)
        config = self.unit_structure.get(name)
        if config is None:
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            return torch.where(valid_mask.bool(), positions.unsqueeze(0), output)
        special = {int(value) for value in config.get("special_ids", ())}
        headers = {int(value) for value in config.get("header_ids", ())}
        prefixes = {int(value) for value in config.get("prefix_ids", ())}
        delimiters = {int(value) for value in config.get("delimiter_ids", ())}
        record_width = int(config.get("record_width", 0))
        delimiter_terminated = bool(config.get("delimiter_terminated", False))
        structural = ~valid_mask.bool()
        for value in special | headers | prefixes:
            structural |= token_ids.eq(value)
        content = ~structural
        if delimiter_terminated:
            is_delimiter = torch.zeros_like(content)
            for value in delimiters:
                is_delimiter |= token_ids.eq(value) & content
            groups = is_delimiter.long().cumsum(dim=1) - is_delimiter.long()
            output = torch.where(content, groups, output)
        elif record_width > 0:
            content_order = content.long().cumsum(dim=1) - 1
            groups = torch.div(content_order, record_width, rounding_mode="floor")
            output = torch.where(content, groups, output)
        else:
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            output = torch.where(content, positions.unsqueeze(0), output)
        return output

    def apply(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: Sequence[Span],
        *,
        fast_state: Optional[Mapping[str, torch.Tensor]] = None,
        anchor: Optional[torch.Tensor] = None,
        anchor_available: Optional[torch.Tensor] = None,
        unit_ids: Optional[torch.Tensor] = None,
        bypass: bool = False,
        return_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply calibration and return encoder/decoder key biases.

        Both returned biases have shape ``[batch, sequence]`` and are additive
        logit biases.  A value of zero is an exact attention identity; padding
        remains controlled by the normal boolean attention mask.
        """
        if inputs_embeds.ndim != 3 or attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("inputs_embeds and attention_mask shapes are incompatible")
        if unit_ids is not None and unit_ids.shape != attention_mask.shape:
            raise ValueError("unit_ids must have shape [batch, sequence]")
        spans = self.validate_spans(spans, inputs_embeds.shape[1])
        masks = self._masks(attention_mask, spans)
        output = inputs_embeds.clone()
        key_gate = torch.ones_like(attention_mask, dtype=inputs_embeds.dtype)
        fast_gate_delta = torch.zeros_like(attention_mask, dtype=inputs_embeds.dtype)
        hard_drop = torch.zeros_like(attention_mask, dtype=torch.bool)
        distances = inputs_embeds.new_zeros(inputs_embeds.shape[0], len(self.modalities))
        gates = inputs_embeds.new_ones(inputs_embeds.shape[0], len(self.modalities))
        alphas = inputs_embeds.new_zeros(inputs_embeds.shape[0], len(self.modalities))
        anchor, anchor_available = self._resolve_anchor(
            inputs_embeds,
            masks,
            anchor,
            anchor_available,
        )
        for index, name in enumerate(self.modalities):
            mask = masks.get(name)
            if mask is None:
                continue
            state = self._state(
                fast_state,
                name,
                inputs_embeds.shape[0],
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            corrected = (
                self._calibrate(output, index, name, state)
                if self.enable_calibration
                else output
            )
            # Only replace the selected span; Formula/CNMR are never calibrated.
            output = torch.where(mask.unsqueeze(-1), corrected, output)
            # Slice-wise gate calculation keeps the local neighbourhood inside
            # the modality rather than allowing Formula/CNMR to influence it.
            positions = mask.any(dim=0).nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            start, end = int(positions.min()), int(positions.max()) + 1
            local_mask = mask[:, start:end]
            if self.enable_gate:
                gate, gate_delta = self._local_gate(
                    output[:, start:end],
                    name,
                    state,
                    unit_ids[:, start:end] if unit_ids is not None else None,
                )
                gate = torch.where(local_mask, gate, torch.ones_like(gate))
            else:
                gate = torch.ones(
                    inputs_embeds.shape[0],
                    end - start,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                gate_delta = torch.zeros_like(gate)
            key_gate[:, start:end] = torch.where(
                local_mask, gate, key_gate[:, start:end]
            )
            fast_gate_delta[:, start:end] = torch.where(
                local_mask,
                gate_delta,
                fast_gate_delta[:, start:end],
            )
            pooled = self._pool(output, mask)
            distance = 1.0 - F.cosine_similarity(
                pooled, anchor, dim=-1, eps=self.eps
            )
            available = mask.any(dim=1)
            distances[:, index] = torch.where(
                available, distance.detach(), distances[:, index]
            )
            gate_mean = (
                (gate * local_mask.to(gate.dtype)).sum(dim=1)
                / local_mask.sum(dim=1).clamp_min(1)
            )
            gates[:, index] = torch.where(
                available, gate_mean.detach(), gates[:, index]
            )
            alphas[:, index] = torch.where(
                available, state.norm(dim=-1).detach(), alphas[:, index]
            )
        hard_keep_fractions = inputs_embeds.new_ones(
            inputs_embeds.shape[0], len(self.modalities)
        )
        if self.hard_gate_strategy is not None:
            keep_ratio = float(self.hard_keep_ratio)
            if not 0.0 < keep_ratio <= 1.0:
                raise ValueError("hard_keep_ratio must lie in (0, 1]")
            generator = torch.Generator(device=inputs_embeds.device)
            generator.manual_seed(self.hard_gate_seed + self._hard_gate_calls)
            self._hard_gate_calls += 1
            for name in self.hard_gate_modalities:
                mask = masks.get(name)
                if mask is None:
                    continue
                keep = torch.zeros_like(mask)
                modality_index = self.modalities.index(name)
                for batch_index in range(mask.shape[0]):
                    row_mask = mask[batch_index]
                    if unit_ids is not None:
                        row_units = torch.unique(unit_ids[batch_index][row_mask])
                        row_units = row_units[row_units.ge(0)]
                    else:
                        row_units = row_mask.nonzero(as_tuple=False).flatten()
                    if row_units.numel() == 0:
                        continue
                    keep_count = max(1, int(round(keep_ratio * row_units.numel())))
                    if self.hard_gate_strategy == "random":
                        permutation = torch.randperm(
                            row_units.numel(),
                            generator=generator,
                            device=row_units.device,
                        )
                        kept_units = row_units[permutation[:keep_count]]
                    elif self.hard_gate_strategy == "learned_topk":
                        if unit_ids is None:
                            unit_scores = key_gate[batch_index, row_units]
                        else:
                            unit_scores = torch.stack(
                                [
                                    key_gate[batch_index][
                                        unit_ids[batch_index].eq(unit) & row_mask
                                    ].mean()
                                    for unit in row_units
                                ]
                            )
                        selected = torch.topk(
                            unit_scores,
                            k=keep_count,
                            largest=True,
                            sorted=False,
                        ).indices
                        kept_units = row_units[selected]
                    else:
                        raise RuntimeError(
                            f"unknown hard-gate strategy: {self.hard_gate_strategy}"
                        )
                    if unit_ids is not None:
                        for unit in kept_units:
                            keep[batch_index] |= unit_ids[batch_index].eq(unit) & row_mask
                        # Header/special tokens have unit -1 and remain visible.
                        keep[batch_index] |= row_mask & unit_ids[batch_index].lt(0)
                    else:
                        keep[batch_index, kept_units] = True
                    hard_keep_fractions[batch_index, modality_index] = (
                        float(keep_count) / float(row_units.numel())
                    )
                hard_drop |= mask & ~keep
        if bypass:
            output = inputs_embeds
            key_gate = torch.ones_like(key_gate)
            fast_gate_delta.zero_()
            hard_drop.zero_()
        key_bias = torch.log(key_gate.clamp_min(self.min_gate))
        key_bias = key_bias.masked_fill(hard_drop, torch.finfo(key_bias.dtype).min)
        diagnostics: dict[str, Any] = {
            "anchor": anchor.detach(),
            "anchor_available": anchor_available.detach(),
            "distances": distances,
            "gates": gates,
            "alphas": alphas,
            "key_gate": key_gate.detach(),
            "fast_gate_delta": fast_gate_delta.detach(),
            "hard_drop": hard_drop.detach(),
            "hard_keep_fractions": hard_keep_fractions.detach(),
            "hard_gate_strategy": self.hard_gate_strategy,
            "calibration_delta_ratio": ((output - inputs_embeds).norm(dim=-1) / inputs_embeds.norm(dim=-1).clamp_min(self.eps)).detach(),
        }
        if not return_diagnostics:
            diagnostics = {}
        return output, key_bias, key_bias, diagnostics

    def configure_random_gate(
        self,
        keep_ratio: Optional[float],
        *,
        modalities: Sequence[str] = ("MSMS",),
        seed: int = 3247,
    ) -> None:
        """Configure deterministic random hard selection.

        This compatibility wrapper does not claim that an arbitrary keep ratio
        matches the learned gate's effective budget.  Use ``configure_hard_gate``
        with the same ratio for random and learned-top-k comparisons.
        """
        self.configure_hard_gate(
            None if keep_ratio is None else "random",
            keep_ratio,
            modalities=modalities,
            seed=seed,
        )

    def configure_hard_gate(
        self,
        strategy: Optional[str],
        keep_ratio: Optional[float],
        *,
        modalities: Sequence[str] = ("MSMS",),
        seed: int = 3247,
    ) -> None:
        """Select an equal number of physical units by learned score or chance."""
        if strategy not in {None, "random", "learned_topk"}:
            raise ValueError("hard-gate strategy must be random or learned_topk")
        if (strategy is None) != (keep_ratio is None):
            raise ValueError("strategy and keep_ratio must either both be set or both be None")
        if keep_ratio is not None and not 0.0 < float(keep_ratio) <= 1.0:
            raise ValueError("random keep ratio must lie in (0, 1]")
        unknown = set(modalities).difference(self.modalities)
        if unknown:
            raise ValueError(f"unknown random-gate modalities: {sorted(unknown)}")
        self.hard_gate_strategy = strategy
        self.hard_keep_ratio = None if keep_ratio is None else float(keep_ratio)
        self.hard_gate_modalities = tuple(str(name) for name in modalities)
        self.hard_gate_seed = int(seed)
        self._hard_gate_calls = 0

    def relation_slack_loss(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: Sequence[Span],
        *,
        fast_state: Optional[Mapping[str, torch.Tensor]] = None,
        anchor: Optional[torch.Tensor] = None,
        anchor_available: Optional[torch.Tensor] = None,
        active_modalities: Optional[Sequence[str]] = None,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute conditional relation slack on independent modality spans."""
        spans = self.validate_spans(spans, inputs_embeds.shape[1])
        masks = self._masks(attention_mask, spans)
        anchor, anchor_available = self._resolve_anchor(
            inputs_embeds,
            masks,
            anchor,
            anchor_available,
        )
        active = None if active_modalities is None else {self.canonical_name(name) for name in active_modalities}
        losses = []
        valid_counts = []
        distances = []
        for index, name in enumerate(self.modalities):
            if active is not None and name not in active:
                continue
            mask = masks.get(name)
            if mask is None:
                continue
            state = self._state(
                fast_state,
                name,
                inputs_embeds.shape[0],
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            calibrated = (
                self._calibrate(inputs_embeds, index, name, state)
                if self.enable_calibration
                else inputs_embeds
            )
            q = self._relation_project(
                self.relation_heads[name], self._pool(calibrated, mask)
            )
            expected = self._relation_project(self.anchor_heads[name], anchor)
            residual = q - expected.detach()
            # Keep Mahalanobis arithmetic in FP32 even under bf16 Stage 1.
            precision = self.relation_precision[index].to(device=residual.device).float()
            mean = self.relation_means[index].to(device=residual.device).float()
            delta = residual.float() - mean
            distance = torch.einsum("bi,ij,bj->b", delta, precision, delta).clamp_min(0.0)
            tau = self.relation_tau[index].to(device=distance.device, dtype=distance.dtype)
            available = mask.any(dim=1) & anchor_available
            losses.append(
                F.relu(distance - tau)
                * available.to(distance.dtype)
                if self.enable_relation
                else distance * 0.0
            )
            valid_counts.append(available.to(distance.dtype))
            distances.append(distance.detach())
        if not losses:
            zero = inputs_embeds.sum() * 0.0
            per_sample = inputs_embeds.new_zeros(inputs_embeds.shape[0]) + zero
            return (per_sample if reduction == "none" else zero), {
                "relation_distances": inputs_embeds.new_zeros(inputs_embeds.shape[0], 0),
                "relation_loss_per_sample": per_sample.detach(),
            }
        stacked_losses = torch.stack(losses, dim=1)
        stacked_valid = torch.stack(valid_counts, dim=1)
        per_sample = stacked_losses.sum(dim=1) / stacked_valid.sum(dim=1).clamp_min(1.0)
        available = stacked_valid.sum(dim=1).gt(0.0)
        loss = self._reduce_per_sample(per_sample, available, reduction)
        return loss, {
            "relation_distances": torch.stack(distances, dim=1),
            "relation_loss_per_sample": per_sample.detach(),
        }

    def relation_residuals(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: Sequence[Span],
        *,
        fast_state: Optional[Mapping[str, torch.Tensor]] = None,
        anchor: Optional[torch.Tensor] = None,
        anchor_available: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-modality conditional residuals and availability masks."""
        spans = self.validate_spans(spans, inputs_embeds.shape[1])
        masks = self._masks(attention_mask, spans)
        anchor, anchor_available = self._resolve_anchor(
            inputs_embeds, masks, anchor, anchor_available
        )
        residuals = []
        availability = []
        for index, name in enumerate(self.modalities):
            mask = masks.get(name)
            if mask is None:
                residuals.append(
                    inputs_embeds.new_zeros(inputs_embeds.shape[0], self.relation_dim)
                )
                availability.append(
                    torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)
                )
                continue
            state = self._state(
                fast_state,
                name,
                inputs_embeds.shape[0],
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            calibrated = (
                self._calibrate(inputs_embeds, index, name, state)
                if self.enable_calibration
                else inputs_embeds
            )
            q = self._relation_project(
                self.relation_heads[name], self._pool(calibrated, mask)
            )
            expected = self._relation_project(self.anchor_heads[name], anchor)
            residuals.append(q - expected)
            availability.append(mask.any(dim=1) & anchor_available)
        return torch.stack(residuals, dim=1), torch.stack(availability, dim=1)

    def relation_pretraining_loss(
        self,
        corrupt_embeds: torch.Tensor,
        clean_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: Sequence[Span],
        *,
        clean_attention_mask: Optional[torch.Tensor] = None,
        clean_spans: Optional[Sequence[Span]] = None,
        anchor: Optional[torch.Tensor] = None,
        anchor_available: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Pretrain relation heads without letting their representation collapse."""
        if clean_attention_mask is None:
            clean_attention_mask = attention_mask
        spans = self.validate_spans(spans, corrupt_embeds.shape[1])
        if clean_spans is None:
            if clean_embeds.shape[1] != corrupt_embeds.shape[1]:
                raise ValueError(
                    "clean_spans are required when clean/corrupt lengths differ"
                )
            clean_spans = spans
        clean_spans = self.validate_spans(clean_spans, clean_embeds.shape[1])
        masks = self._masks(attention_mask, spans)
        clean_masks = self._masks(clean_attention_mask, clean_spans)
        anchor, anchor_available = self._resolve_anchor(
            clean_embeds, clean_masks, anchor, anchor_available
        )
        losses = []
        reconstruction_losses = []
        consistency_losses = []
        prediction_losses = []
        for index, name in enumerate(self.modalities):
            mask = masks.get(name)
            clean_mask = clean_masks.get(name)
            if mask is None or clean_mask is None:
                continue
            available = mask.any(dim=1) & clean_mask.any(dim=1) & anchor_available
            if not available.any():
                continue
            zero_state = self._state(
                None,
                name,
                corrupt_embeds.shape[0],
                corrupt_embeds.device,
                corrupt_embeds.dtype,
            )
            calibrated = (
                self._calibrate(corrupt_embeds, index, name, zero_state)
                if self.enable_calibration
                else corrupt_embeds
            )
            clean_pooled = self._pool(clean_embeds, clean_mask)
            corrupt_pooled = self._pool(calibrated, mask)
            q_clean = self._relation_project(self.relation_heads[name], clean_pooled)
            q_corrupt = self._relation_project(
                self.relation_heads[name], corrupt_pooled
            )
            expected = self._relation_project(self.anchor_heads[name], anchor)
            reconstructed = self.relation_decoders[name](q_clean)
            reconstruction = F.smooth_l1_loss(
                reconstructed[available], clean_pooled.detach()[available]
            )
            consistency = F.smooth_l1_loss(
                q_corrupt[available], q_clean.detach()[available]
            )
            prediction = F.smooth_l1_loss(
                expected[available], q_clean.detach()[available]
            )
            losses.append(reconstruction + consistency + 0.25 * prediction)
            reconstruction_losses.append(reconstruction.detach())
            consistency_losses.append(consistency.detach())
            prediction_losses.append(prediction.detach())
        if not losses:
            zero = corrupt_embeds.sum() * 0.0
            return zero, {
                "relation_reconstruction": zero.detach(),
                "relation_consistency": zero.detach(),
                "anchor_prediction": zero.detach(),
            }
        return torch.stack(losses).mean(), {
            "relation_reconstruction": torch.stack(reconstruction_losses).mean(),
            "relation_consistency": torch.stack(consistency_losses).mean(),
            "anchor_prediction": torch.stack(prediction_losses).mean(),
        }

    def clean_fidelity_loss(
        self,
        repaired: torch.Tensor,
        clean: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.to(repaired.dtype).unsqueeze(-1)
        return ((repaired - clean).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)

    def trust_loss(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        values = [value.pow(2).mean(dim=-1) for value in state.values()]
        if not values:
            return torch.tensor(0.0)
        per_sample = torch.stack(values, dim=1).mean(dim=1)
        available = torch.ones_like(per_sample, dtype=torch.bool)
        return self._reduce_per_sample(per_sample, available, reduction)

    def masked_prediction_loss(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: Sequence[Span],
        *,
        fast_state: Mapping[str, torch.Tensor],
        anchor: Optional[torch.Tensor] = None,
        anchor_available: Optional[torch.Tensor] = None,
        target_embeds: Optional[torch.Tensor] = None,
        unit_ids: Optional[torch.Tensor] = None,
        active_modalities: Optional[Sequence[str]] = None,
        mask_stride: int = 4,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Self-supervise local units by predicting held-out observations.

        The target is the observed input embedding itself.  The selected
        positions are zeroed *after* calibration and before the frozen local
        prediction head, so the predictor cannot trivially copy the target.
        This is the target-domain analogue of the source clean-unit auxiliary
        task; it is intentionally not a SMILES or pseudo-label objective.
        """
        masks = self._masks(
            attention_mask,
            self.validate_spans(spans, inputs_embeds.shape[1]),
        )
        resolved_anchor, resolved_anchor_available = self._resolve_anchor(
            inputs_embeds,
            masks,
            anchor,
            anchor_available,
        )
        repaired, enc_bias, _dec_bias, _diagnostics = self.apply(
            inputs_embeds,
            attention_mask,
            spans,
            fast_state=fast_state,
            anchor=resolved_anchor,
            anchor_available=resolved_anchor_available,
            unit_ids=unit_ids,
            return_diagnostics=False,
        )
        target = (inputs_embeds if target_embeds is None else target_embeds).detach()
        if target.shape != inputs_embeds.shape:
            raise ValueError("target_embeds must match inputs_embeds")
        if unit_ids is not None and unit_ids.shape != attention_mask.shape:
            raise ValueError("unit_ids must have shape [batch, sequence]")
        # The reliability gate must participate in the inner objective so the
        # episodic fast state can actually adapt both calibration and local
        # exposure.  It weights only the prediction context; selected targets
        # and their loss weights remain fixed, preventing the trivial solution
        # of hiding difficult targets to reduce the objective.
        context_reliability = enc_bias.float().clamp(min=-30.0, max=0.0).exp()
        sample_loss_sum = inputs_embeds.new_zeros(inputs_embeds.shape[0]).float()
        sample_loss_count = inputs_embeds.new_zeros(inputs_embeds.shape[0]).float()
        active = None if active_modalities is None else {self.canonical_name(name) for name in active_modalities}
        for index, (data_name, start, end) in enumerate(self.validate_spans(spans, inputs_embeds.shape[1])):
            name = self.canonical_name(data_name)
            if name not in self.modalities:
                continue
            if active is not None and name not in active:
                continue
            valid = attention_mask[:, start:end].bool()
            stride = max(1, int(mask_stride))
            if unit_ids is None:
                positions = torch.arange(end - start, device=inputs_embeds.device)
                selected = valid & (positions[None, :] % stride == 0)
            else:
                local_units = unit_ids[:, start:end]
                selected = (
                    valid
                    & local_units.ge(0)
                    & local_units.remainder(stride).eq(0)
                )
            if not selected.any():
                continue
            segment = repaired[:, start:end].clone()
            segment = segment.masked_fill(selected.unsqueeze(-1), 0.0)
            segment_gate = context_reliability[:, start:end].masked_fill(
                selected,
                0.0,
            )
            prediction = self.mask_predictors[name](
                segment,
                valid,
                resolved_anchor,
                context_gate=segment_gate,
            )
            expected = F.layer_norm(target[:, start:end].float(), (self.d_model,))
            error = F.smooth_l1_loss(
                prediction.float(), expected, reduction="none"
            ).mean(dim=-1)
            sample_loss_sum = sample_loss_sum + (
                error * selected.to(error.dtype)
            ).sum(dim=1)
            sample_loss_count = sample_loss_count + selected.sum(dim=1)
        available = sample_loss_count.gt(0.0)
        if not available.any():
            zero = inputs_embeds.sum() * 0.0
            per_sample = inputs_embeds.new_zeros(inputs_embeds.shape[0]).float() + zero
        else:
            per_sample = sample_loss_sum / sample_loss_count.clamp_min(1.0)
        return self._reduce_per_sample(per_sample, available, reduction)

    def load_source_statistics(self, path: str) -> None:
        """Load fixed source statistics from a torch/pickle-compatible file."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError("spectral source statistics must be a mapping")
        for key, target in (
            ("means", self.means),
            ("bases", self.bases),
            ("relation_means", self.relation_means),
            ("relation_precision", self.relation_precision),
            ("relation_tau", self.relation_tau),
        ):
            if key in payload:
                value = torch.as_tensor(payload[key], dtype=target.dtype)
                if value.shape != target.shape:
                    raise ValueError(f"{key} has shape {tuple(value.shape)}, expected {tuple(target.shape)}")
                target.copy_(value)

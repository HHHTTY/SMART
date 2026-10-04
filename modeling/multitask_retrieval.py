"""Task-specialized multimodal retrieval heads for test-time tuning.

This module is deliberately independent from the sequence decoder.  It consumes
one pooled encoder representation per available spectroscopy modality and
produces:

* a molecular fingerprint prediction for every modality;
* one modality-specific auxiliary prediction per modality;
* a reliability-gated fingerprint prediction fused across modalities.

The fused fingerprint is used for broad candidate recall. Graph-derived MS
fragmentation, 1D H/C NMR atom-environment, and IR functional-group predictions
rerank the candidates before supervised TTT updates.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_FORMULA_CHARGE_SUFFIX = re.compile(r"[+-]\d*$")


def encode_formula_compositions(
    formulas: Sequence[object],
    element_order: Optional[Sequence[str]] = None,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Encode input molecular formulae as element-count vectors.

    Terminal ionic charges are ignored because they do not change elemental
    composition. Unsupported formula syntax is represented by an all-zero row
    so it cannot impose a hard retrieval filter.
    """

    compositions: list[Dict[str, float]] = []
    observed_elements: set[str] = set()
    for formula in formulas:
        value = "" if formula is None else str(formula).strip()
        neutral_formula = _FORMULA_CHARGE_SUFFIX.sub("", value)
        matches = list(_FORMULA_TOKEN.finditer(neutral_formula))
        if not neutral_formula or "".join(match.group(0) for match in matches) != neutral_formula:
            composition: Dict[str, float] = {}
        else:
            composition = {}
            for match in matches:
                element, count = match.groups()
                composition[element] = composition.get(element, 0.0) + float(count or 1)
            observed_elements.update(composition)
        compositions.append(composition)

    elements = (
        tuple(str(element) for element in element_order)
        if element_order is not None
        else tuple(sorted(observed_elements))
    )
    element_indices = {element: index for index, element in enumerate(elements)}
    encoded = torch.zeros((len(compositions), len(elements)), dtype=torch.float32)
    for row, composition in enumerate(compositions):
        for element, count in composition.items():
            column = element_indices.get(element)
            if column is not None:
                encoded[row, column] = count
    return encoded, elements


class MLPHead(nn.Module):
    """Two-layer prediction head shared by fingerprint and auxiliary tasks."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class NMR1DFusion(nn.Module):
    """Fuse unassigned 1D 1H and 13C NMR representations.

    No atom-to-peak assignment or bond connectivity is assumed.  The gate lets
    the model down-weight one nucleus when it is missing or unreliable.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.h_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.c_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_nmr: Optional[torch.Tensor],
        c_nmr: Optional[torch.Tensor],
        h_available: Optional[torch.Tensor] = None,
        c_available: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_nmr is None and c_nmr is None:
            raise ValueError("At least one of 1H or 13C NMR must be available.")

        reference = h_nmr if h_nmr is not None else c_nmr
        if reference is None:
            raise RuntimeError("NMR reference tensor is unexpectedly missing.")
        batch_size = reference.shape[0]
        device = reference.device
        dtype = reference.dtype

        h_mask = (
            torch.ones(batch_size, dtype=torch.bool, device=device)
            if h_available is None and h_nmr is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=device)
            if h_nmr is None
            else h_available.to(device=device, dtype=torch.bool).reshape(batch_size)
        )
        c_mask = (
            torch.ones(batch_size, dtype=torch.bool, device=device)
            if c_available is None and c_nmr is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=device)
            if c_nmr is None
            else c_available.to(device=device, dtype=torch.bool).reshape(batch_size)
        )
        h_hidden = (
            self.h_projection(h_nmr)
            if h_nmr is not None
            else torch.zeros(batch_size, self.h_projection[1].out_features, device=device, dtype=dtype)
        )
        c_hidden = (
            self.c_projection(c_nmr)
            if c_nmr is not None
            else torch.zeros(batch_size, self.c_projection[1].out_features, device=device, dtype=dtype)
        )
        learned_h_weight = torch.sigmoid(
            self.gate(torch.cat((h_hidden, c_hidden), dim=-1))
        )
        both_available = h_mask & c_mask
        h_weight = torch.where(
            both_available.unsqueeze(-1),
            learned_h_weight,
            h_mask.to(dtype=dtype).unsqueeze(-1),
        )
        fused = h_weight * h_hidden + (1.0 - h_weight) * c_hidden
        nmr_available = h_mask | c_mask
        fused = fused * nmr_available.to(dtype=dtype).unsqueeze(-1)
        return fused, h_weight, nmr_available


class ReliabilityAwareFusion(nn.Module):
    """Fuse modality representations with availability, confidence and attention.

    The optional self-attention block contextualizes each modality token before
    reliability-weighted pooling. Missing-modality masking remains explicit.
    """

    def __init__(
        self,
        modalities: Sequence[str],
        input_dim: int,
        hidden_dim: int,
        use_attention: bool = False,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
        use_formula_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.use_attention = bool(use_attention)
        self.use_formula_conditioning = bool(use_formula_conditioning)
        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                )
                for modality in self.modalities
            }
        )
        self.gates = nn.ModuleDict({modality: nn.Linear(hidden_dim, 1) for modality in self.modalities})
        if self.use_attention:
            if hidden_dim % attention_heads != 0:
                raise ValueError(
                    "hidden_dim must be divisible by attention_heads for modality attention."
                )
            self.modality_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
            self.attention_ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(attention_dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.attention_ffn_norm = nn.LayerNorm(hidden_dim)
            self.attention_pool = nn.Linear(hidden_dim, 1)
        if self.use_formula_conditioning:
            if hidden_dim % attention_heads != 0:
                raise ValueError(
                    "hidden_dim must be divisible by attention_heads for formula conditioning."
                )
            self.formula_projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
            )
            self.formula_cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
            self.formula_attention_norm = nn.LayerNorm(hidden_dim)
            self.formula_attention_ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(attention_dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.formula_attention_ffn_norm = nn.LayerNorm(hidden_dim)
            self.formula_fusion = nn.Sequential(
                nn.LayerNorm(2 * hidden_dim),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(
        self,
        representations: Mapping[str, torch.Tensor],
        reliability: Optional[Mapping[str, torch.Tensor]] = None,
        availability: Optional[Mapping[str, torch.Tensor]] = None,
        formula_representation: Optional[torch.Tensor] = None,
        formula_availability: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = [name for name in self.modalities if name in representations]
        if not available:
            raise ValueError("No configured modality representation is available for fusion.")

        reference = representations[available[0]]
        batch_size = reference.shape[0]
        device = reference.device
        dtype = reference.dtype

        projected = []
        gate_logits = []
        availability_masks = []
        for name in self.modalities:
            if name in representations:
                hidden = self.projections[name](representations[name])
                if hidden.shape[0] != batch_size:
                    raise ValueError("All modality representations must have the same batch size.")
                sample_available = (
                    availability[name].to(device=device, dtype=torch.bool).reshape(batch_size)
                    if availability is not None and name in availability
                    else torch.ones(batch_size, dtype=torch.bool, device=device)
                )
                finite_hidden = torch.isfinite(hidden).all(dim=1)
                sample_available = sample_available & finite_hidden
                hidden = torch.nan_to_num(
                    hidden,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                logit = self.gates[name](hidden).squeeze(-1)
                if reliability is not None and name in reliability:
                    confidence = torch.nan_to_num(
                        reliability[name].to(device=device, dtype=dtype).reshape(batch_size),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                    logit = logit + torch.log(confidence.clamp_min(1e-6))
                projected.append(hidden)
                gate_logits.append(logit)
                availability_masks.append(sample_available)
            else:
                projected.append(torch.zeros(batch_size, self.projections[name][1].out_features, device=device, dtype=dtype))
                gate_logits.append(torch.full((batch_size,), -torch.inf, device=device, dtype=dtype))
                availability_masks.append(torch.zeros(batch_size, dtype=torch.bool, device=device))

        stacked_hidden = torch.stack(projected, dim=1)
        stacked_logits = torch.stack(gate_logits, dim=1)
        available_mask = torch.stack(availability_masks, dim=1)
        has_available_modality = available_mask.any(dim=1)
        if self.use_attention:
            # Avoid all-keys-masked rows producing NaN. Such rows remain
            # unavailable after the final zero-weight normalization.
            key_padding_mask = (~available_mask) & has_available_modality.unsqueeze(-1)
            attended, _ = self.modality_attention(
                stacked_hidden,
                stacked_hidden,
                stacked_hidden,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            stacked_hidden = self.attention_norm(stacked_hidden + attended)
            stacked_hidden = self.attention_ffn_norm(
                stacked_hidden + self.attention_ffn(stacked_hidden)
            )
            # Contextual attention pooling augments the measured reliability
            # gate instead of replacing availability semantics.
            stacked_logits = stacked_logits + self.attention_pool(stacked_hidden).squeeze(-1)
        masked_logits = stacked_logits.masked_fill(~available_mask, -torch.inf)
        masked_logits = torch.where(
            has_available_modality.unsqueeze(-1),
            masked_logits,
            torch.zeros_like(masked_logits),
        )
        weights = torch.softmax(masked_logits, dim=1)
        weights = weights * available_mask.to(dtype=dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = (weights.unsqueeze(-1) * stacked_hidden).sum(dim=1)
        if self.use_formula_conditioning and formula_representation is not None:
            formula_hidden = self.formula_projection(formula_representation)
            formula_available = (
                formula_availability.to(device=device, dtype=torch.bool).reshape(batch_size)
                if formula_availability is not None
                else torch.ones(batch_size, dtype=torch.bool, device=device)
            )
            formula_available = formula_available & torch.isfinite(formula_hidden).all(dim=1)
            formula_hidden = torch.nan_to_num(
                formula_hidden,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            context_query = formula_hidden.unsqueeze(1)
            context_key_padding_mask = (~available_mask) & has_available_modality.unsqueeze(-1)
            context_attended, _ = self.formula_cross_attention(
                context_query,
                stacked_hidden,
                stacked_hidden,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )
            formula_hidden = self.formula_attention_norm(
                context_query + context_attended
            ).squeeze(1)
            formula_hidden = self.formula_attention_ffn_norm(
                formula_hidden + self.formula_attention_ffn(formula_hidden)
            )
            conditioned = self.formula_fusion(torch.cat((fused, formula_hidden), dim=-1))
            fused = torch.where(formula_available.unsqueeze(-1), conditioned, fused)
        return fused, weights


class TaskSpecializedRetrievalHeads(nn.Module):
    """Common fingerprint heads plus one chemistry-aware task per modality."""

    def __init__(
        self,
        encoder_dim: int,
        hidden_dim: int,
        fingerprint_dim: int,
        task_dimensions: Mapping[str, int],
        use_modality_attention: bool = False,
        modality_attention_heads: int = 4,
        modality_attention_dropout: float = 0.0,
        use_formula_conditioning: bool = False,
    ) -> None:
        super().__init__()
        expected = {"MSMS", "NMR", "IR"}
        unknown = set(task_dimensions) - expected
        if unknown:
            raise ValueError(f"Unsupported task modalities: {sorted(unknown)}")

        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.fingerprint_dim = fingerprint_dim
        self.task_dimensions = dict(task_dimensions)
        self.modalities = tuple(name for name in ("MSMS", "NMR", "IR") if name in task_dimensions)

        self.nmr_fusion = NMR1DFusion(encoder_dim, encoder_dim)
        self.fingerprint_heads = nn.ModuleDict(
            {name: MLPHead(encoder_dim, hidden_dim, fingerprint_dim) for name in self.modalities}
        )
        self.task_heads = nn.ModuleDict(
            {
                name: MLPHead(encoder_dim, hidden_dim, task_dimensions[name])
                for name in self.modalities
            }
        )
        self.fusion = ReliabilityAwareFusion(
            self.modalities,
            encoder_dim,
            hidden_dim,
            use_attention=use_modality_attention,
            attention_heads=modality_attention_heads,
            attention_dropout=modality_attention_dropout,
            use_formula_conditioning=use_formula_conditioning,
        )
        self.fused_fingerprint_head = MLPHead(hidden_dim, hidden_dim, fingerprint_dim)

    def _canonicalize(
        self,
        representations: Mapping[str, torch.Tensor],
        availability: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        canonical = {
            name: value
            for name, value in representations.items()
            if name in {"MSMS", "NMR", "IR"}
        }
        canonical_availability = {
            name: value
            for name, value in (availability or {}).items()
            if name in {"MSMS", "NMR", "IR"}
        }
        nmr_gate = None
        if "NMR" not in canonical and ("HNMR" in representations or "CNMR" in representations):
            canonical["NMR"], nmr_gate, canonical_availability["NMR"] = self.nmr_fusion(
                representations.get("HNMR"),
                representations.get("CNMR"),
                (availability or {}).get("HNMR"),
                (availability or {}).get("CNMR"),
            )
        for name, hidden in canonical.items():
            canonical_availability.setdefault(
                name,
                torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device),
            )
        return canonical, canonical_availability, nmr_gate

    def forward(
        self,
        representations: Mapping[str, torch.Tensor],
        reliability: Optional[Mapping[str, torch.Tensor]] = None,
        availability: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        canonical, canonical_availability, nmr_gate = self._canonicalize(
            representations, availability
        )
        active = {name: canonical[name] for name in self.modalities if name in canonical}
        if not active:
            raise ValueError("No MSMS, NMR, or IR representation was supplied.")

        modality_fp_logits = {
            name: self.fingerprint_heads[name](hidden) for name, hidden in active.items()
        }
        task_logits = {name: self.task_heads[name](hidden) for name, hidden in active.items()}
        active_availability = {
            name: canonical_availability[name] for name in active
        }
        fused_hidden, modality_weights = self.fusion(
            active,
            reliability,
            active_availability,
            formula_representation=representations.get("Formula"),
            formula_availability=(availability or {}).get("Formula"),
        )
        fused_fp_logits = self.fused_fingerprint_head(fused_hidden)

        return {
            "fused_fingerprint_logits": fused_fp_logits,
            "modality_fingerprint_logits": modality_fp_logits,
            "task_logits": task_logits,
            "modality_weights": modality_weights,
            "modality_order": self.modalities,
            "modality_availability": active_availability,
            "nmr_h_weight": nmr_gate,
        }

    def loss(
        self,
        predictions: Mapping[str, object],
        fingerprint_target: torch.Tensor,
        task_targets: Mapping[str, torch.Tensor],
        fused_fingerprint_weight: float = 1.0,
        modality_fingerprint_weight: float = 0.5,
        task_weight: float = 1.0,
        alignment_weight: float = 0.1,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fused_logits = predictions["fused_fingerprint_logits"]
        modality_logits = predictions["modality_fingerprint_logits"]
        task_logits = predictions["task_logits"]
        if not isinstance(fused_logits, torch.Tensor):
            raise TypeError("fused_fingerprint_logits must be a tensor.")
        if not isinstance(modality_logits, dict) or not isinstance(task_logits, dict):
            raise TypeError("Modality predictions must be dictionaries.")
        modality_availability = predictions.get("modality_availability", {})
        if not isinstance(modality_availability, dict):
            raise TypeError("modality_availability must be a dictionary.")

        fp_target = fingerprint_target.to(device=fused_logits.device, dtype=fused_logits.dtype)
        components: Dict[str, torch.Tensor] = {}
        active_masks = [
            value.to(device=fused_logits.device, dtype=torch.bool)
            for value in modality_availability.values()
        ]
        fused_mask = torch.stack(active_masks).any(dim=0) if active_masks else None
        components["fingerprint_fused"] = self._masked_bce(
            fused_logits, fp_target, fused_mask
        )

        per_modality_fp = []
        alignment_losses = []
        fused_probability = torch.sigmoid(fused_logits).detach()
        for name, logits in modality_logits.items():
            mask = modality_availability.get(name)
            if mask is not None and not mask.any():
                continue
            per_modality_fp.append(self._masked_bce(logits, fp_target, mask))
            alignment_losses.append(self._masked_bce(logits, fused_probability, mask))
        zero = fused_logits.sum() * 0.0
        components["fingerprint_modalities"] = (
            torch.stack(per_modality_fp).mean() if per_modality_fp else zero
        )
        components["fingerprint_alignment"] = (
            torch.stack(alignment_losses).mean() if alignment_losses else zero
        )

        per_task = []
        for name, logits in task_logits.items():
            if name not in task_targets:
                continue
            target = task_targets[name].to(device=logits.device, dtype=logits.dtype)
            mask = modality_availability.get(name)
            if mask is not None and not mask.any():
                continue
            value = self._masked_bce(logits, target, mask)
            components[f"task_{name.lower()}"] = value
            per_task.append(value)
        if not per_task:
            raise ValueError("At least one active modality-specific task target is required.")
        components["tasks"] = torch.stack(per_task).mean()

        total = (
            fused_fingerprint_weight * components["fingerprint_fused"]
            + modality_fingerprint_weight * components["fingerprint_modalities"]
            + task_weight * components["tasks"]
            + alignment_weight * components["fingerprint_alignment"]
        )
        components["total"] = total
        return total, components

    @staticmethod
    def _masked_bce(
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        per_sample = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        ).mean(dim=-1)
        if mask is None:
            return per_sample.mean()
        valid = mask.to(device=logits.device, dtype=torch.bool).reshape(logits.shape[0])
        if not valid.any():
            return logits.sum() * 0.0
        return per_sample[valid].mean()


@dataclass
class RetrievalResult:
    """Ranked candidate indices, scores, and score components."""

    indices: torch.Tensor
    scores: torch.Tensor
    valid_mask: torch.Tensor
    components: Dict[str, torch.Tensor]


class MultimodalCandidateRetriever:
    """Reliability-aware multimodal ranking over the full source library."""

    def __init__(
        self,
        fingerprint_weight: float = 1.0,
        task_weights: Optional[Mapping[str, float]] = None,
        fingerprint_similarity: str = "soft_tanimoto",
        anchor_weight: float = 0.0,
        anchor_top_k_fraction: float = 0.0,
        formula_weight: float = 0.0,
        formula_similarity: str = "cosine",
        reuse_penalty: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        self.fingerprint_weight = fingerprint_weight
        self.task_weights = dict(task_weights or {"MSMS": 1.0, "NMR": 1.0, "IR": 1.0})
        if fingerprint_similarity not in {"soft_tanimoto", "cosine"}:
            raise ValueError(
                "fingerprint_similarity must be 'soft_tanimoto' or 'cosine'."
            )
        self.fingerprint_similarity = fingerprint_similarity
        if not 0.0 <= anchor_weight <= 1.0:
            raise ValueError("anchor_weight must be between 0 and 1.")
        if not 0.0 <= anchor_top_k_fraction <= 1.0:
            raise ValueError(
                "anchor_top_k_fraction must be between 0 and 1."
            )
        self.anchor_weight = float(anchor_weight)
        self.anchor_top_k_fraction = float(anchor_top_k_fraction)
        self.formula_weight = formula_weight
        if reuse_penalty < 0:
            raise ValueError("reuse_penalty must be non-negative.")
        self.reuse_penalty = float(reuse_penalty)
        if formula_similarity not in {"cosine", "weighted_jaccard"}:
            raise ValueError(
                "formula_similarity must be 'cosine' or 'weighted_jaccard'."
            )
        self.formula_similarity = formula_similarity
        self.eps = eps

    def _select_with_anchor_quota(
        self,
        total: torch.Tensor,
        anchor_scores: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k while retaining a configurable initial-model quota."""
        quota = min(
            top_k,
            math.ceil(top_k * self.anchor_top_k_fraction),
        )
        if quota == 0:
            return torch.topk(total, k=top_k, dim=1)

        primary_indices = torch.topk(total, k=top_k, dim=1).indices
        anchor_indices = torch.topk(anchor_scores, k=quota, dim=1).indices
        selected_rows = []
        selected_scores = []
        for row in range(total.shape[0]):
            selected: list[int] = []
            seen: set[int] = set()
            for index in anchor_indices[row].tolist():
                if torch.isfinite(anchor_scores[row, index]) and index not in seen:
                    selected.append(index)
                    seen.add(index)
            for index in primary_indices[row].tolist():
                if torch.isfinite(total[row, index]) and index not in seen:
                    selected.append(index)
                    seen.add(index)
                if len(selected) == top_k:
                    break

            padded = selected + [-1] * (top_k - len(selected))
            row_indices = torch.tensor(padded, device=total.device, dtype=torch.long)
            safe_indices = row_indices.clamp_min(0)
            row_scores = total[row, safe_indices].masked_fill(row_indices.lt(0), -torch.inf)
            order = torch.argsort(row_scores, descending=True)
            selected_rows.append(row_indices[order])
            selected_scores.append(row_scores[order])
        return torch.stack(selected_scores), torch.stack(selected_rows)

    def _soft_tanimoto(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        intersection = query @ keys.T
        denominator = query.sum(dim=1, keepdim=True) + keys.sum(dim=1).unsqueeze(0) - intersection
        return intersection / denominator.clamp_min(self.eps)

    def _cosine(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return F.normalize(query, dim=-1) @ F.normalize(keys, dim=-1).T

    def _standardize(self, scores: torch.Tensor) -> torch.Tensor:
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (scores - mean) / std

    def _standardize_masked(
        self, scores: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Standardize only valid candidates and give invalid ones no score."""
        mask = valid_mask.to(device=scores.device, dtype=torch.bool)
        count = mask.sum(dim=1, keepdim=True)
        safe_count = count.clamp_min(1).to(dtype=scores.dtype)
        masked_scores = torch.where(mask, scores, torch.zeros_like(scores))
        mean = masked_scores.sum(dim=1, keepdim=True) / safe_count
        centered = torch.where(mask, scores - mean, torch.zeros_like(scores))
        variance = centered.square().sum(dim=1, keepdim=True) / safe_count
        std = variance.sqrt().clamp_min(self.eps)
        standardized = (scores - mean) / std
        return torch.where(mask, standardized, torch.zeros_like(scores))

    def _formula_score(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        if self.formula_similarity == "cosine":
            return self._cosine(query, keys)
        intersection = torch.minimum(query[:, None, :], keys[None, :, :]).sum(dim=-1)
        union = torch.maximum(query[:, None, :], keys[None, :, :]).sum(dim=-1)
        return intersection / union.clamp_min(self.eps)

    def rank(
        self,
        predictions: Mapping[str, object],
        candidate_fingerprints: torch.Tensor,
        candidate_tasks: Mapping[str, torch.Tensor],
        top_k: int,
        anchor_fingerprint_logits: Optional[torch.Tensor] = None,
        anchor_candidate_fingerprints: Optional[torch.Tensor] = None,
        query_formula_counts: Optional[torch.Tensor] = None,
        candidate_formula_counts: Optional[torch.Tensor] = None,
        candidate_selection_counts: Optional[torch.Tensor] = None,
    ) -> RetrievalResult:
        fused_logits = predictions["fused_fingerprint_logits"]
        task_logits = predictions["task_logits"]
        modality_weights = predictions["modality_weights"]
        modality_order = predictions["modality_order"]
        if not isinstance(fused_logits, torch.Tensor):
            raise TypeError("fused_fingerprint_logits must be a tensor.")
        if not isinstance(task_logits, dict):
            raise TypeError("task_logits must be a dictionary.")
        if not isinstance(modality_weights, torch.Tensor) or not isinstance(modality_order, tuple):
            raise TypeError("Invalid modality weight output.")

        device = fused_logits.device
        candidate_fp = candidate_fingerprints.to(device=device, dtype=fused_logits.dtype)
        n_candidates = candidate_fp.shape[0]
        if top_k <= 0 or top_k > n_candidates:
            raise ValueError("top_k must be between 1 and the number of candidates.")
        if candidate_fp.ndim != 2 or candidate_fp.shape[1] != fused_logits.shape[1]:
            raise ValueError(
                "Candidate fingerprints must have shape [n_candidates, fingerprint_dim]."
            )
        if candidate_selection_counts is not None:
            selection_counts = candidate_selection_counts.to(
                device=device, dtype=fused_logits.dtype
            ).reshape(-1)
            if selection_counts.shape[0] != n_candidates:
                raise ValueError(
                    "candidate_selection_counts must have one value per candidate."
                )
            if torch.any(selection_counts < 0) or not torch.isfinite(selection_counts).all():
                raise ValueError("candidate_selection_counts must be finite and non-negative.")
        else:
            selection_counts = None

        query_fp_valid = torch.isfinite(fused_logits).all(dim=1)
        candidate_fp_valid = torch.isfinite(candidate_fp).all(dim=1)
        safe_query_fp = torch.nan_to_num(torch.sigmoid(fused_logits))
        safe_candidate_fp = torch.nan_to_num(candidate_fp)
        fp_scores = (
            self._soft_tanimoto(safe_query_fp, safe_candidate_fp)
            if self.fingerprint_similarity == "soft_tanimoto"
            else self._cosine(safe_query_fp, safe_candidate_fp)
        )
        components: Dict[str, torch.Tensor] = {"fingerprint": fp_scores}
        fingerprint_valid = query_fp_valid.unsqueeze(1) & candidate_fp_valid.unsqueeze(0)
        total = self.fingerprint_weight * self._standardize_masked(
            fp_scores, fingerprint_valid
        )
        anchor_scores = None
        anchor_valid = None
        if self.anchor_weight > 0 or self.anchor_top_k_fraction > 0:
            if (
                anchor_fingerprint_logits is None
                or anchor_candidate_fingerprints is None
            ):
                raise ValueError(
                    "anchor_fingerprint_logits and anchor_candidate_fingerprints "
                    "are required when anchor retrieval is enabled."
                )
            anchor_logits = anchor_fingerprint_logits.to(
                device=device, dtype=fused_logits.dtype
            )
            anchor_candidate_fp = anchor_candidate_fingerprints.to(
                device=device, dtype=fused_logits.dtype
            )
            if anchor_logits.shape != fused_logits.shape:
                raise ValueError(
                    "anchor_fingerprint_logits must match fused_fingerprint_logits."
                )
            if anchor_candidate_fp.shape != candidate_fp.shape:
                raise ValueError(
                    "anchor_candidate_fingerprints must match candidate_fingerprints."
                )
            anchor_query_valid = torch.isfinite(anchor_logits).all(dim=1)
            anchor_candidate_valid = torch.isfinite(anchor_candidate_fp).all(dim=1)
            safe_anchor_fp = torch.nan_to_num(torch.sigmoid(anchor_logits))
            safe_anchor_candidate_fp = torch.nan_to_num(anchor_candidate_fp)
            anchor_scores = (
                self._soft_tanimoto(safe_anchor_fp, safe_anchor_candidate_fp)
                if self.fingerprint_similarity == "soft_tanimoto"
                else self._cosine(safe_anchor_fp, safe_anchor_candidate_fp)
            )
            anchor_valid = (
                anchor_query_valid.unsqueeze(1)
                & anchor_candidate_valid.unsqueeze(0)
            )
            anchor_standardized = self._standardize_masked(
                anchor_scores, anchor_valid
            )
            current_standardized = self._standardize_masked(
                fp_scores, fingerprint_valid
            )
            both_valid = fingerprint_valid & anchor_valid
            blended_fingerprint = torch.where(
                both_valid,
                (1.0 - self.anchor_weight) * current_standardized
                + self.anchor_weight * anchor_standardized,
                torch.where(
                    fingerprint_valid,
                    current_standardized,
                    anchor_standardized,
                ),
            )
            total = self.fingerprint_weight * blended_fingerprint
            components["anchor_fingerprint"] = anchor_scores
            components["blended_fingerprint"] = blended_fingerprint

        if query_formula_counts is not None and candidate_formula_counts is not None:
            query_formula = query_formula_counts.to(
                device=device, dtype=fused_logits.dtype
            )
            candidate_formula = candidate_formula_counts.to(
                device=device, dtype=fused_logits.dtype
            )
            if (
                query_formula.ndim != 2
                or query_formula.shape[0] != fused_logits.shape[0]
                or candidate_formula.ndim != 2
                or candidate_formula.shape[0] != n_candidates
                or candidate_formula.shape[1] != query_formula.shape[1]
            ):
                raise ValueError(
                    "Formula counts must have shapes [n_queries, n_elements] and "
                    "[n_candidates, n_elements]."
                )
            formula_score = self._formula_score(query_formula, candidate_formula)
            components["formula"] = formula_score
            query_valid = query_formula.sum(dim=1, keepdim=True).gt(0)
            candidate_valid = candidate_formula.sum(dim=1).gt(0).unsqueeze(0)
            formula_valid = query_valid & candidate_valid
            total = total + self.formula_weight * self._standardize_masked(
                formula_score, formula_valid
            )
        elif self.formula_weight != 0:
            raise ValueError(
                "Formula counts are required when formula_weight is non-zero."
            )

        modality_index = {name: idx for idx, name in enumerate(modality_order)}
        for name, logits in task_logits.items():
            if name not in candidate_tasks or name not in self.task_weights:
                continue
            keys = candidate_tasks[name].to(device=device, dtype=logits.dtype)
            if keys.ndim != 2 or keys.shape[0] != n_candidates or keys.shape[1] != logits.shape[1]:
                raise ValueError(
                    f"Candidate task {name!r} must have shape "
                    f"[{n_candidates}, {logits.shape[1]}]."
                )
            query_task_valid = torch.isfinite(logits).all(dim=1)
            candidate_task_valid = torch.isfinite(keys).all(dim=1)
            safe_logits = torch.nan_to_num(torch.sigmoid(logits))
            safe_keys = torch.nan_to_num(keys)
            task_score = self._cosine(safe_logits, safe_keys)
            components[name] = task_score
            if modality_index.get(name) is None:
                raise ValueError(
                    f"Task output {name!r} is absent from modality_order."
                )
            confidence = torch.nan_to_num(
                modality_weights[:, modality_index[name]]
            ).unsqueeze(-1)
            task_valid = query_task_valid.unsqueeze(1) & candidate_task_valid.unsqueeze(0)
            total = total + (
                self.task_weights[name]
                * confidence
                * self._standardize_masked(task_score, task_valid)
            )

        if self.reuse_penalty > 0 and selection_counts is not None:
            exposure = torch.log1p(selection_counts).unsqueeze(0)
            components["reuse_penalty"] = exposure.expand_as(total)
            total = total - self.reuse_penalty * exposure

        # A model-produced candidate fingerprint containing NaN or Inf cannot
        # be compared reliably. Preserve its global row position, but exclude
        # it from top-k selection instead of letting it poison standardization
        # for every otherwise valid candidate.
        retrieval_valid = fingerprint_valid
        if anchor_valid is not None:
            retrieval_valid = retrieval_valid | anchor_valid
        total = total.masked_fill(~retrieval_valid, -torch.inf)

        if anchor_scores is not None and self.anchor_top_k_fraction > 0:
            anchor_rank_scores = anchor_scores.masked_fill(~anchor_valid, -torch.inf)
            scores, indices = self._select_with_anchor_quota(
                total, anchor_rank_scores, top_k
            )
        else:
            scores, indices = torch.topk(total, k=top_k, dim=1)
        valid_mask = torch.isfinite(scores)
        indices = indices.masked_fill(~valid_mask, -1)
        return RetrievalResult(
            indices=indices,
            scores=scores,
            valid_mask=valid_mask,
            components=components,
        )

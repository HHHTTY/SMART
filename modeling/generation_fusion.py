"""Generation-path multimodal fusion without auxiliary retrieval targets.

Models A and B consume pooled input embeddings and return a context residual.
Model C uses a READ-style self-adaptive attention fusion layer after the shared
encoder. The tokenizer, modality sequence layout, and SMILES decoder interface
remain unchanged.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional

import torch
from torch import nn


CANONICAL_MODALITIES = ("Formula", "NMR", "MSMS", "IR")
DEFAULT_MODALITY_MAP = {
    "Formula": "Formula",
    "MSMS": "MSMS",
    "HNMR": "HNMR",
    "CNMR": "CNMR",
    "Multiplets": "HNMR",
    "Carbon": "CNMR",
    "IR": "IR",
}


def _as_bool_vector(
    value: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize optional availability values to a batch-shaped bool tensor."""
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    result = value.to(device=device, dtype=torch.bool).reshape(-1)
    if result.numel() != batch_size:
        raise ValueError(
            "Modality availability must have one value per batch item; "
            f"got {result.numel()} for batch size {batch_size}."
        )
    return result


class READSelfAdaptiveFusion(nn.Module):
    """Final token-level fusion layer used by READ model C.

    READ adapts only the query, key, and value projections at test time. The
    remaining source-trained attention/output/FFN path stays frozen during TTA.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        ffn_dim: Optional[int] = None,
        selective_adaptation: bool = False,
        selective_adapter_rank: int = 16,
        selective_router_temperature: float = 0.001,
        selective_use_gumbel: bool = True,
        selective_router_mode: str = "global",
        selective_router_sample_scale: float = 1.0,
        selective_router_no_update_threshold: float = 0.85,
        selective_router_no_update_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads

        self.attention_norm = nn.LayerNorm(self.d_model)
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.attention_output = nn.Linear(self.d_model, self.d_model)
        self.attention_dropout = nn.Dropout(dropout)

        hidden_dim = int(ffn_dim or 2 * self.d_model)
        self.ffn_norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        # Paper-faithful selective adaptation: one residual bottleneck Phi_m
        # per modality and a learnable shift semaphore S.  The bottleneck is
        # the lightweight implementation of Phi_m used for high-dimensional
        # generation states; it preserves the paper's Phi_m + I residual form.
        self.selective_adapter_enabled = bool(selective_adaptation)
        self.selective_adapter_rank = int(selective_adapter_rank)
        self.selective_router_temperature = float(selective_router_temperature)
        self.selective_use_gumbel = bool(selective_use_gumbel)
        self.selective_router_mode = str(selective_router_mode)
        self.selective_router_sample_scale = float(selective_router_sample_scale)
        self.selective_router_no_update_threshold = float(
            selective_router_no_update_threshold
        )
        self.selective_router_no_update_temperature = float(
            selective_router_no_update_temperature
        )
        if self.selective_router_mode not in {
            "global",
            "samplewise",
            "loo_softmax",
            "loo_sigmoid",
        }:
            raise ValueError(
                "selective_router_mode must be global, samplewise, loo_softmax, "
                "or loo_sigmoid."
            )
        if self.selective_router_sample_scale < 0:
            raise ValueError("selective_router_sample_scale must be non-negative.")
        if self.selective_router_no_update_temperature <= 0:
            raise ValueError("selective_router_no_update_temperature must be positive.")
        if self.selective_adapter_enabled:
            if self.selective_adapter_rank <= 0:
                raise ValueError("selective_adapter_rank must be positive")
            if self.selective_router_temperature <= 0:
                raise ValueError("selective_router_temperature must be positive")
            down = torch.empty(4, self.d_model, self.selective_adapter_rank)
            nn.init.normal_(down, mean=0.0, std=0.02)
            self.selective_adapter_down = nn.Parameter(down)
            # Zero output starts Phi_m at the identity while allowing gradients
            # to learn the residual during source training and TTA.
            self.selective_adapter_up = nn.Parameter(
                torch.zeros(4, self.selective_adapter_rank, self.d_model)
            )
            self.selective_router_logits = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter("selective_adapter_down", None)
            self.register_parameter("selective_adapter_up", None)
            self.register_parameter("selective_router_logits", None)
        self.last_selective_router_weights: Optional[torch.Tensor] = None
        self.last_selective_adapter_norm: Optional[torch.Tensor] = None
        self.last_selective_shift_gate: Optional[torch.Tensor] = None
        # Runtime-only scores from decoder leave-one-modality-out utility.
        # Keeping them outside the state dict preserves old checkpoints.
        self.selective_router_external_scores: Optional[torch.Tensor] = None

    def _samplewise_selective_router_logits(
        self,
        hidden_states: torch.Tensor,
        valid: torch.Tensor,
        safe_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a per-example router from modality agreement.

        The source checkpoint only contains four global router logits.  This
        mode keeps those logits as the prior and adds a deterministic,
        sample-level reliability correction based on pooled encoder states.
        It therefore remains checkpoint-compatible while allowing multiple
        modalities to be shifted for one sample.  The clean anchor is Formula
        when available, then NMR, then the mean of available modalities.
        """
        batch_size = hidden_states.shape[0]
        dtype = hidden_states.dtype
        pooled = hidden_states.new_zeros((batch_size, 4, self.d_model))
        counts = hidden_states.new_zeros((batch_size, 4))
        pooled.scatter_add_(
            1,
            safe_ids.unsqueeze(-1).expand(-1, -1, self.d_model),
            hidden_states * valid.unsqueeze(-1).to(dtype),
        )
        counts.scatter_add_(1, safe_ids, valid.to(dtype))
        pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
        present = counts.gt(0)
        normalized = torch.nn.functional.normalize(pooled.float(), dim=-1)
        formula_anchor = normalized[:, 0]
        nmr_anchor = normalized[:, 1]
        fallback = (
            normalized * present.unsqueeze(-1).float()
        ).sum(dim=1) / present.sum(dim=1, keepdim=True).clamp_min(1).float()
        anchor = torch.where(
            present[:, 0].unsqueeze(-1), formula_anchor, nmr_anchor
        )
        anchor = torch.where(
            (present[:, 0] | present[:, 1]).unsqueeze(-1), anchor, fallback
        )
        if self.selective_router_external_scores is not None:
            external = self.selective_router_external_scores.to(
                device=hidden_states.device, dtype=torch.float32
            )
            if external.shape != (batch_size, 4):
                raise ValueError(
                    "External selective router scores must have shape [batch, 4]."
                )
            agreement = external
        else:
            agreement = (normalized * anchor.unsqueeze(1)).sum(dim=-1)
        agreement = agreement.clamp(-1.0, 1.0)
        # Center the correction so the global source prior remains the mean
        # level, while each example can independently suppress shifted modes.
        agreement = agreement - (
            agreement * present.float()
        ).sum(dim=1, keepdim=True) / present.sum(dim=1, keepdim=True).clamp_min(1)
        global_logits = self.selective_router_logits.float().view(1, 4)
        sample_logits = global_logits + self.selective_router_sample_scale * agreement
        sample_logits = sample_logits.masked_fill(~present, -1e4)
        safe_present = present.clone()
        safe_present[:, 1] |= ~safe_present.any(dim=1)
        sample_logits = sample_logits.masked_fill(~safe_present, -1e4)
        return sample_logits, present

    def qkv_parameters(self) -> Iterable[nn.Parameter]:
        """Yield exactly the parameters READ is allowed to update at test time."""
        yield from self.query.parameters()
        yield from self.key.parameters()
        yield from self.value.parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_token_counts: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("READ fusion hidden states must have shape [batch, seq, dim].")
        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected hidden dimension {self.d_model}, got {hidden_states.shape[-1]}."
            )
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "READ fusion attention mask must match the hidden-state batch and sequence."
            )
        if key_token_counts is not None and key_token_counts.shape != hidden_states.shape[:2]:
            raise ValueError(
                "READ key token counts must match the hidden-state batch and sequence."
            )
        if modality_ids is not None and modality_ids.shape != hidden_states.shape[:2]:
            raise ValueError("READ modality IDs must match the hidden-state batch and sequence.")

        batch_size, sequence_length, _ = hidden_states.shape
        valid = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not valid.any(dim=1).all():
            raise ValueError("Every READ fusion sample must contain at least one valid token.")

        if self.selective_adapter_enabled and modality_ids is not None:
            assert self.selective_adapter_down is not None
            assert self.selective_adapter_up is not None
            assert self.selective_router_logits is not None
            safe_ids = modality_ids.to(device=hidden_states.device, dtype=torch.long).clamp(0, 3)
            normalized = torch.nn.functional.layer_norm(hidden_states, (self.d_model,))
            down = self.selective_adapter_down.to(hidden_states.device, hidden_states.dtype)[safe_ids]
            up = self.selective_adapter_up.to(hidden_states.device, hidden_states.dtype)[safe_ids]
            bottleneck = torch.einsum("bsd,bsdr->bsr", normalized, down)
            delta = torch.einsum("bsr,bsrd->bsd", bottleneck, up)
            if self.selective_router_mode in {
                "samplewise",
                "loo_softmax",
                "loo_sigmoid",
            }:
                router_logits, present = self._samplewise_selective_router_logits(
                    hidden_states, valid, safe_ids
                )
                if self.selective_router_mode == "loo_sigmoid":
                    # Independent gates allow several modalities to be shifted
                    # simultaneously; this avoids forcing all four branches to
                    # compete for one unit of softmax mass.
                    router = torch.sigmoid(
                        router_logits / self.selective_router_temperature
                    )
                    router = router.masked_fill(~present, 0.0)
                else:
                    router = torch.softmax(
                        router_logits / self.selective_router_temperature, dim=1
                    )
                clean_confidence = router.masked_fill(~present, -1.0).amax(dim=1)
                shift_gate = torch.sigmoid(
                    (
                        self.selective_router_no_update_threshold
                        - clean_confidence
                    )
                    / self.selective_router_no_update_temperature
                )
                route_weight = (
                    (1.0 - router.gather(1, safe_ids))
                    * shift_gate.unsqueeze(1)
                    * valid.to(hidden_states.dtype)
                )
                self.last_selective_shift_gate = shift_gate.detach()
                self.last_selective_router_weights = router.detach().mean(dim=0)
            else:
                router_logits = self.selective_router_logits.float()
                if self.selective_use_gumbel and self.training:
                    uniform = torch.rand_like(router_logits).clamp_(1e-6, 1 - 1e-6)
                    router_logits = router_logits - torch.log(-torch.log(uniform))
                router = torch.softmax(
                    router_logits / self.selective_router_temperature, dim=0
                ).to(device=hidden_states.device, dtype=hidden_states.dtype)
                route_weight = (1.0 - router[safe_ids]) * valid.to(hidden_states.dtype)
                self.last_selective_shift_gate = torch.ones(
                    batch_size, device=hidden_states.device, dtype=hidden_states.dtype
                )
            hidden_states = hidden_states + route_weight.unsqueeze(-1) * delta
            self.last_selective_adapter_norm = delta.detach().norm(dim=-1).mean(dim=1)
        else:
            self.last_selective_router_weights = None
            self.last_selective_adapter_norm = None
            self.last_selective_shift_gate = None

        normalized = self.attention_norm(hidden_states)

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(
                batch_size, sequence_length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query = split_heads(self.query(normalized))
        key = split_heads(self.key(normalized))
        value = split_heads(self.value(normalized))

        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        if key_token_counts is not None:
            counts = key_token_counts.to(device=scores.device, dtype=scores.dtype)
            if (counts[valid] <= 0).any():
                raise ValueError("READ key token counts must be positive for valid tokens.")
            # Each modality receives comparable prior mass even when, for
            # example, an IR sequence has far more tokens than Formula/NMR.
            scores = scores - counts.clamp_min(1.0).log()[:, None, None, :]
        scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value.float()).to(dtype=hidden_states.dtype)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.d_model
        )
        attended = self.attention_output(attended)
        hidden_states = hidden_states + self.attention_dropout(attended)
        hidden_states = hidden_states + self.ffn_dropout(
            self.ffn(self.ffn_norm(hidden_states))
        )
        return torch.where(valid.unsqueeze(-1), hidden_states, torch.zeros_like(hidden_states))


class SelectiveModalityFusion(nn.Module):
    """Official-style full modality adaptors, enabled only during TTA."""

    def __init__(
        self,
        d_model: int,
        *,
        router_temperature: float = 0.001,
        use_gumbel: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if router_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        self.d_model = int(d_model)
        self.router_temperature = float(router_temperature)
        self.use_gumbel = bool(use_gumbel)
        # Match the official implementation: Phi_m is a full zero matrix and
        # the effective transform is I + Phi_m at test time.
        self.modality_adaptors = nn.Parameter(
            torch.zeros(4, self.d_model, self.d_model)
        )
        self.shift_semaphore = nn.Parameter(torch.zeros(4))
        self.adaptation_enabled = False
        self.last_router_weights: Optional[torch.Tensor] = None
        self.last_transform_norm: Optional[torch.Tensor] = None

    def enable_adaptation(self, enabled: bool = True) -> None:
        self.adaptation_enabled = bool(enabled)

    def parameters_for_tta(self) -> Iterable[nn.Parameter]:
        yield self.modality_adaptors
        yield self.shift_semaphore

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model:
            raise ValueError("Selective hidden states must have shape [batch, seq, d_model].")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("Selective attention mask must match hidden states.")
        if modality_ids.shape != hidden_states.shape[:2]:
            raise ValueError("Selective modality IDs must match hidden states.")

        valid = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not self.adaptation_enabled:
            self.last_router_weights = None
            self.last_transform_norm = None
            return torch.where(
                valid.unsqueeze(-1), hidden_states, torch.zeros_like(hidden_states)
            )

        batch_size = hidden_states.shape[0]
        safe_ids = modality_ids.to(device=hidden_states.device, dtype=torch.long).clamp(0, 3)
        adaptors = self.modality_adaptors.to(
            hidden_states.device, hidden_states.dtype
        )
        # Do not index a full d x d matrix with [batch, sequence] IDs: that
        # materializes [batch, sequence, d, d]. Compute each modality branch
        # separately and mask it back into the token sequence instead.
        delta = torch.zeros_like(hidden_states)
        for modality_index in range(4):
            modality_mask = safe_ids.eq(modality_index).unsqueeze(-1)
            if modality_mask.any():
                delta = delta + torch.where(
                    modality_mask,
                    torch.matmul(hidden_states, adaptors[modality_index]),
                    torch.zeros_like(hidden_states),
                )

        router_logits = self.shift_semaphore.float().expand(batch_size, -1)
        if self.use_gumbel:
            router = torch.nn.functional.gumbel_softmax(
                router_logits,
                tau=self.router_temperature,
                hard=False,
                dim=-1,
            )
        else:
            router = torch.softmax(
                router_logits / self.router_temperature, dim=-1
            )
        router = router.to(device=hidden_states.device, dtype=hidden_states.dtype)

        route_weight = router.gather(1, safe_ids) * valid.to(hidden_states.dtype)
        transformed = hidden_states + route_weight.unsqueeze(-1) * delta
        self.last_router_weights = router.detach().mean(dim=0)
        self.last_transform_norm = delta.detach().norm(dim=-1).mean(dim=1)
        return torch.where(valid.unsqueeze(-1), transformed, torch.zeros_like(transformed))


class GenerationFusion(nn.Module):
    """Generation-conditioning variants used by the three pretraining runs.

    ``nmr_anchor_residual`` uses NMR as the default anchor and gates MS/IR
    residuals. ``reliability_router`` learns a masked soft router over the
    available modality summaries. ``read_saf`` adds a final token-level
    self-adaptive attention layer whose Q/K/V projections are repurposed by
    READ at test time. ``formula_cross_attention`` remains available only for
    loading the superseded model-C experiment.
    """

    _ALIASES = {
        "A": "nmr_anchor_residual",
        "nmr_residual": "nmr_anchor_residual",
        "B": "reliability_router",
        "task_space_residual": "reliability_router",
        "C": "read_saf",
        "READ": "read_saf",
        "self_adaptive_attention": "read_saf",
        "selective": "selective_only",
        "selective_tta": "selective_only",
        "formula_attention": "formula_cross_attention",
        "nmr_query_cross_attention": "formula_cross_attention",
        "QMF": "qmf_quality_router",
        "quality_aware": "qmf_quality_router",
    }

    def __init__(
        self,
        d_model: int,
        variant: str,
        modality_map: Optional[Mapping[str, str]] = None,
        dropout: float = 0.0,
        read_n_heads: int = 4,
        quality_temperature: float = 1.0,
        selective_adapter_rank: int = 16,
        selective_router_temperature: float = 0.001,
        selective_use_gumbel: bool = True,
        selective_router_mode: str = "global",
        selective_router_sample_scale: float = 1.0,
        selective_router_no_update_threshold: float = 0.85,
        selective_router_no_update_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        canonical_variant = self._ALIASES.get(str(variant), str(variant))
        valid_variants = {
            "nmr_anchor_residual",
            "reliability_router",
            "formula_cross_attention",
            "read_saf",
            "selective_read_saf",
            "selective_only",
            "qmf_quality_router",
        }
        if canonical_variant not in valid_variants:
            raise ValueError(
                f"Unknown generation fusion variant {variant!r}; "
                f"expected one of {sorted(valid_variants)}."
            )
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if quality_temperature <= 0:
            raise ValueError("quality_temperature must be positive.")

        self.d_model = int(d_model)
        self.variant = canonical_variant
        self.quality_temperature = float(quality_temperature)
        self.modality_map = dict(DEFAULT_MODALITY_MAP)
        if modality_map:
            self.modality_map.update({str(k): str(v) for k, v in modality_map.items()})

        if self.variant == "selective_only":
            # Keep this experiment structurally isolated: no READ-SAF, legacy
            # gate/router, pooled projection, or cross-attention parameters are
            # instantiated. The fixed scale exists only for wrapper logging.
            self.read_fusion = None
            self.selective_fusion = SelectiveModalityFusion(
                d_model=d_model,
                router_temperature=selective_router_temperature,
                use_gumbel=selective_use_gumbel,
            )
            self.register_buffer("output_scale", torch.tensor(1.0), persistent=True)
            self.quality_heads = None
            self.last_quality_scores = None
            self.last_quality_availability = None
            self.last_quality_weights = None
            return

        hidden = max(32, d_model // 2)
        self.modality_projection = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                )
                for name in CANONICAL_MODALITIES
            }
        )
        self.nmr_merge = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.fallback = nn.Parameter(torch.zeros(d_model))

        self.anchor_gate = nn.Sequential(
            nn.Linear(3 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.router = nn.Sequential(
            nn.Linear(4 * d_model + 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )
        self.router_residual = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.quality_heads = (
            nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden, 1),
                    )
                    for name in CANONICAL_MODALITIES
                }
            )
            if self.variant == "qmf_quality_router"
            else None
        )
        self.last_quality_scores: Optional[torch.Tensor] = None
        self.last_quality_availability: Optional[torch.Tensor] = None
        self.last_quality_weights: Optional[torch.Tensor] = None

        # Four heads are not available for every tiny smoke-test dimension.
        n_heads = min(4, d_model)
        while n_heads > 1 and d_model % n_heads != 0:
            n_heads -= 1
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        if read_n_heads <= 0 or d_model % read_n_heads != 0:
            raise ValueError("read_n_heads must be positive and divide d_model.")
        self.read_fusion = (
            READSelfAdaptiveFusion(
                d_model=d_model,
                n_heads=read_n_heads,
                dropout=dropout,
                ffn_dim=2 * d_model,
            )
            if self.variant in {"read_saf", "selective_read_saf"}
            else None
        )
        if self.read_fusion is not None and self.variant == "selective_read_saf":
            self.read_fusion = READSelfAdaptiveFusion(
                d_model=d_model,
                n_heads=read_n_heads,
                dropout=dropout,
                ffn_dim=2 * d_model,
                selective_adaptation=True,
                selective_adapter_rank=selective_adapter_rank,
                selective_router_temperature=selective_router_temperature,
                selective_use_gumbel=selective_use_gumbel,
                selective_router_mode=selective_router_mode,
                selective_router_sample_scale=selective_router_sample_scale,
                selective_router_no_update_threshold=selective_router_no_update_threshold,
                selective_router_no_update_temperature=selective_router_no_update_temperature,
            )
        self.selective_fusion = (
            SelectiveModalityFusion(
                d_model=d_model,
                router_temperature=selective_router_temperature,
                use_gumbel=selective_use_gumbel,
            )
            if self.variant == "selective_only"
            else None
        )

        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Start exactly at the normal concatenation baseline. The first update
        # learns whether a residual is useful; subsequent updates then train
        # the variant-specific router/attention through the non-zero scale.
        initial_scale = (
            1.0
            if self.variant
            in {"read_saf", "selective_read_saf", "selective_only", "qmf_quality_router"}
            else 0.0
        )
        self.output_scale = nn.Parameter(torch.tensor(initial_scale))

    @staticmethod
    def _canonicalize_variant(variant: str) -> str:
        return GenerationFusion._ALIASES.get(str(variant), str(variant))

    @property
    def uses_encoder_token_fusion(self) -> bool:
        """Whether this variant fuses full token states after the encoder."""
        return self.variant in {"read_saf", "selective_read_saf", "selective_only"}

    def read_qkv_parameters(self) -> list[nn.Parameter]:
        """Return the exact READ test-time trainable parameter set."""
        if self.read_fusion is None:
            raise RuntimeError("READ Q/K/V parameters require variant='read_saf'.")
        return list(self.read_fusion.qkv_parameters())

    def selective_tta_parameters(self) -> list[nn.Parameter]:
        if self.selective_fusion is None:
            raise RuntimeError("Selective TTA parameters require variant='selective_only'.")
        return list(self.selective_fusion.parameters_for_tta())

    def forward_encoder(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_token_counts: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the C-only final self-adaptive attention fusion layer."""
        if self.variant == "selective_only":
            if self.selective_fusion is None or modality_ids is None:
                raise RuntimeError("Selective-only fusion requires modality IDs.")
            return self.selective_fusion(
                hidden_states,
                attention_mask,
                modality_ids,
            )
        if self.read_fusion is None:
            raise RuntimeError("Encoder-token fusion is unavailable for this variant.")
        fused = self.read_fusion(
            hidden_states,
            attention_mask,
            key_token_counts=key_token_counts,
            modality_ids=modality_ids,
        )
        scale = self.output_scale.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states + scale * (fused - hidden_states)

    def _summary(
        self,
        pooled: Mapping[str, torch.Tensor],
        availability: Mapping[str, torch.Tensor],
        name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = pooled.get(name)
        if value is None:
            return (
                torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
                torch.zeros(batch_size, dtype=torch.bool, device=device),
            )
        value = value.to(device=device, dtype=dtype)
        if value.ndim != 2 or value.shape != (batch_size, self.d_model):
            raise ValueError(
                f"Pooled {name} representation must have shape "
                f"[{batch_size}, {self.d_model}], got {tuple(value.shape)}."
            )
        present = _as_bool_vector(availability.get(name), batch_size, device)
        value = torch.where(present.unsqueeze(-1), value, torch.zeros_like(value))
        value = self.modality_projection[name](value)
        value = torch.where(present.unsqueeze(-1), value, torch.zeros_like(value))
        return value, present

    def _nmr_summary(
        self,
        pooled: Mapping[str, torch.Tensor],
        availability: Mapping[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The wrapper normally supplies a combined NMR summary. Supporting H/C
        # separately here keeps the module safe for direct unit tests and TTA.
        if "NMR" in pooled:
            return self._summary(pooled, availability, "NMR", batch_size, device, dtype)
        h = pooled.get("HNMR")
        c = pooled.get("CNMR")
        if h is None and c is None:
            return (
                torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
                torch.zeros(batch_size, dtype=torch.bool, device=device),
            )
        zeros = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        h = zeros if h is None else h.to(device=device, dtype=dtype)
        c = zeros if c is None else c.to(device=device, dtype=dtype)
        h_present = (
            torch.zeros(batch_size, dtype=torch.bool, device=device)
            if "HNMR" not in pooled
            else _as_bool_vector(availability.get("HNMR"), batch_size, device)
        )
        c_present = (
            torch.zeros(batch_size, dtype=torch.bool, device=device)
            if "CNMR" not in pooled
            else _as_bool_vector(availability.get("CNMR"), batch_size, device)
        )
        present = h_present | c_present
        merged = torch.where(
            (h_present & c_present).unsqueeze(-1),
            self.nmr_merge(torch.cat([h, c], dim=-1)),
            torch.where(h_present.unsqueeze(-1), h, c),
        )
        merged = self.modality_projection["NMR"](merged)
        return torch.where(present.unsqueeze(-1), merged, torch.zeros_like(merged)), present

    @staticmethod
    def _fallback_anchor(
        nmr: torch.Tensor,
        nmr_present: torch.Tensor,
        msms: torch.Tensor,
        msms_present: torch.Tensor,
        ir: torch.Tensor,
        ir_present: torch.Tensor,
        formula: torch.Tensor,
        formula_present: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """Use NMR first and a masked mean when NMR is absent."""
        count = (
            msms_present.to(msms.dtype)
            + ir_present.to(ir.dtype)
            + formula_present.to(formula.dtype)
        ).clamp_min(1.0)
        mean = (
            msms * msms_present.unsqueeze(-1)
            + ir * ir_present.unsqueeze(-1)
            + formula * formula_present.unsqueeze(-1)
        ) / count.unsqueeze(-1)
        mean = torch.where(
            (msms_present | ir_present | formula_present).unsqueeze(-1),
            mean,
            fallback.view(1, -1),
        )
        return torch.where(nmr_present.unsqueeze(-1), nmr, mean)

    def forward(
        self,
        pooled: Mapping[str, torch.Tensor],
        availability: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return a batch of context vectors to add to encoder embeddings."""
        if self.uses_encoder_token_fusion:
            raise RuntimeError(
                "READ model C must call forward_encoder() on post-encoder token states."
            )
        availability = availability or {}
        first = next(iter(pooled.values()), None)
        if first is None:
            raise ValueError("At least one pooled modality representation is required.")
        if first.ndim != 2:
            raise ValueError("Pooled modality representations must be rank-2 tensors.")
        batch_size = first.shape[0]
        device, dtype = first.device, first.dtype

        formula, formula_present = self._summary(
            pooled, availability, "Formula", batch_size, device, dtype
        )
        nmr, nmr_present = self._nmr_summary(
            pooled, availability, batch_size, device, dtype
        )
        msms, msms_present = self._summary(
            pooled, availability, "MSMS", batch_size, device, dtype
        )
        ir, ir_present = self._summary(
            pooled, availability, "IR", batch_size, device, dtype
        )
        anchor = self._fallback_anchor(
            nmr,
            nmr_present,
            msms,
            msms_present,
            ir,
            ir_present,
            formula,
            formula_present,
            self.fallback.to(device=device, dtype=dtype),
        )

        if self.variant == "nmr_anchor_residual":
            gate_input = torch.cat([anchor, msms, ir], dim=-1)
            gate_logits = self.anchor_gate(gate_input)
            tta_gate_logit_delta = getattr(self, "tta_gate_logit_delta", None)
            if tta_gate_logit_delta is not None:
                gate_logits = gate_logits + tta_gate_logit_delta.to(
                    device=device, dtype=dtype
                ).view(1, 2)
            gates = torch.sigmoid(gate_logits)
            context = anchor
            context = context + gates[:, 0:1] * msms_present.unsqueeze(-1) * (msms - anchor)
            context = context + gates[:, 1:2] * ir_present.unsqueeze(-1) * (ir - anchor)
            context = context + 0.1 * formula_present.unsqueeze(-1) * (formula - anchor)
        elif self.variant == "reliability_router":
            tokens = torch.stack([formula, nmr, msms, ir], dim=1)
            present = torch.stack(
                [formula_present, nmr_present, msms_present, ir_present], dim=1
            )
            router_input = torch.cat([tokens.reshape(batch_size, -1), present.to(dtype)], dim=-1)
            router_logits = self.router(router_input)
            # Avoid all -inf rows in softmax for samples with no available token.
            safe_present = present.clone()
            safe_present[:, 1] |= ~safe_present.any(dim=1)
            router_logits = router_logits.masked_fill(~safe_present, -1e4)
            weights = torch.softmax(router_logits.float(), dim=-1).to(dtype)
            context = (weights.unsqueeze(-1) * tokens).sum(dim=1)
            context = anchor + self.router_residual(context - anchor)
        elif self.variant == "qmf_quality_router":
            if self.quality_heads is None:
                raise RuntimeError("QMF quality heads were not initialized.")
            tokens = torch.stack([formula, nmr, msms, ir], dim=1)
            present = torch.stack(
                [formula_present, nmr_present, msms_present, ir_present], dim=1
            )
            scores = torch.stack(
                [
                    self.quality_heads[name](tokens[:, index]).squeeze(-1)
                    for index, name in enumerate(CANONICAL_MODALITIES)
                ],
                dim=1,
            )
            safe_present = present.clone()
            safe_present[:, 1] |= ~safe_present.any(dim=1)
            masked_scores = scores.masked_fill(~safe_present, -1e4)
            weights = torch.softmax(
                masked_scores.float() / self.quality_temperature,
                dim=-1,
            ).to(dtype)
            context = (weights.unsqueeze(-1) * tokens).sum(dim=1)
            self.last_quality_scores = scores
            self.last_quality_availability = present
            self.last_quality_weights = weights
        else:
            # Formula is the query when present; otherwise fall back to the
            # same NMR/available-token anchor used by the other variants.
            query = torch.where(formula_present.unsqueeze(-1), formula, anchor)
            tokens = torch.stack([nmr, msms, ir], dim=1)
            present = torch.stack([nmr_present, msms_present, ir_present], dim=1)
            has_spectroscopy = present.any(dim=1)
            safe_present = present.clone()
            safe_present[:, 0] |= ~safe_present.any(dim=1)
            key_padding_mask = ~safe_present
            query = self.query_projection(query).unsqueeze(1)
            keys = self.key_projection(tokens)
            values = self.value_projection(tokens)
            attended, _ = self.cross_attention(
                query,
                keys,
                values,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            # A zero NMR sentinel only makes attention numerically safe. It is
            # not evidence: Formula-only examples must keep the anchor path.
            use_attention = formula_present & has_spectroscopy
            context = anchor + use_attention.unsqueeze(-1) * attended.squeeze(1)

        return self.output(context) * self.output_scale.to(dtype=dtype)

"""Entropy-minimization test-time adaptation for spectra-to-SMILES models.

The module supports the original sequence-level SGEM objective and Huang et al.'s
token-level autoregressive EM objective. It scores generated beam-search paths
with teacher forcing and updates only an explicitly selected parameter subset.
The module is deliberately independent of dataset labels and does not derive
Formula constraints from ``target_smiles``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .read_tta import observed_input_batch


Tensor = torch.Tensor
AllowedTokenMaskBuilder = Callable[[Tensor, Tensor], Optional[Tensor]]
AcceptanceFn = Callable[[Tensor, Tensor], bool]
BeamRerankFn = Callable[[nn.Module, Mapping[str, Any], Tensor, int], Tensor]
AdaptSequenceFilterFn = Callable[[Tensor], Optional[Tensor]]


@dataclass(frozen=True)
class SGEMConfig:
    """Runtime settings for one-sample SGEM adaptation."""

    adapt_beams: int = 5
    final_beams: int = 10
    steps: int = 1
    lr: float = 3e-6
    lr_final: Optional[float] = None
    alpha: float = 1.25
    temperature: float = 2.5
    entropy_objective: str = "renyi"
    adaptation_objective: str = "legacy_sgem"
    token_em_normalization: str = "trajectory"
    confidence_threshold: Optional[float] = None
    negative_coeff: float = 0.25
    negative_threshold_coeff: float = 0.4
    kl_weight: float = 0.1
    max_grad_norm: float = 1.0
    max_parameter_delta: float = 0.0
    parameter_scope: str = "adapter"
    episodic: bool = True
    optimizer: str = "adamw"
    refresh_pseudo_each_step: bool = False
    beam_rerank: str = "none"
    adapt_all_beams: bool = False

    def validate(self) -> None:
        if self.adapt_beams < 1 or self.final_beams < 1:
            raise ValueError("SGEM beam widths must be positive.")
        if self.steps < 1:
            raise ValueError("SGEM requires at least one adaptation step.")
        if self.lr <= 0.0:
            raise ValueError("SGEM learning rate must be positive.")
        if self.lr_final is not None and self.lr_final < 0.0:
            raise ValueError("SGEM final learning rate must be non-negative.")
        if self.alpha <= 0.0 or abs(self.alpha - 1.0) < 1e-8:
            raise ValueError("SGEM alpha must be positive and different from one.")
        if self.temperature <= 0.0:
            raise ValueError("SGEM temperature must be positive.")
        if self.entropy_objective not in {"renyi", "shannon"}:
            raise ValueError("SGEM entropy objective must be renyi or shannon.")
        if self.adaptation_objective not in {
            "legacy_sgem",
            "token_entropy",
            "token_policy_gradient",
            "token_em",
        }:
            raise ValueError(
                "Adaptation objective must be legacy_sgem, token_entropy, "
                "token_policy_gradient, or token_em."
            )
        if self.token_em_normalization not in {"trajectory", "tokens"}:
            raise ValueError(
                "Token EM normalization must be trajectory or tokens."
            )
        if self.confidence_threshold is not None and not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("SGEM confidence threshold must be between zero and one.")
        if self.negative_coeff < 0.0 or self.negative_threshold_coeff < 0.0:
            raise ValueError("SGEM negative-sampling settings must be non-negative.")
        if self.kl_weight < 0.0:
            raise ValueError("SGEM KL weight must be non-negative.")
        if self.max_grad_norm < 0.0 or self.max_parameter_delta < 0.0:
            raise ValueError("SGEM trust-region settings must be non-negative.")
        if self.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError("SGEM optimizer must be one of adam, adamw, or sgd.")
        if self.beam_rerank not in {
            "none",
            "nll",
            "entropy",
            "confidence",
            "formula",
        }:
            raise ValueError(
                "SGEM beam_rerank must be none, nll, entropy, confidence, or formula."
            )


@dataclass(frozen=True)
class SGEMTTAResult:
    """Diagnostics and predictions for one adapted target batch."""

    predictions_before: Tensor
    predictions_after: Tensor
    pseudo_sequence: Tensor
    updated: bool
    accepted: bool
    loss: float
    generalized_entropy: float
    policy_gradient: float
    token_entropy: float
    negative_sampling: float
    kl_anchor: float
    adaptation_sequence_count: int
    gradient_norm: float
    max_gradient_norm: float
    parameter_delta: float
    skip_reason: Optional[str] = None


def _validate_sequence_inputs(logits: Tensor, token_mask: Tensor) -> None:
    if logits.ndim != 3:
        raise ValueError("SGEM logits must have shape [batch, length, vocabulary].")
    if token_mask.shape != logits.shape[:2]:
        raise ValueError("SGEM token mask must match logits batch and length dimensions.")
    if not token_mask.to(dtype=torch.bool).any(dim=1).all():
        raise ValueError("Every SGEM sample must contain at least one scored token.")


def renyi_entropy_loss(
    logits: Tensor,
    token_mask: Tensor,
    *,
    alpha: float = 1.25,
    temperature: float = 2.5,
) -> Tensor:
    """SGEM Eq. (1), averaged over valid autoregressive positions."""
    _validate_sequence_inputs(logits, token_mask)
    if alpha <= 0.0 or abs(alpha - 1.0) < 1e-8:
        raise ValueError("SGEM alpha must be positive and different from one.")
    if temperature <= 0.0:
        raise ValueError("SGEM temperature must be positive.")

    scores = logits.float()
    log_prob = F.log_softmax(scores / float(temperature), dim=-1)
    log_power_sum = torch.logsumexp(float(alpha) * log_prob, dim=-1)
    per_token = log_power_sum / (1.0 - float(alpha))
    valid = token_mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0)


def shannon_entropy_loss(
    logits: Tensor,
    token_mask: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Token-level Shannon entropy used by SLM-TTA.

    The distribution is computed directly from the model logits.  The
    generated sequence supplies only the autoregressive prefixes; it is not a
    target for cross-entropy or any other supervised loss.
    """
    _validate_sequence_inputs(logits, token_mask)
    if temperature <= 0.0:
        raise ValueError("Entropy temperature must be positive.")
    probabilities = torch.softmax(logits.float() / float(temperature), dim=-1)
    per_token = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    valid = token_mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0)


def autoregressive_token_em_loss(
    logits: Tensor,
    generated_tokens: Tensor,
    token_mask: Tensor,
    *,
    objective: str = "token_em",
    temperature: float = 1.0,
    normalization: str = "trajectory",
) -> tuple[Tensor, dict[str, float]]:
    """Compute the complete token-level EM loss for autoregressive models.

    This implements Eqs. (13)-(14) from Huang et al. (2026). With multiple
    generated trajectories, the policy-gradient term uses a leave-one-out
    token-entropy baseline. The paper's Eq. 13 normalizes each trajectory by
    its own valid-token length before averaging trajectories. The legacy
    ``normalization='tokens'`` mode divides the whole beam group by its total
    token count for backwards-compatible ablations. ``token_entropy`` and
    ``token_policy_gradient`` expose the two terms as controlled ablations.
    """
    _validate_sequence_inputs(logits, token_mask)
    if generated_tokens.shape != token_mask.shape:
        raise ValueError(
            "Generated tokens must match the SGEM token mask batch and length dimensions."
        )
    if objective not in {"token_entropy", "token_policy_gradient", "token_em"}:
        raise ValueError(
            "Autoregressive EM objective must be token_entropy, "
            "token_policy_gradient, or token_em."
        )
    if temperature <= 0.0:
        raise ValueError("Autoregressive EM temperature must be positive.")
    if normalization not in {"trajectory", "tokens"}:
        raise ValueError(
            "Autoregressive EM normalization must be trajectory or tokens."
        )

    scores = logits.float() / float(temperature)
    log_probabilities = F.log_softmax(scores, dim=-1)
    probabilities = log_probabilities.exp()
    token_entropies = -(probabilities * log_probabilities).sum(dim=-1)
    valid = token_mask.to(device=logits.device, dtype=token_entropies.dtype)
    lengths = valid.sum(dim=1)
    if not lengths.gt(0).all():
        raise ValueError("Every autoregressive EM trajectory must contain a scored token.")

    sequence_entropies = (token_entropies * valid).sum(dim=1)
    token_log_probabilities = log_probabilities.gather(
        dim=-1,
        index=generated_tokens.to(device=logits.device, dtype=torch.long).unsqueeze(-1),
    ).squeeze(-1)
    sequence_log_probabilities = (token_log_probabilities * valid).sum(dim=1)

    trajectory_count = int(logits.shape[0])
    detached_entropy = sequence_entropies.detach()
    if trajectory_count == 1:
        advantages = detached_entropy
    else:
        leave_one_out = (
            detached_entropy.sum() - detached_entropy
        ) / float(trajectory_count - 1)
        advantages = detached_entropy - leave_one_out

    if normalization == "trajectory":
        # Eq. 13: normalize each complete trajectory by its own valid-token
        # length, then average trajectories so long SMILES do not dominate.
        per_trajectory_policy = (
            advantages * sequence_log_probabilities / lengths.clamp_min(1.0)
        )
        per_trajectory_entropy = sequence_entropies / lengths.clamp_min(1.0)
        policy_gradient = per_trajectory_policy.mean()
        token_entropy = per_trajectory_entropy.mean()
    else:
        denominator = lengths.sum().clamp_min(1.0)
        policy_gradient = (advantages * sequence_log_probabilities).sum() / denominator
        token_entropy = sequence_entropies.sum() / denominator
    if objective == "token_entropy":
        total = token_entropy
    elif objective == "token_policy_gradient":
        total = policy_gradient
    else:
        total = policy_gradient + token_entropy

    return total, {
        "generalized_entropy": float(token_entropy.detach().cpu()),
        "policy_gradient": float(policy_gradient.detach().cpu()),
        "token_entropy": float(token_entropy.detach().cpu()),
        "negative_sampling": 0.0,
        "kl_anchor": 0.0,
    }


def confidence_token_mask(
    logits: Tensor,
    token_mask: Tensor,
    threshold: Optional[float],
) -> Tensor:
    """Apply SLM-TTA's max-probability token filter to valid positions."""
    _validate_sequence_inputs(logits, token_mask)
    if threshold is None:
        return token_mask.to(device=logits.device, dtype=torch.bool)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Confidence threshold must be between zero and one.")
    confidence = torch.softmax(logits.float(), dim=-1).amax(dim=-1)
    return token_mask.to(device=logits.device, dtype=torch.bool) & confidence.ge(
        float(threshold)
    )


def negative_sampling_loss(
    logits: Tensor,
    token_mask: Tensor,
    *,
    temperature: float = 2.5,
    threshold: Optional[float] = None,
    threshold_coeff: float = 0.4,
    excluded_token_ids: Iterable[int] = (),
    allowed_token_mask: Optional[Tensor] = None,
) -> Tensor:
    """SGEM Eq. (2) with optional SMILES grammar/Formula filtering.

    Low-probability classes are treated as negatives only when they are in the
    optional ``allowed_token_mask``.  This prevents a SMILES grammar mask from
    turning every impossible token into an adaptation target.  The mask is
    expected to have shape ``[batch, length, vocabulary]``.
    """
    _validate_sequence_inputs(logits, token_mask)
    if temperature <= 0.0:
        raise ValueError("SGEM temperature must be positive.")
    if threshold is None:
        threshold = float(threshold_coeff) / max(1, int(logits.shape[-1]))
    if threshold < 0.0:
        raise ValueError("SGEM negative threshold must be non-negative.")
    if allowed_token_mask is not None:
        if allowed_token_mask.shape != logits.shape:
            raise ValueError(
                "SGEM allowed-token mask must match logits [batch, length, vocabulary]."
            )
        allowed = allowed_token_mask.to(device=logits.device, dtype=torch.bool)
    else:
        allowed = torch.ones_like(logits, dtype=torch.bool)

    scores = logits.float()
    p_temp = torch.softmax(scores / float(temperature), dim=-1)
    p_unit = torch.softmax(scores, dim=-1)
    negative_mask = (p_unit < float(threshold)) & allowed
    for token_id in excluded_token_ids:
        token_id = int(token_id)
        if 0 <= token_id < negative_mask.shape[-1]:
            negative_mask[..., token_id] = False
    negative_mass = (p_temp * negative_mask.to(dtype=p_temp.dtype)).sum(dim=-1)
    per_token = -torch.log1p(-negative_mass.clamp(max=1.0 - 1e-6))
    valid = token_mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0)


def kl_anchor_loss(
    logits: Tensor,
    anchor_logits: Tensor,
    token_mask: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Token-level KL(P_0 || P_theta) to the pre-update distribution.

    This is the direction used by the official SGEM implementation:
    F.kl_div(log P_theta, P_0). The frozen distribution is detached so the
    anchor cannot move during adaptation.
    """
    _validate_sequence_inputs(logits, token_mask)
    if anchor_logits.shape != logits.shape:
        raise ValueError("SGEM anchor logits must have the same shape as logits.")
    if temperature <= 0.0:
        raise ValueError("KL temperature must be positive.")
    probabilities = torch.softmax(
        anchor_logits.float() / float(temperature), dim=-1
    ).detach()
    log_current = F.log_softmax(logits.float() / float(temperature), dim=-1)
    per_token = F.kl_div(log_current, probabilities, reduction="none").sum(dim=-1)
    valid = token_mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0)


def sgem_loss(
    logits: Tensor,
    token_mask: Tensor,
    *,
    alpha: float = 1.25,
    temperature: float = 2.5,
    negative_coeff: float = 0.25,
    negative_threshold: Optional[float] = None,
    negative_threshold_coeff: float = 0.4,
    excluded_token_ids: Iterable[int] = (),
    allowed_token_mask: Optional[Tensor] = None,
    anchor_logits: Optional[Tensor] = None,
    kl_weight: float = 0.1,
    entropy_objective: str = "renyi",
    confidence_threshold: Optional[float] = None,
) -> tuple[Tensor, dict[str, float]]:
    """Calculate entropy adaptation plus optional legacy SGEM terms."""
    if negative_coeff < 0.0 or kl_weight < 0.0:
        raise ValueError("SGEM loss weights must be non-negative.")
    effective_mask = confidence_token_mask(logits, token_mask, confidence_threshold)
    valid_rows = effective_mask.to(dtype=torch.bool).any(dim=1)
    if not valid_rows.any():
        zero = logits.float().sum() * 0.0
        return zero, {
            "generalized_entropy": 0.0,
            "negative_sampling": 0.0,
            "kl_anchor": 0.0,
        }
    # The primitive losses require at least one scored token per row. A
    # confidence threshold can remove all tokens from individual samples, so
    # exclude those rows while retaining gradients for the remaining ones.
    loss_logits = logits[valid_rows]
    loss_mask = effective_mask[valid_rows]
    loss_allowed = allowed_token_mask[valid_rows] if allowed_token_mask is not None else None
    loss_anchor = anchor_logits[valid_rows] if anchor_logits is not None else None
    if entropy_objective == "shannon":
        generalized = shannon_entropy_loss(
            loss_logits,
            loss_mask,
            temperature=temperature,
        )
    elif entropy_objective == "renyi":
        generalized = renyi_entropy_loss(
            loss_logits,
            loss_mask,
            alpha=alpha,
            temperature=temperature,
        )
    else:
        raise ValueError("SGEM entropy objective must be renyi or shannon.")
    negative = negative_sampling_loss(
        loss_logits,
        loss_mask,
        temperature=temperature,
        threshold=negative_threshold,
        threshold_coeff=negative_threshold_coeff,
        excluded_token_ids=excluded_token_ids,
        allowed_token_mask=loss_allowed,
    )
    if loss_anchor is None or kl_weight == 0.0:
        anchor = generalized.new_zeros(())
    else:
        anchor = kl_anchor_loss(loss_logits, loss_anchor, loss_mask)
    total = generalized + float(negative_coeff) * negative + float(kl_weight) * anchor
    return total, {
        "generalized_entropy": float(generalized.detach().cpu()),
        "negative_sampling": float(negative.detach().cpu()),
        "kl_anchor": float(anchor.detach().cpu()),
    }


def _module_by_path(model: nn.Module, paths: Sequence[str]) -> list[nn.Module]:
    modules: list[nn.Module] = []
    seen: set[int] = set()
    for path in paths:
        current: Any = model
        for part in path.split("."):
            if not hasattr(current, part):
                current = None
                break
            current = getattr(current, part)
        if isinstance(current, nn.Module) and id(current) not in seen:
            modules.append(current)
            seen.add(id(current))
    return modules


def _layernorm_modules_for_scope(model: nn.Module, scope: str) -> list[nn.Module]:
    """Select active LayerNorm modules without touching decoder parameters."""
    modules: list[nn.Module] = []
    seen: set[int] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm) or id(module) in seen:
            continue
        lowered = name.lower()
        selected = False
        if scope == "read_layernorm":
            selected = lowered in {
                "generation_fusion.read_fusion.attention_norm",
                "generation_fusion.read_fusion.ffn_norm",
            }
        elif scope == "modality_layernorm":
            selected = lowered.startswith(
                (
                    "multimodal_embedding.embedding_norm_dict.",
                    "hf_model.embedding.embedding_norm_dict.",
                )
            ) and not lowered.endswith(".smiles")
        elif scope == "encoder_layernorm":
            encoder_prefix = (
                "hf_model.encoder.layers.",
                "hf_model.model.encoder.layers.",
            )
            selected = (
                lowered.startswith(encoder_prefix)
                and lowered.endswith(
                    (
                        "self_attn_layer_norm",
                        "final_layer_norm",
                        "norm1",
                        "norm2",
                    )
                )
            ) or lowered in {"hf_model.encoder.norm", "hf_model.model.encoder.norm"}
        elif scope == "all_layernorm":
            is_encoder_side = (
                lowered.startswith(
                    (
                        "generation_fusion.",
                        "multimodal_embedding.",
                        "hf_model.embedding.",
                        "hf_model.encoder.",
                        "hf_model.model.encoder.",
                    )
                )
                and "decoder" not in lowered
                and not lowered.endswith(".smiles")
            )
            selected = is_encoder_side
        elif scope == "model_layernorm":
            # Huang et al. (2026) update every Whisper LayerNorm, including
            # decoder norms. Restrict this to the seq2seq backbone: multimodal
            # input and READ-fusion norms do not exist in the paper's model.
            # Keep the scope separate from the historical encoder-side
            # ``all_layernorm`` so old runs remain stable.
            selected = lowered.startswith(
                (
                    "hf_model.encoder.",
                    "hf_model.decoder.",
                    "hf_model.model.encoder.",
                    "hf_model.model.decoder.",
                )
            )
        if selected:
            modules.append(module)
            seen.add(id(module))
    return modules


def _enable_tta_modules(
    model: nn.Module,
    selected_ids: Optional[set[int]] = None,
) -> None:
    """Enable selected modules that expose an explicit TTA switch."""
    for module in model.modules():
        enable = getattr(module, "enable_adaptation", None)
        parameters_for_tta = getattr(module, "parameters_for_tta", None)
        if callable(enable) and callable(parameters_for_tta):
            if selected_ids is not None:
                module_ids = {id(parameter) for parameter in parameters_for_tta()}
                if not module_ids.intersection(selected_ids):
                    continue
            enable(True)


def configure_sgem_parameters(
    model: nn.Module,
    scope: str = "adapter",
) -> list[tuple[str, nn.Parameter]]:
    """Freeze the model and expose an explicit encoder-side parameter subset.

    ``adapter`` selects explicit ``qkv_parameters``/``parameters_for_tta``
    modules, or existing modules named ``adapter``/``vittt``. ``encoder``
    matches SGEM's paper setting. The LayerNorm scopes isolate the active READ
    fusion norms, modality embedding norms, or backbone encoder norms.
    ``all_layernorm`` combines those encoder-side normalization layers to
    match SLM-TTA's parameter choice. ``model_layernorm`` selects encoder and
    decoder LayerNorms as in Huang et al. (2026). The broader ``encoder_side``
    scope additionally includes multimodal embedding and generation fusion
    parameters for controlled ablations.
    """
    normalized = str(scope).strip().lower()
    valid_scopes = {
        "adapter",
        "encoder",
        "encoder_side",
        "read_layernorm",
        "modality_layernorm",
        "encoder_layernorm",
        "all_layernorm",
        "model_layernorm",
    }
    if normalized not in valid_scopes:
        raise ValueError(
            "SGEM scope must be adapter, encoder, encoder_side, read_layernorm, "
            "modality_layernorm, encoder_layernorm, all_layernorm, or "
            "model_layernorm."
        )

    model.eval()
    model.requires_grad_(False)
    selected_modules: list[nn.Module] = []
    if normalized == "encoder":
        selected_modules = _module_by_path(model, ("hf_model.encoder", "hf_model.model.encoder"))
    elif normalized == "encoder_side":
        selected_modules = _module_by_path(
            model,
            (
                "multimodal_embedding",
                "hf_model.encoder",
                "hf_model.model.encoder",
                "generation_fusion",
            ),
        )
    elif normalized in {
        "read_layernorm",
        "modality_layernorm",
        "encoder_layernorm",
        "all_layernorm",
        "model_layernorm",
    }:
        selected_modules = _layernorm_modules_for_scope(model, normalized)
    else:
        # Prefer modules that explicitly expose their safe TTA parameter set.
        # For Model C this resolves to READ Q/K/V rather than the whole fusion
        # block. Generic adapter/vittt names are accepted as a fallback.
        explicit_parameters: list[nn.Parameter] = []
        for name, module in model.named_modules():
            lowered = name.lower()
            for method_name in ("qkv_parameters", "parameters_for_tta"):
                method = getattr(module, method_name, None)
                if callable(method):
                    try:
                        explicit_parameters.extend(list(method()))
                    except (RuntimeError, ValueError):
                        pass
            if name and ("adapter" in lowered or "vittt" in lowered):
                selected_modules.append(module)
        explicit_ids = {id(parameter) for parameter in explicit_parameters}
        selected_ids = explicit_ids | {
            id(parameter)
            for module in selected_modules
            for parameter in module.parameters()
        }
        selected = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if id(parameter) in selected_ids
        ]
        if not selected:
            raise RuntimeError(
                "SGEM adapter scope found no explicit TTA adapter/QKV parameters; "
                "use a LayerNorm scope, encoder, or encoder_side for a deliberate "
                "broader ablation.",
            )
        _enable_tta_modules(model, selected_ids)
        for _, parameter in selected:
            parameter.requires_grad_(True)
        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in selected_ids
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected SGEM-trainable parameters: {unexpected[:10]}")
        return selected

    selected_ids: set[int] = set()
    for module in selected_modules:
        selected_ids.update(id(parameter) for parameter in module.parameters())
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) in selected_ids
    ]
    if not selected:
        raise RuntimeError(f"SGEM found no parameters for scope {normalized!r}.")
    for _, parameter in selected:
        parameter.requires_grad_(True)

    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in selected_ids
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected SGEM-trainable parameters: {unexpected[:10]}")
    return selected


def _optimizer(
    parameters: Sequence[nn.Parameter],
    *,
    name: str,
    lr: float,
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=0.0)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    raise ValueError(f"Unknown SGEM optimizer: {name!r}")


def _first_beam(sequences: Tensor, *, batch_size: int, beams: int) -> Tensor:
    if sequences.ndim != 2:
        raise ValueError("Generated sequences must have shape [batch*beams, length].")
    expected = batch_size * beams
    if sequences.shape[0] != expected:
        raise ValueError(f"Expected {expected} generated sequences, got {sequences.shape[0]}.")
    return sequences.reshape(batch_size, beams, -1)[:, 0, :].contiguous()


def _select_adaptation_sequences(
    pseudo_all: Tensor,
    *,
    batch_size: int,
    beams: int,
    adapt_all_beams: bool,
    filter_fn: Optional[AdaptSequenceFilterFn],
) -> tuple[Tensor, Tensor, bool]:
    """Select the reported pseudo path and the paths used by the TTA loss.

    A candidate filter must inspect the complete beam pool. This matters for
    observable constraints such as molecular Formula: beam 1 can fail the
    constraint while a later beam is usable. Filtering a single beam first
    would silently turn ``adapt_beams`` into an ineffective setting.
    """
    pseudo = _first_beam(pseudo_all, batch_size=batch_size, beams=beams)
    candidates = pseudo_all
    if filter_fn is not None:
        if batch_size != 1:
            raise ValueError(
                "Adaptation sequence filtering currently requires batch size 1."
            )
        filtered = filter_fn(pseudo_all)
        if filtered is None or filtered.shape[0] == 0:
            return pseudo, pseudo, True
        candidates = filtered
        pseudo = candidates[:1].contiguous()
    adapt_sequences = candidates if adapt_all_beams else pseudo
    return pseudo, adapt_sequences, False


def _rerank_beams(
    logits: Tensor,
    next_tokens: Tensor,
    token_mask: Tensor,
    sequences: Tensor,
    *,
    batch_size: int,
    beams: int,
    mode: str,
) -> Tensor:
    """Move the best unlabeled candidate to beam-1, preserving the pool.

    ``generate`` already returns candidates in decoder-score order. This helper
    deliberately uses only teacher-forced model probabilities on each complete
    candidate, so it can test whether SGEM's objective provides a useful
    candidate-level signal without reading target SMILES.
    """
    if mode == "none":
        return sequences
    if mode == "formula":
        raise ValueError(
            "Formula beam reranking requires a beam_rerank_fn callback "
            "that supplies the observable Formula input."
        )
    if mode not in {"nll", "entropy", "confidence"}:
        raise ValueError(f"Unknown beam rerank mode: {mode!r}")
    expected = batch_size * beams
    if sequences.ndim != 2 or sequences.shape[0] != expected:
        raise ValueError(
            f"Expected [batch*beams, length] with {expected} rows, got {tuple(sequences.shape)}."
        )
    if logits.shape[:2] != next_tokens.shape or next_tokens.shape != token_mask.shape:
        raise ValueError("Beam scoring tensors have incompatible shapes.")
    if logits.shape[0] != expected:
        raise ValueError("Beam logits do not match the requested beam count.")

    scores = logits.float()
    probabilities = torch.softmax(scores, dim=-1)
    safe_tokens = next_tokens.to(device=logits.device, dtype=torch.long)
    token_prob = probabilities.gather(-1, safe_tokens.unsqueeze(-1)).squeeze(-1)
    valid = token_mask.to(device=logits.device, dtype=torch.bool)
    lengths = valid.sum(dim=1).clamp_min(1).to(dtype=scores.dtype)
    if mode == "nll":
        candidate_scores = token_prob.clamp_min(1e-12).log().mul(valid).sum(dim=1) / lengths
        order = torch.argsort(
            candidate_scores.reshape(batch_size, beams),
            dim=1,
            descending=True,
            stable=True,
        )
    elif mode == "confidence":
        candidate_scores = probabilities.amax(dim=-1).mul(valid).sum(dim=1) / lengths
        order = torch.argsort(
            candidate_scores.reshape(batch_size, beams),
            dim=1,
            descending=True,
            stable=True,
        )
    else:
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        candidate_scores = entropy.mul(valid).sum(dim=1) / lengths
        order = torch.argsort(
            candidate_scores.reshape(batch_size, beams),
            dim=1,
            descending=False,
            stable=True,
        )

    view = sequences.reshape(batch_size, beams, -1)
    row_indices = torch.arange(batch_size, device=sequences.device).unsqueeze(1)
    return view[row_indices, order].reshape(expected, -1).contiguous()


def rerank_generated_beams(
    model: nn.Module,
    observed: Mapping[str, Any],
    sequences: Tensor,
    *,
    beams: int,
    mode: str,
    generation_state: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    """Rerank a generated beam pool using model-only candidate scores."""
    if mode == "none":
        return sequences
    if mode == "formula":
        raise ValueError(
            "Formula beam reranking requires a beam_rerank_fn callback "
            "that supplies the observable Formula input."
        )
    if generation_state is None:
        generation_state = model.prepare_generation_state(observed)
    logits, next_tokens, token_mask = model.score_generated_sequences(
        observed,
        sequences,
        generation_state=generation_state,
    )
    batch_size = int(sequences.shape[0] // beams)
    return _rerank_beams(
        logits,
        next_tokens,
        token_mask,
        sequences,
        batch_size=batch_size,
        beams=beams,
        mode=mode,
    )


def _observed_batch_size(observed: Mapping[str, Any]) -> Optional[int]:
    """Infer batch size from collator metadata, without consulting labels.

    The project collator keeps modality tensors sequence-first, so an input
    tensor's batch dimension is not reliably ``shape[0]``.  Availability
    vectors are batch-first and are therefore the preferred source.  The
    modality padding masks are the next best source and follow the same
    sequence-first convention as the rest of the collator output.
    """
    availability = observed.get("encoder_modality_availability")
    if isinstance(availability, Mapping):
        sizes = {
            int(value.reshape(-1).numel())
            for value in availability.values()
            if isinstance(value, Tensor)
        }
        if len(sizes) > 1:
            raise ValueError(
                "Observed modality availability vectors disagree on batch size."
            )
        if sizes:
            return sizes.pop()

    masks = observed.get("encoder_modality_pad_masks")
    if isinstance(masks, Mapping):
        sizes = {
            int(value.shape[1])
            for value in masks.values()
            if isinstance(value, Tensor) and value.ndim >= 2
        }
        if len(sizes) > 1:
            raise ValueError("Observed modality padding masks disagree on batch size.")
        if sizes:
            return sizes.pop()
    return None


def _parameter_delta(parameters: Sequence[nn.Parameter], reference: Sequence[Tensor]) -> float:
    with torch.no_grad():
        squared = sum(
            (parameter.detach().float() - initial.detach().float()).square().sum()
            for parameter, initial in zip(parameters, reference)
        )
    return float(squared.sqrt().cpu())


def _project_parameter_delta_(
    parameters: Sequence[nn.Parameter],
    reference: Sequence[Tensor],
    max_delta: float,
) -> float:
    delta = _parameter_delta(parameters, reference)
    if max_delta > 0.0 and delta > max_delta:
        scale = max_delta / max(delta, 1e-12)
        with torch.no_grad():
            for parameter, initial in zip(parameters, reference):
                parameter.copy_(initial + (parameter - initial) * scale)
        return float(max_delta)
    return delta


@contextmanager
def restored_sgem_parameters(
    parameters: Iterable[nn.Parameter],
) -> Iterator[None]:
    """Restore selected parameters after one episodic target sample."""
    parameters = list(parameters)
    saved = [parameter.detach().clone() for parameter in parameters]
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in zip(parameters, saved):
                parameter.copy_(value)


def _special_token_ids(model: nn.Module) -> tuple[int, ...]:
    tokenizer = getattr(model, "target_tokenizer", None)
    values = {
        getattr(tokenizer, name, None)
        for name in ("pad_token_id", "bos_token_id", "eos_token_id")
    }
    return tuple(sorted(int(value) for value in values if value is not None))


def _score_reference_logits(
    model: nn.Module,
    observed: Mapping[str, Any],
    pseudo: Tensor,
    parameters: Sequence[nn.Parameter],
    reference: Sequence[Tensor],
) -> Tensor:
    """Score a refreshed pseudo path with the frozen parameters.

    The official implementation keeps a second frozen model for its KL term.
    We avoid a second full model copy by temporarily swapping only the selected
    parameters, then restoring the adapted values immediately.
    """
    with torch.no_grad():
        current = [parameter.detach().clone() for parameter in parameters]
        try:
            for parameter, initial in zip(parameters, reference):
                parameter.copy_(initial)
            anchor_logits, _, _ = model.score_generated_sequences(observed, pseudo)
            return anchor_logits.detach()
        finally:
            for parameter, value in zip(parameters, current):
                parameter.copy_(value)


def adapt_sgem_batch(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    config: SGEMConfig = SGEMConfig(),
    batch_size: Optional[int] = None,
    logits_processor: Optional[Any] = None,
    adapt_logits_processor: Optional[Any] = None,
    final_logits_processor: Optional[Any] = None,
    parameters: Optional[list[tuple[str, nn.Parameter]]] = None,
    allowed_token_mask_builder: Optional[AllowedTokenMaskBuilder] = None,
    acceptance_fn: Optional[AcceptanceFn] = None,
    beam_rerank_fn: Optional[BeamRerankFn] = None,
    adapt_sequence_filter_fn: Optional[AdaptSequenceFilterFn] = None,
) -> SGEMTTAResult:
    """Adapt one unlabeled batch and return predictions before/after adaptation.

    The paper's single-instance protocol is enforced when ``episodic=True``.
    ``acceptance_fn`` receives ``(predictions_before, predictions_after)`` and
    can reject an update after external RDKit/Formula/multimodal checks. The
    callback is never given labels by this module.

    ``logits_processor`` is retained for callers that use the same processor
    for both passes. When adaptation and final beam widths differ, callers
    should provide separate processors through ``adapt_logits_processor`` and
    ``final_logits_processor`` because Formula processors encode beam count.
    """
    config.validate()
    if config.beam_rerank == "formula" and beam_rerank_fn is None:
        raise ValueError(
            "SGEM beam_rerank='formula' requires a beam_rerank_fn callback "
            "that supplies the observable Formula input."
        )
    observed = observed_input_batch(batch)
    if "encoder_input" not in observed:
        raise KeyError("SGEM requires encoder_input.")
    observed_size = _observed_batch_size(observed)
    if batch_size is None:
        actual_batch_size = int(observed_size) if observed_size is not None else 1
    else:
        actual_batch_size = int(batch_size)
    if actual_batch_size < 1:
        raise ValueError("SGEM batch size must be positive.")
    if observed_size is not None and observed_size != actual_batch_size:
        raise ValueError(
            "Explicit SGEM batch_size does not match the observed input batch length."
        )
    if config.episodic and actual_batch_size != 1:
        raise ValueError("Episodic SGEM requires batch size 1.")

    selected = parameters or configure_sgem_parameters(model, config.parameter_scope)
    _enable_tta_modules(model, {id(parameter) for _, parameter in selected})
    selected_parameters = [parameter for _, parameter in selected]
    reference = [parameter.detach().clone() for parameter in selected_parameters]
    adapt_processor = logits_processor if adapt_logits_processor is None else adapt_logits_processor
    final_processor = logits_processor if final_logits_processor is None else final_logits_processor

    def apply_beam_rerank(
        sequences: Tensor,
        beams: int,
        rerank_state: Optional[Mapping[str, Any]],
    ) -> Tensor:
        if beam_rerank_fn is not None:
            return beam_rerank_fn(model, observed, sequences, beams)
        return rerank_generated_beams(
            model,
            observed,
            sequences,
            beams=beams,
            mode=config.beam_rerank,
            generation_state=rerank_state,
        )

    model.eval()
    try:
        with torch.inference_mode(False), torch.enable_grad():
            # The HF wrapper can retain one encoder graph for the initial
            # before/pseudo/teacher-forced passes.  Lightweight test doubles
            # and older wrappers simply fall back to their original methods.
            prepare_state = getattr(model, "prepare_generation_state", None)
            generation_state = prepare_state(observed) if callable(prepare_state) else None

            def generate_initial(
                *,
                beams: int,
                processor: Optional[Any],
            ) -> Tensor:
                # HF beam search expands and may mutate encoder-output
                # containers.  Keep the prepared state exclusively for the
                # differentiable scoring pass; each generation call prepares
                # an isolated encoder state so its candidate order matches the
                # ordinary frozen path.
                return model.generate(
                    observed,
                    n_beams=beams,
                    logits_processor=processor,
                )

            def score_initial(sequence: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                if generation_state is not None:
                    return model.score_generated_sequences(
                        observed,
                        sequence,
                        generation_state=generation_state,
                    )
                return model.score_generated_sequences(observed, sequence)

            with torch.no_grad():
                predictions_before = generate_initial(
                    beams=config.final_beams,
                    processor=final_processor,
                )
                pseudo_all = generate_initial(
                    beams=config.adapt_beams,
                    processor=adapt_processor,
                )
                predictions_before = apply_beam_rerank(
                    predictions_before,
                    config.final_beams,
                    generation_state,
                )
                pseudo_all = apply_beam_rerank(
                    pseudo_all,
                    config.adapt_beams,
                    generation_state,
                )
                pseudo, adapt_sequences, filter_empty = _select_adaptation_sequences(
                    pseudo_all,
                    batch_size=actual_batch_size,
                    beams=config.adapt_beams,
                    adapt_all_beams=config.adapt_all_beams,
                    filter_fn=adapt_sequence_filter_fn,
                )
                # Keep decoder beam order for the prediction path, but allow
                # SGEM to use every candidate as an unlabeled fixed path. This
                # avoids selecting a model-confident yet chemically wrong beam
                # solely because it has lower self-entropy.
                anchor_logits, generated_tokens, token_mask = score_initial(
                    adapt_sequences
                )
                anchor_logits = anchor_logits.detach()

            optimizer = _optimizer(
                selected_parameters,
                name=config.optimizer,
                lr=config.lr,
            )
            scheduler = None
            if config.lr_final is not None:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, config.steps),
                    eta_min=config.lr_final,
                )

            diagnostics = {
                "generalized_entropy": 0.0,
                "policy_gradient": 0.0,
                "token_entropy": 0.0,
                "negative_sampling": 0.0,
                "kl_anchor": 0.0,
            }
            loss_value = 0.0
            updated = False
            skip_reason = None
            adaptation_sequence_count = 0
            gradient_norm = 0.0
            max_gradient_norm = 0.0
            for step_index in range(config.steps):
                if filter_empty:
                    skip_reason = "adapt_sequence_filter_empty"
                    break
                if step_index > 0:
                    if config.refresh_pseudo_each_step:
                        with torch.no_grad():
                            pseudo_all = model.generate(
                                observed,
                                n_beams=config.adapt_beams,
                                logits_processor=adapt_processor,
                            )
                            # Model-score rerankers cannot reuse the state whose
                            # graph was consumed by the preceding update.
                            if config.beam_rerank in {
                                "nll",
                                "entropy",
                                "confidence",
                            }:
                                generation_state = (
                                    prepare_state(observed)
                                    if callable(prepare_state)
                                    else None
                                )
                            pseudo_all = apply_beam_rerank(
                                pseudo_all,
                                config.adapt_beams,
                                generation_state,
                            )
                            pseudo, adapt_sequences, filter_empty = (
                                _select_adaptation_sequences(
                                    pseudo_all,
                                    batch_size=actual_batch_size,
                                    beams=config.adapt_beams,
                                    adapt_all_beams=config.adapt_all_beams,
                                    filter_fn=adapt_sequence_filter_fn,
                                )
                            )
                        if filter_empty:
                            skip_reason = "adapt_sequence_filter_empty"
                            break
                        if config.adaptation_objective == "legacy_sgem":
                            anchor_logits = _score_reference_logits(
                                model,
                                observed,
                                adapt_sequences,
                                selected_parameters,
                                reference,
                            )
                    # Build the differentiable state only after all temporary
                    # parameter swaps. Otherwise autograd observes an in-place
                    # version change when the selected parameters are restored.
                    generation_state = (
                        prepare_state(observed) if callable(prepare_state) else None
                    )
                optimizer.zero_grad(set_to_none=True)
                adaptation_sequence_count = int(adapt_sequences.shape[0])
                if generation_state is not None:
                    logits, current_tokens, current_mask = model.score_generated_sequences(
                        observed,
                        adapt_sequences,
                        generation_state=generation_state,
                    )
                else:
                    logits, current_tokens, current_mask = model.score_generated_sequences(
                        observed, adapt_sequences
                    )
                if (
                    not config.refresh_pseudo_each_step
                    and (
                        current_tokens.shape != generated_tokens.shape
                        or not torch.equal(
                            current_tokens.detach(), generated_tokens.detach()
                        )
                    )
                ):
                    raise RuntimeError(
                        "SGEM model changed the fixed pseudo sequence unexpectedly."
                    )
                generated_tokens = current_tokens.detach()
                allowed_token_mask = None
                if allowed_token_mask_builder is not None:
                    allowed_token_mask = allowed_token_mask_builder(current_tokens, logits)
                effective_mask = confidence_token_mask(
                    logits,
                    current_mask,
                    config.confidence_threshold,
                )
                if not effective_mask.any():
                    skip_reason = "confidence_filter_empty"
                    break
                if config.adaptation_objective == "legacy_sgem":
                    loss, diagnostics = sgem_loss(
                        logits,
                        current_mask,
                        alpha=config.alpha,
                        temperature=config.temperature,
                        negative_coeff=config.negative_coeff,
                        negative_threshold_coeff=config.negative_threshold_coeff,
                        excluded_token_ids=_special_token_ids(model),
                        allowed_token_mask=allowed_token_mask,
                        anchor_logits=anchor_logits,
                        kl_weight=config.kl_weight,
                        entropy_objective=config.entropy_objective,
                        confidence_threshold=config.confidence_threshold,
                    )
                    diagnostics = {
                        **diagnostics,
                        "policy_gradient": 0.0,
                        "token_entropy": diagnostics["generalized_entropy"],
                    }
                else:
                    # The new autoregressive EM ablations intentionally exclude
                    # SGEM negative sampling and KL so each objective is isolated.
                    valid_rows = effective_mask.any(dim=1)
                    loss, diagnostics = autoregressive_token_em_loss(
                        logits[valid_rows],
                        current_tokens[valid_rows],
                        effective_mask[valid_rows],
                        objective=config.adaptation_objective,
                        temperature=config.temperature,
                        normalization=config.token_em_normalization,
                    )
                if not torch.isfinite(loss):
                    skip_reason = "non_finite_loss"
                    break
                loss.backward()
                if config.max_grad_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        selected_parameters, config.max_grad_norm
                    )
                    gradient_norm = float(total_norm.detach().cpu())
                else:
                    parameter_grad_norms = [
                        parameter.grad.detach().float().norm(2)
                        for parameter in selected_parameters
                        if parameter.grad is not None
                    ]
                    gradient_norm = (
                        float(torch.stack(parameter_grad_norms).norm(2).cpu())
                        if parameter_grad_norms
                        else 0.0
                    )
                max_gradient_norm = max(max_gradient_norm, gradient_norm)
                optimizer.step()
                updated = True
                if scheduler is not None:
                    scheduler.step()
                loss_value = float(loss.detach().cpu())

            parameter_delta = _project_parameter_delta_(
                selected_parameters,
                reference,
                config.max_parameter_delta,
            )
            with torch.no_grad():
                final_rerank_state = (
                    prepare_state(observed)
                    if callable(prepare_state)
                    and config.beam_rerank in {"nll", "entropy", "confidence"}
                    else None
                )
                predictions_after = model.generate(
                    observed,
                    n_beams=config.final_beams,
                    logits_processor=final_processor,
                )
                predictions_after = apply_beam_rerank(
                    predictions_after,
                    config.final_beams,
                    final_rerank_state,
                )

            accepted = (
                acceptance_fn(predictions_before, predictions_after)
                if acceptance_fn
                else True
            )
            if not accepted:
                with torch.no_grad():
                    for parameter, initial in zip(selected_parameters, reference):
                        parameter.copy_(initial)
                predictions_after = predictions_before.detach().clone()

        result = SGEMTTAResult(
            predictions_before=predictions_before.detach(),
            predictions_after=predictions_after.detach(),
            pseudo_sequence=pseudo.detach(),
            updated=updated,
            accepted=bool(accepted),
            loss=loss_value,
            generalized_entropy=diagnostics["generalized_entropy"],
            policy_gradient=diagnostics["policy_gradient"],
            token_entropy=diagnostics["token_entropy"],
            negative_sampling=diagnostics["negative_sampling"],
            kl_anchor=diagnostics["kl_anchor"],
            adaptation_sequence_count=adaptation_sequence_count,
            gradient_norm=gradient_norm,
            max_gradient_norm=max_gradient_norm,
            parameter_delta=parameter_delta if accepted and updated else 0.0,
            skip_reason=(
                skip_reason
                or (None if accepted else "acceptance_fn_rejected_update")
            ),
        )
        return result
    finally:
        if config.episodic:
            with torch.no_grad():
                for parameter, initial in zip(selected_parameters, reference):
                    parameter.copy_(initial)


__all__ = [
    "SGEMConfig",
    "SGEMTTAResult",
    "adapt_sgem_batch",
    "autoregressive_token_em_loss",
    "configure_sgem_parameters",
    "kl_anchor_loss",
    "negative_sampling_loss",
    "renyi_entropy_loss",
    "rerank_generated_beams",
    "shannon_entropy_loss",
    "confidence_token_mask",
    "restored_sgem_parameters",
    "sgem_loss",
]

"""READ test-time adaptation for multimodal spectra-to-SMILES generation.

The ICLR 2024 READ method updates only Q/K/V in the final multimodal attention
layer. Classification confidence is replaced here by the length-normalized
geometric mean of per-token confidence on an internally generated pseudo-SMILES.
No target SMILES or structure-derived label is accepted by the adaptation path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import nn


_OBSERVED_BATCH_KEYS = (
    "encoder_input",
    "encoder_pad_mask",
    "encoder_modality_pad_masks",
    "encoder_modality_availability",
)


@dataclass(frozen=True)
class READTTAResult:
    """Before/after predictions and scalar diagnostics for one READ update."""

    predictions_before: torch.Tensor
    predictions_after: torch.Tensor
    loss: float
    confidence_loss: float
    balance_entropy: float
    mean_sequence_confidence: float


def observed_input_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only fields that can be observed at test time."""
    observed = {key: batch[key] for key in _OBSERVED_BATCH_KEYS if key in batch}
    if "encoder_input" not in observed:
        raise KeyError("READ TTA requires encoder_input.")
    return observed


def read_confidence_objective(
    confidence: torch.Tensor,
    *,
    gamma: float = math.exp(-1.0),
) -> torch.Tensor:
    """Equation 6 from READ: p * log(e * gamma / p)."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("READ confidence threshold gamma must lie in (0, 1).")
    eps = torch.finfo(confidence.dtype).eps
    confidence = confidence.clamp(min=eps, max=1.0)
    return confidence * (1.0 - confidence.log() + math.log(gamma))


def _sequence_confidence(
    probabilities: torch.Tensor,
    generated_tokens: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Length-normalized confidence for each generated SMILES sequence."""
    if generated_tokens.shape != probabilities.shape[:2]:
        raise ValueError("Generated tokens must match probability batch and length.")
    token_confidence = probabilities.gather(
        dim=-1,
        index=generated_tokens.to(device=probabilities.device).unsqueeze(-1),
    ).squeeze(-1)
    mask = token_mask.to(device=probabilities.device, dtype=probabilities.dtype)
    counts = mask.sum(dim=1).clamp_min(1.0)
    eps = torch.finfo(probabilities.dtype).eps
    mean_log_confidence = (
        token_confidence.clamp_min(eps).log() * mask
    ).sum(dim=1) / counts
    return mean_log_confidence.exp()


def read_sequence_loss(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    gamma: float = math.exp(-1.0),
    balance_weight: float = 0.0,
    excluded_token_ids: Iterable[int] = (),
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Task-adapted READ loss on pseudo-SMILES decoder distributions.

    READ's class-balance entropy has no direct sequence analogue: SMILES tokens
    are intentionally non-uniform. It is available only as an explicit paper
    control and is disabled by default.
    """
    if logits.ndim != 3:
        raise ValueError("READ sequence logits must have shape [batch, length, vocab].")
    if token_mask.shape != logits.shape[:2]:
        raise ValueError("READ token mask must match logits batch and length dimensions.")
    if generated_tokens.shape != logits.shape[:2]:
        raise ValueError("READ generated tokens must match logits batch and length dimensions.")
    if balance_weight < 0:
        raise ValueError("READ balance_weight must be non-negative.")
    if not token_mask.to(dtype=torch.bool).any(dim=1).all():
        raise ValueError("Every READ sample must contain at least one scored token.")

    probabilities = torch.softmax(logits.float(), dim=-1)
    token_mask = token_mask.to(device=logits.device, dtype=torch.bool)
    sequence_confidence = _sequence_confidence(
        probabilities,
        generated_tokens,
        token_mask,
    )
    confidence_loss = read_confidence_objective(
        sequence_confidence, gamma=gamma
    ).mean()

    valid_weights = token_mask.to(dtype=probabilities.dtype).unsqueeze(-1)
    token_mass = (probabilities * valid_weights).sum(dim=(0, 1))
    vocabulary_mask = torch.ones_like(token_mass)
    for token_id in excluded_token_ids:
        if 0 <= int(token_id) < token_mass.numel():
            vocabulary_mask[int(token_id)] = 0.0
    token_mass = token_mass * vocabulary_mask
    marginal = token_mass / token_mass.sum().clamp_min(
        torch.finfo(token_mass.dtype).eps
    )
    positive = marginal > 0
    balance_entropy = -(marginal[positive] * marginal[positive].log()).sum()
    loss = confidence_loss - float(balance_weight) * balance_entropy
    return loss, {
        "confidence_loss": confidence_loss,
        "balance_entropy": balance_entropy,
        "mean_sequence_confidence": sequence_confidence.mean(),
    }


def _read_generation_fusion(model: nn.Module) -> nn.Module:
    fusion = getattr(model, "generation_fusion", None)
    if fusion is None or not bool(getattr(fusion, "uses_encoder_token_fusion", False)):
        raise ValueError("READ TTA requires model C with generation fusion variant 'read_saf'.")
    return fusion


def configure_read_tta_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Freeze the model and expose only final READ attention Q/K/V."""
    fusion = _read_generation_fusion(model)
    model.eval()
    model.requires_grad_(False)
    parameters = list(fusion.read_qkv_parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    if not parameters:
        raise ValueError("READ fusion exposed no Q/K/V parameters.")
    non_fp32 = [parameter.dtype for parameter in parameters if parameter.dtype != torch.float32]
    if non_fp32:
        raise TypeError(
            "READ Q/K/V parameters must remain FP32 master weights; "
            f"found {sorted({str(dtype) for dtype in non_fp32})}."
        )
    allowed = {id(parameter) for parameter in parameters}
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in allowed
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected READ-trainable parameters: {unexpected}")
    return parameters


def build_read_optimizer(
    model: nn.Module,
    *,
    lr: float = 1e-4,
) -> tuple[torch.optim.Optimizer, list[nn.Parameter]]:
    """Build the paper's Adam optimizer for online READ adaptation."""
    parameters = configure_read_tta_parameters(model)
    optimizer = torch.optim.Adam(
        parameters,
        lr=lr,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    )
    return optimizer, parameters


def _special_token_ids(model: nn.Module) -> tuple[int, ...]:
    tokenizer = getattr(model, "target_tokenizer", None)
    values = {
        getattr(tokenizer, name, None)
        for name in ("pad_token_id", "bos_token_id", "eos_token_id")
    }
    return tuple(sorted(int(value) for value in values if value is not None))


def adapt_read_batch(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    parameters: Optional[list[nn.Parameter]] = None,
    steps: int = 1,
    lr: float = 1e-4,
    gamma: float = math.exp(-1.0),
    balance_weight: float = 0.0,
) -> READTTAResult:
    """Adapt one unlabeled target batch and return before/after predictions."""
    if steps <= 0:
        raise ValueError("READ adaptation requires at least one step.")
    observed = observed_input_batch(batch)
    allowed_parameters = configure_read_tta_parameters(model)
    allowed_ids = {id(parameter) for parameter in allowed_parameters}
    if parameters is not None and {id(parameter) for parameter in parameters} != allowed_ids:
        raise ValueError("READ parameters must be exactly the final fusion Q/K/V set.")
    parameters = allowed_parameters
    if optimizer is None:
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=0.0,
            betas=(0.9, 0.999),
        )
    else:
        optimizer_ids = {
            id(parameter)
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        }
        if optimizer_ids != allowed_ids:
            raise ValueError(
                "READ optimizer parameters must be exactly the final fusion Q/K/V set."
            )

    model.eval()
    with torch.no_grad():
        predictions_before = model.generate(observed, n_beams=1)
    pseudo_sequences = predictions_before
    diagnostics: Dict[str, torch.Tensor] = {}
    loss = torch.zeros((), device=predictions_before.device)

    for step_index in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits, generated_tokens, token_mask = model.score_generated_sequences(
            observed, pseudo_sequences
        )
        loss, diagnostics = read_sequence_loss(
            logits,
            generated_tokens,
            token_mask,
            gamma=gamma,
            balance_weight=balance_weight,
            excluded_token_ids=_special_token_ids(model),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step_index + 1 < int(steps):
            with torch.no_grad():
                pseudo_sequences = model.generate(observed, n_beams=1)

    with torch.no_grad():
        predictions_after = model.generate(observed, n_beams=1)
    return READTTAResult(
        predictions_before=predictions_before.detach(),
        predictions_after=predictions_after.detach(),
        loss=float(loss.detach().cpu()),
        confidence_loss=float(diagnostics["confidence_loss"].detach().cpu()),
        balance_entropy=float(diagnostics["balance_entropy"].detach().cpu()),
        mean_sequence_confidence=float(
            diagnostics["mean_sequence_confidence"].detach().cpu()
        ),
    )

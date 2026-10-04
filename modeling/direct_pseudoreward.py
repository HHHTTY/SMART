"""Direct teacher-forced pseudo-sequence objectives for continual TTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .molecular_pseudo_reward import (
    canonical_smiles,
    formula_signature,
    molecular_formula,
)


@dataclass(frozen=True)
class PseudoGateDecision:
    canonical: str | None
    valid: bool
    formula_match: bool
    update: bool
    skip_reason: str | None


def evaluate_pseudo_gate(pseudo_smiles: str, observed_formula: str) -> PseudoGateDecision:
    """Apply the label-free validity and observed-Formula update gate."""
    canonical = canonical_smiles(pseudo_smiles)
    if canonical is None:
        return PseudoGateDecision(None, False, False, False, "invalid_pseudo")
    predicted_formula = molecular_formula(canonical)
    formula_match = (
        predicted_formula is not None
        and formula_signature(predicted_formula) == formula_signature(observed_formula)
    )
    if not formula_match:
        return PseudoGateDecision(
            canonical, True, False, False, "observed_formula_mismatch"
        )
    return PseudoGateDecision(canonical, True, True, True, None)


def pad_pseudo_token_sequences(
    model: Any, sequences: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Pad frozen-teacher token IDs without decode/re-tokenize changes."""
    if not sequences:
        raise ValueError("pseudo token sequence list is empty")
    device = next(model.parameters()).device
    rows = [sequence.detach().flatten().to(device) for sequence in sequences]
    if any(row.numel() < 2 for row in rows):
        raise ValueError("pseudo token sequences must contain at least BOS and EOS")
    return pad_sequence(
        rows,
        batch_first=True,
        padding_value=int(model.target_tokenizer.pad_token_id),
    )


def length_normalized_sequence_nll(
    logits: torch.Tensor,
    next_tokens: torch.Tensor,
    *,
    pad_token_id: int,
) -> torch.Tensor:
    """Return one FP32, length-normalized teacher-forced NLL per sequence."""
    if logits.shape[:2] != next_tokens.shape:
        raise ValueError("logits and next-token targets have incompatible shapes")
    token_nll = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        next_tokens.reshape(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).view_as(next_tokens)
    valid = next_tokens.ne(pad_token_id)
    return token_nll.masked_fill(~valid, 0.0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def direct_pseudo_sequence_nll(
    model: Any,
    batch: Mapping[str, Any],
    pseudo_sequences: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a fixed pseudo sequence through the complete encoder-decoder."""
    state = model.prepare_generation_state(dict(batch))
    decoder_input = pseudo_sequences[:, :-1].contiguous()
    next_tokens = pseudo_sequences[:, 1:].contiguous()
    pad_token_id = int(model.target_tokenizer.pad_token_id)
    output = model.hf_model(
        encoder_outputs=state["encoder_outputs"],
        attention_mask=state["attention_mask"],
        decoder_input_ids=decoder_input,
        decoder_attention_mask=decoder_input.ne(pad_token_id).long(),
        labels=None,
        use_cache=False,
    )
    per_sequence = length_normalized_sequence_nll(
        output.logits,
        next_tokens,
        pad_token_id=pad_token_id,
    )
    return per_sequence.mean(), per_sequence


def gradient_pair_statistics(
    full_gradients: Sequence[torch.Tensor | None],
    subset_gradients: Sequence[torch.Tensor | None],
) -> dict[str, float | int]:
    """Compute FP32 norms and cosine without concatenating model gradients."""
    if len(full_gradients) != len(subset_gradients):
        raise ValueError("gradient sequences must have equal length")
    first = next(
        (value for pair in zip(full_gradients, subset_gradients) for value in pair if value is not None),
        None,
    )
    device = first.device if first is not None else torch.device("cpu")
    full_sq = torch.zeros((), dtype=torch.float64, device=device)
    subset_sq = torch.zeros((), dtype=torch.float64, device=device)
    dot = torch.zeros((), dtype=torch.float64, device=device)
    full_tensors = 0
    subset_tensors = 0
    shared_tensors = 0
    full_elements = 0
    subset_elements = 0
    for full, subset in zip(full_gradients, subset_gradients):
        if full is not None:
            current = full.detach().float()
            full_sq += torch.sum(current * current, dtype=torch.float64)
            full_tensors += 1
            full_elements += current.numel()
        if subset is not None:
            current = subset.detach().float()
            subset_sq += torch.sum(current * current, dtype=torch.float64)
            subset_tensors += 1
            subset_elements += current.numel()
        if full is not None and subset is not None:
            dot += torch.sum(
                full.detach().float() * subset.detach().float(), dtype=torch.float64
            )
            shared_tensors += 1
    full_norm = full_sq.sqrt()
    subset_norm = subset_sq.sqrt()
    denominator = full_norm * subset_norm
    cosine = dot / denominator if float(denominator) > 0 else None
    return {
        "full_norm": float(full_norm),
        "subset_norm": float(subset_norm),
        "cosine": float(cosine) if cosine is not None else None,
        "full_tensors": full_tensors,
        "subset_tensors": subset_tensors,
        "shared_tensors": shared_tensors,
        "full_elements": full_elements,
        "subset_elements": subset_elements,
    }


def install_combined_gradients(
    parameters: Sequence[torch.nn.Parameter],
    full_gradients: Sequence[torch.Tensor | None],
    subset_gradients: Sequence[torch.Tensor | None] | None,
    *,
    subset_weight: float,
) -> tuple[int, int]:
    """Install the exact gradient of full_loss + weight * subset_loss."""
    if len(parameters) != len(full_gradients):
        raise ValueError("parameter and full-gradient sequences have unequal length")
    if subset_gradients is not None and len(parameters) != len(subset_gradients):
        raise ValueError("parameter and subset-gradient sequences have unequal length")
    tensors = 0
    elements = 0
    for index, (parameter, full) in enumerate(zip(parameters, full_gradients)):
        subset = None if subset_gradients is None else subset_gradients[index]
        if full is None and subset is None:
            parameter.grad = None
            continue
        if full is None:
            combined = subset.detach().mul(subset_weight)
        elif subset is None:
            combined = full.detach().clone()
        else:
            combined = full.detach().add(subset.detach(), alpha=subset_weight)
        if not torch.isfinite(combined).all():
            raise FloatingPointError("non-finite direct pseudo-reward gradient")
        parameter.grad = combined
        tensors += 1
        elements += combined.numel()
    return tensors, elements

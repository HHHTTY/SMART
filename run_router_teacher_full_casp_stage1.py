#!/usr/bin/env python
"""CASP Stage 1 with frozen or shared router-view and Full-view branches."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import itertools
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import run_batchwise_pseudoreward_tta as routed
import run_pmgfa_direct_casp as pmgfa
import __main__
try:
    from tools import build_chemotion_compatible_preprocessor as _chemotion_compat
except ImportError:
    _chemotion_compat = None
if _chemotion_compat is not None:
    for _compat_name in (
        "ChemotionFormulaPreprocessor",
        "ChemotionCarbonPreprocessor",
        "ChemotionMultipletPreprocessor",
        "ChemotionMSMSTextPreprocessor",
    ):
        setattr(__main__, _compat_name, getattr(_chemotion_compat, _compat_name))
from analytical_fm.modeling.direct_pseudoreward import (
    direct_pseudo_sequence_nll,
    pad_pseudo_token_sequences,
)
from analytical_fm.modeling.molecular_pseudo_reward import (
    fingerprint_distance_target,
    fingerprint_listwise_loss,
    length_normalized_sequence_log_probs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ms-column", default=None)
    parser.add_argument("--ir-column", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--beams", type=int, default=10)
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum epochs; 0 runs without a fixed cap and requires early stopping.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--teacher-mode",
        choices=("frozen", "shared"),
        default="frozen",
        help=(
            "Frozen uses the historical stop-gradient teacher. Shared reuses the "
            "same model parameters for router-view and Full-view branches and "
            "backpropagates through both branches of encoder alignment."
        ),
    )
    parser.add_argument(
        "--update-scope", choices=("all_norm", "full_model"), default="all_norm"
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("global", "matched_route"),
        default="global",
        help=(
            "Align whole-view pooled states or concatenate matched Formula and "
            "router-selected modality states before NT-Xent."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument(
        "--projector-dim",
        type=int,
        default=0,
        help="Use a paired residual MLP projector before encoder alignment; 0 disables it.",
    )
    parser.add_argument("--projector-hidden-dim", type=int, default=512)
    parser.add_argument(
        "--hyperbolic-align-weight",
        type=float,
        default=0.0,
        help="Heterogeneous Lorentz alignment weight; independent of Euclidean NT-Xent.",
    )
    parser.add_argument("--hyperbolic-teacher-curvature", type=float, default=0.5)
    parser.add_argument("--hyperbolic-student-curvature", type=float, default=1.0)
    parser.add_argument("--hyperbolic-intermediate-curvature", type=float, default=0.75)
    parser.add_argument("--hyperbolic-temperature", type=float, default=0.2)
    parser.add_argument("--pseudo-ce-weight", type=float, default=0.0)
    parser.add_argument("--decoder-kl-weight", type=float, default=0.0)
    parser.add_argument("--decoder-hidden-weight", type=float, default=0.0)
    parser.add_argument(
        "--router-aux-ce-weight",
        type=float,
        default=0.0,
        help="Auxiliary CE for predicting the frozen router's selected modality subset from Full encoder state.",
    )
    parser.add_argument(
        "--decoder-infonce-weight",
        type=float,
        default=0.0,
        help="Weight for projected decoder hidden InfoNCE on the shared teacher prefix.",
    )
    parser.add_argument(
        "--decoder-projector-dim",
        type=int,
        default=256,
        help="Output dimension of the decoder residual-MLP projector.",
    )
    parser.add_argument(
        "--decoder-projector-hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension of the decoder residual-MLP projector.",
    )
    parser.add_argument("--decoder-temperature", type=float, default=2.0)
    parser.add_argument("--fingerprint-listwise-weight", type=float, default=0.0)
    parser.add_argument("--fingerprint-candidates", type=int, default=5)
    parser.add_argument("--fingerprint-target-temperature", type=float, default=0.25)
    parser.add_argument("--fingerprint-policy-temperature", type=float, default=1.0)
    parser.add_argument("--fingerprint-formula-penalty", type=float, default=0.5)
    parser.add_argument("--fingerprint-invalid-penalty", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=0.8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--update-frequency",
        choices=("batch", "epoch"),
        default="batch",
        help=(
            "Apply one optimizer update per batch (historical default) or "
            "accumulate the mean epoch gradient and update once at epoch end."
        ),
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3247)
    parser.add_argument("--save-epochs", default="1,3,6,9,12,15")
    parser.add_argument(
        "--checkpoint-interval-epochs",
        type=int,
        default=0,
        help="Also save interval checkpoints every N epochs; 0 disables interval saves.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop after this many epochs without objective improvement; 0 disables.")
    parser.add_argument("--early-stop-min-epochs", type=int, default=1)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _autocast(args: argparse.Namespace, device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and args.precision == "bf16",
    )


class ResidualProjector(nn.Module):
    """Stable shared initialization for teacher/student contrastive projections."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.skip = nn.Linear(input_dim, output_dim, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.orthogonal_(self.skip.weight)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value.float())
        return F.normalize(self.skip(normalized) + self.residual(normalized), dim=-1)


def _model_hidden_size(model: Any) -> int:
    config = model.hf_model.config
    for name in ("d_model", "hidden_size"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise RuntimeError("could not infer encoder hidden size from model config")


class RouterAuxiliaryHead(nn.Module):
    """Training-only classifier for the frozen router action, not generation."""

    def __init__(self, input_dim: int, classes: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, classes)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(value.float()))


def _router_subset_labels(
    selected: Sequence[tuple[str, ...]], device: torch.device
) -> torch.Tensor:
    order = {"HNMR": 0, "CNMR": 1, "MSMS": 2, "IR": 3}
    labels = []
    for names in selected:
        mask = 0
        for name in names:
            canonical = _canonical_modality(name)
            if canonical not in order:
                raise ValueError(f"unknown router modality for auxiliary CE: {name}")
            mask |= 1 << order[canonical]
        labels.append(mask)
    return torch.tensor(labels, dtype=torch.long, device=device)


def _decoder_hidden(output: Any) -> torch.Tensor:
    value = getattr(output, "decoder_hidden_states", None)
    if isinstance(value, Mapping):
        value = value.get("last_hidden_state")
    elif isinstance(value, (tuple, list)):
        value = value[-1]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("model output did not expose decoder hidden states")
    return value


def _batched_decoder_outputs(
    model: Any,
    batch: Mapping[str, Any],
    padded_tokens: torch.Tensor,
    excluded: frozenset[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.excluded_input_modalities = excluded
    state = model.prepare_generation_state(dict(batch))
    decoder_input = padded_tokens[:, :-1].contiguous()
    next_tokens = padded_tokens[:, 1:].contiguous()
    pad_token_id = int(model.target_tokenizer.pad_token_id)
    output = model.hf_model(
        encoder_outputs=state["encoder_outputs"],
        attention_mask=state["attention_mask"],
        decoder_input_ids=decoder_input,
        decoder_attention_mask=decoder_input.ne(pad_token_id).long(),
        labels=None,
        use_cache=False,
    )
    valid = next_tokens.ne(pad_token_id)
    return output.logits.float(), _decoder_hidden(output).float(), valid


def _masked_decoder_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher decoder logits have different shapes")
    if valid.shape != student_logits.shape[:2]:
        raise ValueError("decoder KL mask has an incompatible shape")
    teacher_log_prob = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    student_log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    token_kl = (teacher_log_prob.exp() * (teacher_log_prob - student_log_prob)).sum(-1)
    return temperature**2 * token_kl.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1)


def _masked_decoder_hidden_alignment(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError("student and teacher decoder hidden states have different shapes")
    token_distance = 1.0 - F.cosine_similarity(
        student_hidden.float(), teacher_hidden.detach().float(), dim=-1
    )
    return token_distance.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1)


def _masked_decoder_projected_infonce(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    valid: torch.Tensor,
    student_projector: ResidualProjector,
    teacher_projector: ResidualProjector,
    temperature: float,
) -> torch.Tensor:
    """Symmetric batch InfoNCE over masked decoder trajectories.

    The residual MLP is a training-time projection only; the generator itself
    remains unchanged at inference. Teacher features and its projector are
    detached, while the student projector is optimized jointly with the model.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError("student and teacher decoder hidden states have different shapes")
    if valid.shape != student_hidden.shape[:2]:
        raise ValueError("decoder InfoNCE mask has an incompatible shape")
    weights = valid.to(dtype=student_hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    teacher_pool = (teacher_hidden.detach() * weights).sum(dim=1) / denom
    student_pool = (student_hidden * weights).sum(dim=1) / denom
    with torch.no_grad():
        teacher_proj = teacher_projector(teacher_pool)
    student_proj = student_projector(student_pool)
    logits = teacher_proj @ student_proj.transpose(0, 1) / max(float(temperature), 1e-4)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def _router_decoder_alignment(
    teacher: Any,
    student: Any,
    batch: Mapping[str, Any],
    route_rows: Mapping[int, Mapping[str, Any]],
    input_modalities: frozenset[str],
    selected: list[tuple[str, ...]],
    temperature: float,
    decoder_teacher_projector: ResidualProjector | None = None,
    decoder_student_projector: ResidualProjector | None = None,
    decoder_infonce_temperature: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Align soft decoder behavior on a shared router-Top-1 prefix, without CE."""
    source_rows = [int(value) for value in batch["source_row_indices"]]
    sequences = [
        torch.tensor(route_rows[row]["teacher_pseudo_token_ids"], dtype=torch.long)
        for row in source_rows
    ]
    padded = pad_pseudo_token_sequences(student, sequences)
    student_logits, student_hidden, valid = _batched_decoder_outputs(
        student, batch, padded, frozenset()
    )
    batch_size = len(source_rows)
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, names in enumerate(selected):
        groups[names].append(index)
    teacher_logits_rows: list[torch.Tensor | None] = [None] * batch_size
    teacher_hidden_rows: list[torch.Tensor | None] = [None] * batch_size
    with torch.no_grad():
        for names, indices in groups.items():
            current = routed._select_batch(batch, indices, batch_size)
            current_tokens = padded[torch.tensor(indices, device=padded.device)]
            keep = frozenset(("Formula", *names))
            excluded = frozenset(input_modalities.difference(keep))
            logits, hidden, _ = _batched_decoder_outputs(
                teacher, current, current_tokens, excluded
            )
            for local_index, row_index in enumerate(indices):
                teacher_logits_rows[row_index] = logits[local_index]
                teacher_hidden_rows[row_index] = hidden[local_index]
    if any(value is None for value in teacher_logits_rows + teacher_hidden_rows):
        raise RuntimeError("router decoder alignment did not cover the batch")
    teacher_logits = torch.stack([value for value in teacher_logits_rows if value is not None])
    teacher_hidden = torch.stack([value for value in teacher_hidden_rows if value is not None])
    kl = _masked_decoder_kl(student_logits, teacher_logits, valid, temperature)
    hidden_loss = _masked_decoder_hidden_alignment(
        student_hidden, teacher_hidden, valid
    )
    if (decoder_teacher_projector is None) != (decoder_student_projector is None):
        raise ValueError("decoder teacher/student projector must be supplied together")
    decoder_infonce = torch.zeros((), device=student_logits.device)
    if decoder_teacher_projector is not None and decoder_student_projector is not None:
        decoder_infonce = _masked_decoder_projected_infonce(
            student_hidden,
            teacher_hidden,
            valid,
            decoder_student_projector,
            decoder_teacher_projector,
            decoder_infonce_temperature,
        )
    return kl, hidden_loss, decoder_infonce, {
        "decoder_valid_tokens": float(valid.sum().detach().cpu()),
        "decoder_teacher_entropy": float(
            (-(F.softmax(teacher_logits.float(), -1) * F.log_softmax(teacher_logits.float(), -1)).sum(-1))
            .masked_fill(~valid, 0.0)
            .sum()
            .div(valid.sum().clamp_min(1))
            .detach()
            .cpu()
        ),
    }


def _representation(
    model: Any, batch: Mapping[str, Any], excluded: frozenset[str]
) -> torch.Tensor:
    model.excluded_input_modalities = excluded
    input_ids, attention_mask, embeddings = model._prepare_generation_inputs(
        dict(batch), apply_modality_dropout=False
    )
    encoded = model._encode_generation_inputs(embeddings, attention_mask, input_ids)
    hidden = encoded["last_hidden_state"].float()
    mask = attention_mask.bool().unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1).float()


def _router_teacher_representation(
    teacher: Any,
    batch: Mapping[str, Any],
    route_rows: Mapping[int, Mapping[str, Any]],
    input_modalities: frozenset[str],
    alignment_mode: str,
    track_grad: bool = False,
) -> tuple[torch.Tensor, list[tuple[str, ...]]]:
    batch_size = len(batch["input_formulas"])
    source_rows = [int(value) for value in batch["source_row_indices"]]
    selected: list[tuple[str, ...]] = []
    for source_row in source_rows:
        row = route_rows[source_row]
        names = tuple(str(value) for value in row.get("selected_modalities", (row["selected_modality"],)))
        names = tuple(value for value in names if value != "Formula")
        if not names:
            raise ValueError(f"route row {source_row} has no selected spectral modality")
        selected.append(names)

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, names in enumerate(selected):
        groups[names].append(index)
    pieces: list[torch.Tensor | None] = [None] * batch_size
    context = nullcontext() if track_grad else torch.no_grad()
    with context:
        for names, indices in groups.items():
            current = routed._select_batch(batch, indices, batch_size)
            keep = frozenset(("Formula", *names))
            excluded = frozenset(input_modalities.difference(keep))
            if alignment_mode == "global":
                values = _representation(teacher, current, excluded)
            else:
                values = _matched_route_representation(
                    teacher, current, [names] * len(indices), excluded
                )
            if not track_grad:
                values = values.detach()
            for local_index, row_index in enumerate(indices):
                pieces[row_index] = values[local_index]
    if any(value is None for value in pieces):
        raise RuntimeError("router teacher representation did not cover the batch")
    return torch.stack([value for value in pieces if value is not None]), selected


def _matched_route_representation(
    model: Any,
    batch: Mapping[str, Any],
    selected: list[tuple[str, ...]],
    excluded: frozenset[str],
) -> torch.Tensor:
    """Concatenate contextualized Formula and selected-modality anchors."""
    pooled = _pooled_modality_hidden(model, batch, excluded)
    formula = pooled.get("Formula")
    if formula is None:
        raise ValueError("matched-route alignment requires Formula tokens")
    selected_rows: list[torch.Tensor] = []
    for row, names in enumerate(selected):
        values = [pooled[name][row] for name in names if name in pooled]
        if len(values) != len(names):
            missing = sorted(set(names).difference(pooled))
            raise ValueError(f"matched-route alignment is missing modalities: {missing}")
        selected_rows.append(torch.stack(values).mean(dim=0))
    selected_tensor = torch.stack(selected_rows)
    return torch.cat(
        (F.normalize(formula.float(), dim=-1), F.normalize(selected_tensor.float(), dim=-1)),
        dim=-1,
    )


def _canonical_modality(name: str) -> str:
    upper = str(name).upper()
    aliases = {
        "FORMULA": "Formula",
        "MOLECULAR_FORMULA": "Formula",
        "HNMR": "HNMR",
        "H_NMR": "HNMR",
        "CNMR": "CNMR",
        "C_NMR": "CNMR",
        "MSMS": "MSMS",
        "MS/MS": "MSMS",
        "MS2": "MSMS",
        "IR": "IR",
        "IR_SPECTRA": "IR",
    }
    return aliases.get(upper, str(name))


def _pooled_modality_hidden(
    model: Any, batch: Mapping[str, Any], excluded: frozenset[str]
) -> dict[str, torch.Tensor]:
    model.excluded_input_modalities = excluded
    input_ids, attention_mask, embeddings = model._prepare_generation_inputs(
        dict(batch), apply_modality_dropout=False
    )
    encoded = model._encode_generation_inputs(embeddings, attention_mask, input_ids)
    hidden = encoded["last_hidden_state"]
    pooled: dict[str, torch.Tensor] = {}
    for name, start, end in model._exposure_modality_spans(input_ids):
        mask = attention_mask[:, start:end].to(hidden.dtype)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled[_canonical_modality(name)] = (
            hidden[:, start:end] * mask.unsqueeze(-1)
        ).sum(dim=1) / denominator
    return pooled


def symmetric_nt_xent(
    teacher: torch.Tensor,
    student: torch.Tensor,
    temperature: float,
    *,
    detach_teacher: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if teacher.shape != student.shape:
        raise ValueError("teacher and student representations must have equal shape")
    batch_size = teacher.shape[0]
    if batch_size < 2:
        raise ValueError("symmetric NT-Xent requires at least two samples")
    left_value = teacher.detach() if detach_teacher else teacher
    left = F.normalize(left_value.float(), dim=-1)
    right = F.normalize(student.float(), dim=-1)
    representations = torch.cat((left, right), dim=0)
    similarities = representations @ representations.T
    logits = similarities / float(temperature)
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(self_mask, -torch.inf)
    positives = torch.cat((
        torch.arange(batch_size, 2 * batch_size, device=logits.device),
        torch.arange(batch_size, device=logits.device),
    ))
    loss = F.cross_entropy(logits, positives)
    row = torch.arange(2 * batch_size, device=logits.device)
    positive_similarity = similarities[row, positives]
    negative_mask = ~self_mask
    negative_mask[row, positives] = False
    diagnostics = {
        "positive_similarity": float(positive_similarity.mean().detach().cpu()),
        "negative_similarity": float(similarities[negative_mask].mean().detach().cpu()),
        "contrastive_top1": float(logits.argmax(dim=1).eq(positives).float().mean().detach().cpu()),
    }
    return loss, diagnostics


def _bounded_lorentz_exp(
    value: torch.Tensor, curvature: float, eps: float = 1e-6
) -> torch.Tensor:
    """Map Euclidean tangent vectors to Lorentz space with bounded radius."""
    if curvature <= 0:
        raise ValueError("hyperbolic curvature must be positive")
    x = value.float()
    sqrt_curv = float(curvature) ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = 3.0 / sqrt_curv
    bounded_norm = max_norm * torch.tanh(norm / max_norm)
    direction = x / norm
    radius = sqrt_curv * bounded_norm
    return torch.sinh(radius) * direction / sqrt_curv


def _lorentz_log0(value: torch.Tensor, curvature: float, eps: float = 1e-6) -> torch.Tensor:
    """Map Lorentz space components back to the tangent space at the origin."""
    x = value.float()
    sqrt_curv = float(curvature) ** 0.5
    time = torch.sqrt(1.0 / float(curvature) + x.square().sum(dim=-1, keepdim=True))
    distance = torch.acosh((sqrt_curv * time).clamp_min(1.0 + eps)) / sqrt_curv
    return distance * x / (sqrt_curv * x.norm(dim=-1, keepdim=True).clamp_min(eps))


def heterogeneous_lorentz_alignment(
    teacher: torch.Tensor,
    student: torch.Tensor,
    *,
    teacher_curvature: float,
    student_curvature: float,
    intermediate_curvature: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align paired representations through heterogeneous Lorentz manifolds.

    This is the generation-task proxy used here: each view is embedded with its
    own curvature, transported through the origin to a shared intermediate
    curvature, and scored by cross-batch geodesic distance. It preserves the
    paper's heterogeneous-manifold idea without inventing molecular tree labels.
    """
    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError("teacher and student representations must be equal-shaped matrices")
    if teacher.shape[0] < 2:
        raise ValueError("hyperbolic alignment requires at least two samples")
    if temperature <= 0:
        raise ValueError("hyperbolic temperature must be positive")
    # Lorentz products and acosh are numerically fragile in BF16. The runner's
    # outer autocast must not lower precision inside this block.
    with torch.autocast(device_type=teacher.device.type, enabled=False):
        with torch.no_grad():
            teacher_own = _bounded_lorentz_exp(teacher.detach(), teacher_curvature)
            teacher_tangent = _lorentz_log0(teacher_own, teacher_curvature)
            teacher_mid = _bounded_lorentz_exp(teacher_tangent, intermediate_curvature)
        student_own = _bounded_lorentz_exp(student, student_curvature)
        student_tangent = _lorentz_log0(student_own, student_curvature)
        student_mid = _bounded_lorentz_exp(student_tangent, intermediate_curvature)

        def distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            curv = float(intermediate_curvature)
            left_time = torch.sqrt(1.0 / curv + left.square().sum(dim=-1, keepdim=True))
            right_time = torch.sqrt(1.0 / curv + right.square().sum(dim=-1, keepdim=True))
            lorentz_inner = left @ right.T - left_time @ right_time.T
            argument = (-curv * lorentz_inner).clamp_min(1.0 + 1e-6)
            return torch.acosh(argument) / (curv**0.5)

        distances = distance(student_mid, teacher_mid)
        logits = -distances / float(temperature)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
    diagnostics = {
        "hyperbolic_positive_distance": float(distances.diag().mean().detach().cpu()),
        "hyperbolic_negative_distance": float(
            distances[~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)].mean().detach().cpu()
        ),
        "hyperbolic_top1": float((logits.argmax(dim=1) == labels).float().mean().detach().cpu()),
    }
    return loss, diagnostics


def _router_pseudo_ce(
    student: Any,
    batch: Mapping[str, Any],
    route_rows: Mapping[int, Mapping[str, Any]],
) -> tuple[torch.Tensor, int]:
    source_rows = [int(value) for value in batch["source_row_indices"]]
    sequences = [
        torch.tensor(route_rows[row]["teacher_pseudo_token_ids"], dtype=torch.long)
        for row in source_rows
    ]
    padded = pad_pseudo_token_sequences(student, sequences)
    student.excluded_input_modalities = frozenset()
    _mean, per_sequence = direct_pseudo_sequence_nll(student, batch, padded)
    accepted = torch.tensor(
        [bool(route_rows[row].get("pseudo_gate_update", False)) for row in source_rows],
        dtype=torch.bool,
        device=per_sequence.device,
    )
    if not bool(accepted.any()):
        return per_sequence.sum() * 0.0, 0
    return per_sequence[accepted].mean(), int(accepted.sum().item())


def _tokenize_smiles_candidates(model: Any, smiles: list[str]) -> torch.Tensor:
    encoded = model.target_tokenizer(
        smiles,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"]
    if token_ids.ndim != 2 or token_ids.shape[0] != len(smiles):
        raise ValueError("candidate tokenizer returned an incompatible tensor")
    if token_ids.shape[1] < 2:
        raise ValueError("fingerprint candidates must include at least BOS and EOS")
    return token_ids.to(next(model.parameters()).device)


def _router_fingerprint_listwise(
    student: Any,
    batch: Mapping[str, Any],
    route_rows: Mapping[int, Mapping[str, Any]],
    *,
    candidates: int,
    target_temperature: float,
    policy_temperature: float,
    formula_penalty: float,
    invalid_penalty: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rank frozen router-teacher beams by detached chemical similarity."""
    source_rows = [int(value) for value in batch["source_row_indices"]]
    candidate_rows: list[list[str]] = []
    pseudo_smiles: list[str] = []
    accepted_rows: list[bool] = []
    for source_row in source_rows:
        route = route_rows[source_row]
        beams = [str(value) for value in route.get("teacher_beam_smiles", ())]
        if len(beams) < candidates:
            raise ValueError(
                f"route row {source_row} has {len(beams)} beams, expected {candidates}"
            )
        candidate_rows.append(beams[:candidates])
        pseudo_smiles.append(str(route["teacher_pseudo_smiles"]))
        accepted_rows.append(bool(route.get("pseudo_gate_update", False)))

    device = next(student.parameters()).device
    accepted = torch.tensor(accepted_rows, dtype=torch.bool, device=device)
    if not bool(accepted.any()):
        zero = next(student.parameters()).sum() * 0.0
        return zero, {
            "fingerprint_target_entropy": 0.0,
            "fingerprint_target_top1_mass": 0.0,
            "fingerprint_mean_tanimoto": 0.0,
        }

    target = fingerprint_distance_target(
        candidate_rows,
        pseudo_smiles,
        [str(value) for value in batch["input_formulas"]],
        temperature=target_temperature,
        formula_penalty=formula_penalty,
        invalid_penalty=invalid_penalty,
        device=device,
    )
    flat_smiles = [value for row in candidate_rows for value in row]
    candidate_tokens = _tokenize_smiles_candidates(student, flat_smiles)
    batch_size = len(source_rows)
    student.excluded_input_modalities = frozenset()
    generation_state = student.prepare_generation_state(dict(batch))
    logits, next_tokens, _valid = student.score_generated_sequences(
        dict(batch), candidate_tokens, generation_state=generation_state
    )
    sequence_log_probs = length_normalized_sequence_log_probs(
        logits,
        next_tokens,
        pad_token_id=int(student.target_tokenizer.pad_token_id),
        eos_token_id=student.target_tokenizer.eos_token_id,
    )
    sequence_log_probs = sequence_log_probs.view(batch_size, candidates)
    loss = fingerprint_listwise_loss(
        sequence_log_probs[accepted],
        target.probabilities[accepted],
        policy_temperature=policy_temperature,
    )
    probabilities = target.probabilities[accepted]
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    diagnostics = {
        "fingerprint_target_entropy": float(entropy.mean().detach().cpu()),
        "fingerprint_target_top1_mass": float(probabilities[:, 0].mean().detach().cpu()),
        "fingerprint_mean_tanimoto": float(target.tanimoto[accepted].mean().detach().cpu()),
    }
    return loss, diagnostics


def _configure_trainable(
    model: Any, scope: str
) -> tuple[list[torch.nn.Parameter], list[str]]:
    parameters: list[torch.nn.Parameter] = []
    names: list[str] = []
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        selected = scope == "full_model" or (
            parameter.ndim <= 2 and ("norm" in lowered or "layernorm" in lowered)
        )
        parameter.requires_grad_(selected)
        if selected:
            parameters.append(parameter)
            names.append(name)
    if not parameters:
        raise RuntimeError(f"{scope} scope selected no parameters")
    return parameters, names


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs < 0 or args.checkpoint_interval_epochs < 0:
        raise ValueError("epoch limits and checkpoint intervals must be non-negative")
    if args.epochs == 0 and args.early_stop_patience < 1:
        raise ValueError("unbounded Stage 1 requires --early-stop-patience > 0")
    if args.alignment_weight < 0 or args.hyperbolic_align_weight < 0 or args.pseudo_ce_weight < 0:
        raise ValueError("alignment and pseudo-CE weights must be non-negative")
    if args.fingerprint_listwise_weight < 0:
        raise ValueError("fingerprint-listwise-weight must be non-negative")
    if args.fingerprint_candidates < 2:
        raise ValueError("fingerprint-candidates must be at least 2")
    if args.fingerprint_target_temperature <= 0 or args.fingerprint_policy_temperature <= 0:
        raise ValueError("fingerprint temperatures must be positive")
    if args.fingerprint_formula_penalty < 0 or args.fingerprint_invalid_penalty < 0:
        raise ValueError("fingerprint penalties must be non-negative")
    if args.projector_dim < 0 or args.projector_hidden_dim < 1:
        raise ValueError("projector dimensions are invalid")
    if args.decoder_kl_weight < 0 or args.decoder_hidden_weight < 0 or args.decoder_infonce_weight < 0 or args.router_aux_ce_weight < 0:
        raise ValueError("decoder alignment weights must be non-negative")
    if args.decoder_infonce_weight > 0 and args.decoder_projector_dim < 1:
        raise ValueError("decoder projector dimension must be positive when decoder InfoNCE is enabled")
    if args.decoder_temperature <= 0:
        raise ValueError("decoder-temperature must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    save_epochs = {int(value) for value in args.save_epochs.split(",") if value.strip()}

    route_manifest, route_rows = pmgfa._load_route_records(args.route_manifest)
    data_config, preprocessors, raw, loader = routed._build_input_only_loader(args)
    if not set(range(len(raw))).issubset(route_rows):
        raise ValueError("route manifest does not cover the input-only dataset")
    input_modalities = frozenset(
        name for name, metadata in data_config.items() if not metadata.get("target", False)
    )
    student = routed._build_model(args, data_config, preprocessors, device, trainable=True).eval()
    if args.teacher_mode == "shared":
        teacher = student
    else:
        teacher = routed._build_model(args, data_config, preprocessors, device, trainable=False).eval()
        teacher.requires_grad_(False)
    parameters, parameter_names = _configure_trainable(student, args.update_scope)
    teacher_projector: ResidualProjector | None = None
    student_projector: ResidualProjector | None = None
    if args.projector_dim > 0:
        representation_dim = _model_hidden_size(student) * (
            2 if args.alignment_mode == "matched_route" else 1
        )
        teacher_projector = ResidualProjector(
            representation_dim, args.projector_dim, args.projector_hidden_dim
        ).to(device)
        student_projector = copy.deepcopy(teacher_projector).to(device)
        teacher_projector.requires_grad_(False).eval()
        student_projector.train()
        projector_parameters = list(student_projector.parameters())
        parameters.extend(projector_parameters)
        parameter_names.extend(
            f"alignment_projector.{name}"
            for name, _parameter in student_projector.named_parameters()
        )
    decoder_teacher_projector: ResidualProjector | None = None
    decoder_student_projector: ResidualProjector | None = None
    if args.decoder_infonce_weight > 0:
        decoder_dim = _model_hidden_size(student)
        decoder_teacher_projector = ResidualProjector(
            decoder_dim, args.decoder_projector_dim, args.decoder_projector_hidden_dim
        ).to(device)
        decoder_student_projector = copy.deepcopy(decoder_teacher_projector).to(device)
        decoder_teacher_projector.requires_grad_(False).eval()
        decoder_student_projector.train()
        parameters.extend(list(decoder_student_projector.parameters()))
        parameter_names.extend(
            f"decoder_alignment_projector.{name}"
            for name, _parameter in decoder_student_projector.named_parameters()
        )
    router_aux_head: RouterAuxiliaryHead | None = None
    if args.router_aux_ce_weight > 0:
        router_aux_head = RouterAuxiliaryHead(_model_hidden_size(student), classes=16).to(device)
        router_aux_head.train()
        parameters.extend(list(router_aux_head.parameters()))
        parameter_names.extend(
            f"router_aux_head.{name}" for name, _parameter in router_aux_head.named_parameters()
        )
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    route_counts: Counter[str] = Counter()
    history: list[dict[str, Any]] = []
    effective_gradient_names: set[str] = set()
    started = time.time()
    best_epoch_objective = float("inf")
    stale_epochs = 0
    completed_epoch = 0

    epoch_iterator = itertools.count(1) if args.epochs == 0 else range(1, args.epochs + 1)
    for epoch in epoch_iterator:
        epoch_rows: list[dict[str, float]] = []
        if args.update_frequency == "epoch":
            optimizer.zero_grad(set_to_none=True)
        epoch_batch_count = max(1, len(loader))
        epoch_grad_norm = 0.0
        for batch_index, raw_batch in enumerate(loader):
            batch = routed._move(routed.adaptation_view(raw_batch), device)
            with _autocast(args, device):
                teacher_rep, selected = _router_teacher_representation(
                    teacher,
                    batch,
                    route_rows,
                    input_modalities,
                    args.alignment_mode,
                    track_grad=args.teacher_mode == "shared",
                )
                if args.alignment_mode == "global":
                    student_rep = _representation(student, batch, frozenset())
                else:
                    student_rep = _matched_route_representation(
                        student, batch, selected, frozenset()
                    )
                if teacher_projector is not None and student_projector is not None:
                    with torch.no_grad():
                        teacher_rep = teacher_projector(teacher_rep)
                    student_rep = student_projector(student_rep)
                loss, diagnostics = symmetric_nt_xent(
                    teacher_rep,
                    student_rep,
                    args.temperature,
                    detach_teacher=args.teacher_mode != "shared",
                )
                hyperbolic_loss = torch.zeros((), device=device)
                hyperbolic_diagnostics = {
                    "hyperbolic_positive_distance": 0.0,
                    "hyperbolic_negative_distance": 0.0,
                    "hyperbolic_top1": 0.0,
                }
                if args.hyperbolic_align_weight > 0:
                    hyperbolic_loss, hyperbolic_diagnostics = heterogeneous_lorentz_alignment(
                        teacher_rep,
                        student_rep,
                        teacher_curvature=args.hyperbolic_teacher_curvature,
                        student_curvature=args.hyperbolic_student_curvature,
                        intermediate_curvature=args.hyperbolic_intermediate_curvature,
                        temperature=args.hyperbolic_temperature,
                    )
                pseudo_ce = torch.zeros((), device=device)
                accepted_pseudo = 0
                if args.pseudo_ce_weight > 0:
                    pseudo_ce, accepted_pseudo = _router_pseudo_ce(
                        student, batch, route_rows
                    )
                decoder_kl = torch.zeros((), device=device)
                decoder_hidden = torch.zeros((), device=device)
                decoder_infonce = torch.zeros((), device=device)
                router_aux_ce = torch.zeros((), device=device)
                decoder_diagnostics = {
                    "decoder_valid_tokens": 0.0,
                    "decoder_teacher_entropy": 0.0,
                }
                if (
                    args.decoder_kl_weight > 0
                    or args.decoder_hidden_weight > 0
                    or args.decoder_infonce_weight > 0
                ):
                    decoder_kl, decoder_hidden, decoder_infonce, decoder_diagnostics = (
                        _router_decoder_alignment(
                            teacher,
                            student,
                            batch,
                            route_rows,
                            input_modalities,
                            selected,
                            args.decoder_temperature,
                            decoder_teacher_projector,
                            decoder_student_projector,
                            args.temperature,
                        )
                    )
                if router_aux_head is not None:
                    aux_rep = _representation(student, batch, frozenset())
                    router_aux_ce = F.cross_entropy(
                        router_aux_head(aux_rep), _router_subset_labels(selected, device)
                    )
                fingerprint_listwise = torch.zeros((), device=device)
                fingerprint_diagnostics = {
                    "fingerprint_target_entropy": 0.0,
                    "fingerprint_target_top1_mass": 0.0,
                    "fingerprint_mean_tanimoto": 0.0,
                }
                if args.fingerprint_listwise_weight > 0:
                    fingerprint_listwise, fingerprint_diagnostics = (
                        _router_fingerprint_listwise(
                            student,
                            batch,
                            route_rows,
                            candidates=args.fingerprint_candidates,
                            target_temperature=args.fingerprint_target_temperature,
                            policy_temperature=args.fingerprint_policy_temperature,
                            formula_penalty=args.fingerprint_formula_penalty,
                            invalid_penalty=args.fingerprint_invalid_penalty,
                        )
                    )
                objective = (
                    float(args.alignment_weight) * loss
                    + float(args.hyperbolic_align_weight) * hyperbolic_loss
                    + float(args.pseudo_ce_weight) * pseudo_ce
                    + float(args.decoder_kl_weight) * decoder_kl
                    + float(args.decoder_hidden_weight) * decoder_hidden
                    + float(args.decoder_infonce_weight) * decoder_infonce
                    + float(args.router_aux_ce_weight) * router_aux_ce
                    + float(args.fingerprint_listwise_weight) * fingerprint_listwise
                )
            if args.update_frequency == "batch":
                optimizer.zero_grad(set_to_none=True)
                objective_for_backward = objective
            else:
                # Keep the epoch-level update on the mean batch objective so its
                # gradient scale is comparable to the historical batch mode.
                objective_for_backward = objective / float(epoch_batch_count)
            objective_for_backward.backward()
            effective_gradient_names.update(
                name
                for name, parameter in student.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            )
            if student_projector is not None:
                effective_gradient_names.update(
                    f"alignment_projector.{name}"
                    for name, parameter in student_projector.named_parameters()
                    if parameter.grad is not None
                )
            if args.update_frequency == "batch":
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                optimizer.step()
            else:
                grad_norm = torch.zeros((), device=device)
            route_counts.update(str(tuple(value)) for value in selected)
            record = {
                "loss": float(loss.detach().cpu()),
                "pseudo_ce": float(pseudo_ce.detach().cpu()),
                "decoder_kl": float(decoder_kl.detach().cpu()),
                "decoder_hidden": float(decoder_hidden.detach().cpu()),
                "decoder_infonce": float(decoder_infonce.detach().cpu()),
                "router_aux_ce": float(router_aux_ce.detach().cpu()),
                "fingerprint_listwise": float(fingerprint_listwise.detach().cpu()),
                "objective": float(objective.detach().cpu()),
                "hyperbolic_alignment": float(hyperbolic_loss.detach().cpu()),
                "accepted_pseudo": float(accepted_pseudo),
                "positive_similarity": diagnostics["positive_similarity"],
                "negative_similarity": diagnostics["negative_similarity"],
                "contrastive_top1": diagnostics["contrastive_top1"],
                "grad_norm": float(grad_norm.detach().cpu()),
                **fingerprint_diagnostics,
                **decoder_diagnostics,
                **hyperbolic_diagnostics,
            }
            epoch_rows.append(record)
            if batch_index == 0 or (batch_index + 1) % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "batch": batch_index + 1, **record}), flush=True)
        if args.update_frequency == "epoch":
            epoch_grad_norm = float(
                torch.as_tensor(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip))
                .detach()
                .cpu()
            )
            optimizer.step()
        summary = {
            "epoch": epoch,
            "update_frequency": args.update_frequency,
            "epoch_grad_norm": epoch_grad_norm,
            **{
                key: float(np.mean([row[key] for row in epoch_rows]))
                for key in epoch_rows[0]
            },
            "elapsed_seconds": time.time() - started,
        }
        history.append(summary)
        print(json.dumps({"epoch_summary": summary}), flush=True)
        completed_epoch = epoch
        epoch_objective = float(summary["objective"])
        if epoch_objective < best_epoch_objective - args.early_stop_min_delta:
            best_epoch_objective = epoch_objective
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            epoch in save_epochs
            or (args.epochs > 0 and epoch == args.epochs)
            or (
                args.checkpoint_interval_epochs > 0
                and epoch % args.checkpoint_interval_epochs == 0
            )
            or (
                args.early_stop_patience > 0
                and epoch >= args.early_stop_min_epochs
                and stale_epochs >= args.early_stop_patience
            )
        ):
            torch.save(
                {
                    "method": (
                        "CASP Stage1 shared router-view/Full model"
                        if args.teacher_mode == "shared"
                        else "CASP Stage1 frozen router teacher to Full student"
                    ),
                    "stage": "stage1",
                    "epoch": epoch,
                    "teacher_mode": args.teacher_mode,
                    "update_frequency": args.update_frequency,
                    "student_state_dict": student.state_dict(),
                    "alignment_projector_state_dict": (
                        student_projector.state_dict()
                        if student_projector is not None
                        else None
                    ),
                    "decoder_alignment_projector_state_dict": (
                        decoder_student_projector.state_dict()
                        if decoder_student_projector is not None
                        else None
                    ),
                    "router_aux_head_state_dict": (
                        router_aux_head.state_dict() if router_aux_head is not None else None
                    ),
                    "trainable_parameter_names": parameter_names,
                    "target_labels_loaded_during_adaptation": False,
                    "route_manifest": str(args.route_manifest),
                    "history": history,
                },
                args.run_dir / f"stage1_epoch_{epoch}.pt",
            )
        if (args.early_stop_patience > 0 and epoch >= args.early_stop_min_epochs
                and stale_epochs >= args.early_stop_patience):
            print(json.dumps({"early_stop": True, "epoch": epoch,
                              "best_objective": best_epoch_objective,
                              "stale_epochs": stale_epochs}), flush=True)
            break

    manifest = {
        "method": (
            "CASP Stage1 shared router-view/Full model"
            if args.teacher_mode == "shared"
            else "CASP Stage1 frozen router-view teacher to Full student"
        ),
        "teacher_mode": args.teacher_mode,
        "teacher_view": route_manifest.get(
            "teacher_view", "Formula + per-sample router-selected modalities"
        ),
        "student_view": "Full",
        "objective": (
            "weighted symmetric NT-Xent on encoder representations plus "
            "optional frozen-router pseudo-SMILES CE"
        ),
        "alignment_mode": args.alignment_mode,
        "alignment_weight": args.alignment_weight,
        "projector_dim": args.projector_dim,
        "projector_hidden_dim": args.projector_hidden_dim,
        "decoder_projector_dim": args.decoder_projector_dim,
        "decoder_projector_hidden_dim": args.decoder_projector_hidden_dim,
        "hyperbolic_align_weight": args.hyperbolic_align_weight,
        "hyperbolic_teacher_curvature": args.hyperbolic_teacher_curvature,
        "hyperbolic_student_curvature": args.hyperbolic_student_curvature,
        "hyperbolic_intermediate_curvature": args.hyperbolic_intermediate_curvature,
        "hyperbolic_temperature": args.hyperbolic_temperature,
        "pseudo_ce_weight": args.pseudo_ce_weight,
        "decoder_kl_weight": args.decoder_kl_weight,
        "decoder_hidden_weight": args.decoder_hidden_weight,
        "decoder_infonce_weight": args.decoder_infonce_weight,
        "router_aux_ce_weight": args.router_aux_ce_weight,
        "decoder_temperature": args.decoder_temperature,
        "fingerprint_listwise_weight": args.fingerprint_listwise_weight,
        "fingerprint_candidates": args.fingerprint_candidates,
        "fingerprint_target_temperature": args.fingerprint_target_temperature,
        "fingerprint_policy_temperature": args.fingerprint_policy_temperature,
        "fingerprint_formula_penalty": args.fingerprint_formula_penalty,
        "fingerprint_invalid_penalty": args.fingerprint_invalid_penalty,
        "fingerprint_reference": route_manifest.get(
            "teacher_pseudo_source", "frozen router-teacher Top-1 pseudo-SMILES"
        ),
        "fingerprint_candidate_source": route_manifest.get(
            "fingerprint_candidate_source", "frozen router-teacher beam candidates"
        ),
        "teacher_stop_gradient": args.teacher_mode != "shared",
        "shared_model_parameters": args.teacher_mode == "shared",
        "update_frequency": args.update_frequency,
        "update_scope": args.update_scope,
        "requested_update_scope": args.update_scope,
        "trainable_parameter_count": int(sum(value.numel() for value in parameters)),
        "effective_gradient_parameter_tensor_count": len(effective_gradient_names),
        "effective_gradient_parameter_names": sorted(effective_gradient_names),
        "route_counts_across_epochs": dict(route_counts),
        "route_manifest": route_manifest,
        "target_labels_loaded_during_adaptation": False,
        "history": history,
        "final_checkpoint": str(args.run_dir / f"stage1_epoch_{completed_epoch}.pt"),
    }
    (args.run_dir / "adaptation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.run_dir / "adaptation.complete.json").write_text(
        json.dumps({"complete": True, "epoch": completed_epoch}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"done": True, "run_dir": str(args.run_dir)}), flush=True)


if __name__ == "__main__":
    main()

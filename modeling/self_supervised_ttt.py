"""Label-free per-sample test-time adaptation for multimodal spectra.

The objective deliberately uses only the observed input modalities.  A masked
view must produce the same retrieval representation as the unmasked view, and
the available modality heads should agree on the fused fingerprint.  This is
an adaptation signal, not a replacement for supervised SMILES training.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value


def _mask_modality(
    batch: Mapping[str, Any],
    modality: str,
    mask_fraction: float,
) -> Dict[str, Any]:
    """Return a partially masked input view without reading structure labels."""
    view = _clone(dict(batch))
    encoder_input = view.get("encoder_input", {})
    if modality not in encoder_input:
        return view
    masks = view.get("encoder_modality_pad_masks")
    pad_mask = masks.get(modality) if isinstance(masks, dict) else None
    if not isinstance(pad_mask, torch.Tensor):
        raise KeyError(f"Missing pad mask for modality {modality!r}.")
    valid = ~pad_mask
    selected = (torch.rand(valid.shape, device=valid.device) < mask_fraction) & valid
    if not selected.any() and valid.any():
        selected.flatten()[valid.flatten().nonzero()[0]] = True
    value = encoder_input[modality]
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, torch.Tensor):
                masked = item.clone()
                masked[selected] = 0
                value[key] = masked
    elif isinstance(value, torch.Tensor):
        value[selected] = 0
    masks[modality] = pad_mask | selected
    availability = view.get("encoder_modality_availability")
    if isinstance(availability, dict) and modality in availability:
        availability[modality] = (~masks[modality]).any(dim=0)
    return view


def _drop_modality(batch: Mapping[str, Any], modality: str) -> Dict[str, Any]:
    """Remove one encoder modality from a cloned batch."""
    view = _clone(dict(batch))
    for key in (
        "encoder_input",
        "encoder_modality_pad_masks",
        "encoder_modality_availability",
    ):
        values = view.get(key)
        if isinstance(values, dict):
            values.pop(modality, None)
    return view


def formula_composition_targets(
    formulas: Iterable[object],
    element_order: Optional[Iterable[str]] = None,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Parse formulae into normalized element-count targets for the auxiliary task."""
    import re

    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    rows = []
    elements = set()
    for formula in formulas:
        text = "" if formula is None else str(formula).strip()
        row = {}
        matches = list(pattern.finditer(text))
        if "".join(match.group(0) for match in matches) == text:
            for match in matches:
                element, count = match.groups()
                row[element] = row.get(element, 0.0) + float(count or 1)
                elements.add(element)
        rows.append(row)
    order = tuple(element_order) if element_order is not None else tuple(sorted(elements))
    target = torch.zeros((len(rows), len(order)), dtype=torch.float32)
    indices = {element: index for index, element in enumerate(order)}
    for row_index, row in enumerate(rows):
        for element, count in row.items():
            # The calibration vocabulary is learned from training formulas. A
            # test-only element must not crash TTT or silently expand the
            # already calibrated Formula head; represent it as an unknown
            # (zero contribution), matching the retrieval encoder behavior.
            column = indices.get(element)
            if column is not None:
                target[row_index, column] = count
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    return target, order


def _prediction_loss(
    original: Mapping[str, Any],
    augmented: Mapping[str, Any],
    *,
    consistency_weight: float,
    modality_weight: float,
) -> torch.Tensor:
    """Compute symmetric probability/logit consistency without structure labels."""
    original_fp = original["fused_fingerprint_logits"]
    augmented_fp = augmented["fused_fingerprint_logits"]
    loss = F.smooth_l1_loss(torch.sigmoid(augmented_fp), torch.sigmoid(original_fp).detach())
    modality_losses = []
    augmented_tasks = augmented.get("modality_fingerprint_logits", {})
    augmented_availability = augmented.get("modality_availability", {})
    fused_reference = torch.sigmoid(original_fp).detach()
    for name, logits in augmented_tasks.items():
        mask = augmented_availability.get(name)
        if not isinstance(mask, torch.Tensor) or mask.any():
            modality_losses.append(
                F.smooth_l1_loss(torch.sigmoid(logits), fused_reference)
            )
    if modality_losses:
        loss = loss + modality_weight * torch.stack(modality_losses).mean()
    return consistency_weight * loss


def self_supervised_ttt_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    mask_modalities: Iterable[str] = ("MSMS", "HNMR", "CNMR", "IR"),
    consistency_weight: float = 1.0,
    modality_weight: float = 0.5,
    mask_fraction: float = 0.15,
    formula_head: Optional[nn.Module] = None,
    formula_targets: Optional[torch.Tensor] = None,
    formula_weight: float = 0.5,
) -> torch.Tensor:
    """Build a label-free loss from original and masked input views."""
    with torch.no_grad():
        original = model.predict_multitask_retrieval(batch)
    losses = []
    for modality in mask_modalities:
        masked_batch = _mask_modality(batch, modality, mask_fraction)
        if modality not in batch.get("encoder_input", {}):
            continue
        try:
            augmented = model.predict_multitask_retrieval(masked_batch)
        except ValueError:
            continue
        losses.append(
            _prediction_loss(
                original,
                augmented,
                consistency_weight=consistency_weight,
                modality_weight=modality_weight,
            )
        )
    if formula_head is not None and formula_targets is not None:
        formula_free = model.predict_multitask_retrieval(
            _drop_modality(batch, "Formula")
        )
        formula_prediction = formula_head(
            torch.sigmoid(formula_free["fused_fingerprint_logits"])
        )
        losses.append(
            formula_weight
            * F.smooth_l1_loss(
                formula_prediction,
                formula_targets.to(
                    device=formula_prediction.device,
                    dtype=formula_prediction.dtype,
                ),
            )
        )
    if not losses:
        raise ValueError("No available modality can produce a masked self-supervised view.")
    return torch.stack(losses).mean()


def _encoder_modules(model: nn.Module) -> list[nn.Module]:
    """Return the unique input embedding and encoder modules of a wrapper."""
    modules: list[nn.Module] = []
    hf_model = getattr(model, "hf_model", None)
    candidates = [
        getattr(model, "multimodal_embedding", None),
        getattr(hf_model, "embedding", None),
        getattr(hf_model, "encoder", None),
        getattr(getattr(hf_model, "model", None), "encoder", None),
    ]
    seen: set[int] = set()
    for candidate in candidates:
        if isinstance(candidate, nn.Module) and id(candidate) not in seen:
            modules.append(candidate)
            seen.add(id(candidate))
    return modules


def configure_ttt_parameters(
    model: nn.Module,
    scope: str = "all",
) -> list[nn.Parameter]:
    """Select parameters updated by supervised retrieval TTT.

    ``all`` preserves the historical behavior. ``encoder`` freezes the
    decoder and output head, while ``encoder_layernorm`` makes the smallest
    possible affine adaptation to the multimodal embedding and encoder.
    """
    normalized_scope = str(scope).strip().lower()
    valid_scopes = {"all", "encoder", "encoder_layernorm"}
    if normalized_scope not in valid_scopes:
        raise ValueError(
            f"Unknown TTT trainable scope {scope!r}; expected one of {sorted(valid_scopes)}."
        )

    if normalized_scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        encoder_modules = _encoder_modules(model)
        if normalized_scope == "encoder":
            for module in encoder_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        else:
            for encoder_module in encoder_modules:
                for module in encoder_module.modules():
                    if isinstance(module, nn.LayerNorm):
                        for parameter in module.parameters(recurse=False):
                            parameter.requires_grad_(True)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(
            f"The model has no parameters available for TTT scope {normalized_scope!r}."
        )
    return trainable


def project_parameter_delta_(
    parameters: Iterable[nn.Parameter],
    reference: Iterable[torch.Tensor],
    max_l2: float = 0.0,
) -> float:
    """Project an adaptation update into an L2 ball around its reference."""
    parameter_list = list(parameters)
    reference_list = list(reference)
    if len(parameter_list) != len(reference_list):
        raise ValueError("Parameters and reference snapshots must have equal length.")
    if max_l2 < 0:
        raise ValueError("max_l2 must be non-negative.")

    with torch.no_grad():
        squared_l2 = sum(
            (parameter.detach().float() - initial.detach().float()).square().sum()
            for parameter, initial in zip(parameter_list, reference_list)
        )
        update_l2 = float(squared_l2.sqrt().cpu())
        if max_l2 > 0 and update_l2 > max_l2:
            scale = max_l2 / update_l2
            for parameter, initial in zip(parameter_list, reference_list):
                parameter.copy_(initial + (parameter - initial) * scale)
            return float(max_l2)
    return update_l2


def enable_ttt_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Freeze the model and expose only encoder LayerNorm affine parameters."""
    return configure_ttt_parameters(model, scope="encoder_layernorm")


@contextmanager
def restored_ttt_parameters(model: nn.Module, parameters: Iterable[nn.Parameter]) -> Iterator[None]:
    """Restore adapted parameters after one test sample."""
    saved = [parameter.detach().clone() for parameter in parameters]
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in zip(parameters, saved):
                parameter.copy_(value)


def _adapt_inplace(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    steps: int = 3,
    lr: float = 1e-4,
    mask_modalities: Iterable[str] = ("MSMS", "HNMR", "CNMR", "IR"),
    consistency_weight: float = 1.0,
    modality_weight: float = 0.5,
    mask_fraction: float = 0.15,
    formula_head: Optional[nn.Module] = None,
    formula_targets: Optional[torch.Tensor] = None,
    formula_weight: float = 0.5,
    parameters: Optional[list[nn.Parameter]] = None,
) -> tuple[list[nn.Parameter], float]:
    """Adapt LayerNorms for one sample while leaving them adapted."""
    parameters = parameters or enable_ttt_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    model.eval()
    last_loss = 0.0
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        loss = self_supervised_ttt_loss(
            model,
            batch,
            mask_modalities=mask_modalities,
            consistency_weight=consistency_weight,
            modality_weight=modality_weight,
            mask_fraction=mask_fraction,
            formula_head=formula_head,
            formula_targets=formula_targets,
            formula_weight=formula_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return parameters, last_loss


@contextmanager
def adapted_sample(
    model: nn.Module,
    batch: Mapping[str, Any],
    **kwargs: Any,
) -> Iterator[float]:
    """Adapt one sample, yield its loss for generation, then restore the model."""
    parameters = enable_ttt_parameters(model)
    with restored_ttt_parameters(model, parameters):
        _, loss = _adapt_inplace(model, batch, parameters=parameters, **kwargs)
        yield loss


def adapt_one_sample(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    steps: int = 3,
    lr: float = 1e-4,
    mask_modalities: Iterable[str] = ("MSMS", "HNMR", "CNMR", "IR"),
    consistency_weight: float = 1.0,
    modality_weight: float = 0.5,
    mask_fraction: float = 0.15,
    formula_head: Optional[nn.Module] = None,
    formula_targets: Optional[torch.Tensor] = None,
    formula_weight: float = 0.5,
) -> float:
    """Adapt one sample in place; callers should use ``restored_ttt_parameters``."""
    _, loss = _adapt_inplace(
        model,
        batch,
        steps=steps,
        lr=lr,
        mask_modalities=mask_modalities,
        consistency_weight=consistency_weight,
        modality_weight=modality_weight,
        mask_fraction=mask_fraction,
        formula_head=formula_head,
        formula_targets=formula_targets,
        formula_weight=formula_weight,
    )
    return loss

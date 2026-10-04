"""Label-free model and router utilities for the isolated sample-wise TTT path.

This module intentionally contains no candidate sampling, reward ranking, or
policy-gradient code. It is kept separate from the historical batch-wise
runner so the strict continual protocol has one auditable dependency path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
import run_appmb as base
from analytical_fm.data.datamodules import MultiModalDataCollator
from analytical_fm.modeling.modality_subset_router import (
    ACTION_SUBSETS,
    ROUTED_MODALITIES,
    ModalitySubsetRouter,
    action_index,
)
from analytical_fm.modeling.pmgfa_shift_features import SourceShiftStatistics


ADAPTATION_KEYS = frozenset(
    {
        "encoder_input",
        "encoder_pad_mask",
        "encoder_modality_pad_masks",
        "encoder_modality_availability",
        "input_formulas",
        "source_row_indices",
    }
)
FORBIDDEN_ADAPTATION_KEYS = frozenset(
    {
        "target",
        "target_mask",
        "target_smiles",
        "smiles",
        "canonical_smiles",
        "inchi",
        "inchikey",
    }
)


def adaptation_view(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist observable fields before any online router or loss call."""
    leaked = set(batch).intersection(FORBIDDEN_ADAPTATION_KEYS)
    if leaked:
        raise ValueError(f"target-label fields reached adaptation: {sorted(leaked)}")
    output = {key: value for key, value in batch.items() if key in ADAPTATION_KEYS}
    if "encoder_input" not in output or "input_formulas" not in output:
        raise ValueError("adaptation batch is missing encoder inputs or observed Formula")
    return output


class EncoderOnlyCollator:
    """Collate only observable encoder modalities from raw rows."""

    def __init__(self, collator: MultiModalDataCollator) -> None:
        self.collator = collator
        self.input_modalities = tuple(collator.input_modalities)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        values = {
            name: [sample[name] for sample in samples] for name in self.input_modalities
        }
        encoder_input, global_pad, modality_pad = self.collator.prepare_encoder_input(
            values, self.collator.return_tensors
        )
        return adaptation_view(
            {
                "encoder_input": encoder_input,
                "encoder_pad_mask": global_pad,
                "encoder_modality_pad_masks": modality_pad,
                "encoder_modality_availability": {
                    name: (~mask).any(dim=0) for name, mask in modality_pad.items()
                },
                "input_formulas": [str(value) for value in values["Formula"]],
                "source_row_indices": [int(sample["source_row_index"]) for sample in samples],
            }
        )


def resolve_parquet(path: Path) -> Path:
    if path.is_file():
        return path
    preferred = path / "test.parquet"
    if preferred.exists():
        return preferred
    parquet_files = sorted(path.glob("*.parquet"))
    if len(parquet_files) != 1:
        raise ValueError(f"cannot resolve one target parquet under {path}")
    return parquet_files[0]


def build_model(
    args: Any,
    data_config: dict[str, Any],
    preprocessors: dict[str, Any],
    device: torch.device,
    *,
    trainable: bool,
):
    model = base._build_model(args, data_config, preprocessors, device)
    if model.multitask_retrieval_heads is not None:
        raise RuntimeError("sample-wise direct pseudo-reward requires the generation model")
    model.excluded_input_modalities = frozenset()
    model.requires_grad_(trainable)
    return model


def move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def decode_groups(model: Any, sequences: torch.Tensor, width: int) -> list[list[str]]:
    decoded = model.target_tokenizer.batch_decode(
        sequences.detach().cpu(), skip_special_tokens=True
    )
    if len(decoded) % width:
        raise RuntimeError("generated sequence count is not divisible by group width")
    return [decoded[index : index + width] for index in range(0, len(decoded), width)]


def autocast(args: Any, device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and args.precision == "bf16",
    )


def availability_from_batch(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    values = batch["encoder_modality_availability"]
    return torch.stack(
        [values[name].to(device=device, dtype=torch.bool) for name in ROUTED_MODALITIES],
        dim=1,
    )


def model_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def observable_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(str(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def relative_drift(student: Any, teacher: Any) -> float:
    numerator = torch.zeros((), device=next(student.parameters()).device)
    denominator = torch.zeros_like(numerator)
    with torch.no_grad():
        for current, reference in zip(student.parameters(), teacher.parameters()):
            numerator += (current.float() - reference.float()).square().sum()
            denominator += reference.float().square().sum()
    return float((numerator.sqrt() / denominator.sqrt().clamp_min(1e-12)).cpu())


def router_decision(
    mode: str,
    features: RouterFeatureBatch | None,
    diagnostics: Mapping[str, torch.Tensor],
    availability: torch.Tensor,
    *,
    router: ModalitySubsetRouter | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = availability.shape[0]
    if mode == "fixed_fc":
        action = action_index(("CNMR",))
        if not availability[:, ROUTED_MODALITIES.index("CNMR")].all():
            raise ValueError("fixed FC teacher requires CNMR for every row")
        selected = torch.full((batch_size,), action, dtype=torch.long, device=availability.device)
        confidence = torch.ones(batch_size, device=availability.device)
        probabilities = torch.zeros(batch_size, len(ACTION_SUBSETS), device=availability.device)
        probabilities[:, action] = 1.0
        return selected, probabilities, confidence
    if mode == "random":
        action_mask = torch.tensor(
            [[name in subset for name in ROUTED_MODALITIES] for subset in ACTION_SUBSETS],
            dtype=torch.bool,
            device=availability.device,
        )
        valid = (~action_mask[None] | availability[:, None]).all(dim=-1)
        probabilities = valid.float() / valid.float().sum(dim=1, keepdim=True)
        selected = torch.multinomial(
            probabilities.cpu(), 1, generator=generator
        ).squeeze(1).to(availability.device)
        return selected, probabilities, torch.ones(batch_size, device=availability.device)
    if mode == "learned":
        if router is None or features is None:
            raise ValueError("learned routing requires source statistics and a router checkpoint")
        router.eval()
        with torch.no_grad():
            selected, probabilities = router.select(features)
        confidence = probabilities.gather(1, selected[:, None]).squeeze(1)
        return selected, probabilities, confidence
    raise ValueError(f"unknown router mode: {mode}")


def load_router(args: Any, device: torch.device):
    router = None
    if args.router_mode == "learned":
        if args.router_checkpoint is None:
            raise ValueError("--router-checkpoint is required for learned routing")
        payload = torch.load(args.router_checkpoint, map_location="cpu", weights_only=False)
        if not payload.get("source_validation_accepted", False):
            raise ValueError("router checkpoint did not pass source validation acceptance")
        router = ModalitySubsetRouter.from_checkpoint(payload["router"]).to(device).eval()
    source = None
    if args.source_stats is not None:
        payload = torch.load(args.source_stats, map_location="cpu", weights_only=False)
        source = SourceShiftStatistics.from_mapping(payload)
    if args.router_mode == "learned" and source is None:
        raise ValueError("learned routing requires --source-stats")
    return router, source

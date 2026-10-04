#!/usr/bin/env python
"""Evaluate Stage-1/Stage-2 checkpoints under matched modality combinations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from datasets import load_dataset
from rdkit import Chem

# Chemotion-compatible preprocessors were built by a standalone script, so
# older pickles record their proxy classes under ``__main__``.  Register the
# classes before unpickling to keep those artifacts loadable from this CLI.
import __main__
from tools import build_chemotion_compatible_preprocessor as _chemotion_compat

for _name in (
    "ChemotionFormulaPreprocessor",
    "ChemotionCarbonPreprocessor",
    "ChemotionMultipletPreprocessor",
    "ChemotionMSMSTextPreprocessor",
):
    setattr(__main__, _name, getattr(_chemotion_compat, _name))

import run_batchwise_pseudoreward_tta as base


COMBOS = {
    "F": frozenset(("Formula",)),
    "FH": frozenset(("Formula", "HNMR")),
    "FC": frozenset(("Formula", "CNMR")),
    "FHI": frozenset(("Formula", "HNMR", "IR")),
    "FM": frozenset(("Formula", "MSMS")),
    "FI": frozenset(("Formula", "IR")),
    "FHC": frozenset(("Formula", "HNMR", "CNMR")),
    "FHCIR": frozenset(("Formula", "HNMR", "CNMR", "IR")),
    "FHCI": frozenset(("Formula", "HNMR", "CNMR", "IR")),
    "FHCMS": frozenset(("Formula", "HNMR", "CNMR", "MSMS")),
    "FCH": frozenset(("Formula", "CNMR", "HNMR")),
    "FCM": frozenset(("Formula", "CNMR", "MSMS")),
    "FCI": frozenset(("Formula", "CNMR", "IR")),
    "FCHM": frozenset(("Formula", "CNMR", "HNMR", "MSMS")),
    "FCHI": frozenset(("Formula", "CNMR", "HNMR", "IR")),
    "FCMI": frozenset(("Formula", "CNMR", "MSMS", "IR")),
    "FULL": frozenset(("Formula", "CNMR", "HNMR", "MSMS", "IR")),
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--preprocessor", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--stage1-checkpoint", type=Path, required=True)
    p.add_argument("--stage2-checkpoint", type=Path, required=True)
    p.add_argument("--single-checkpoint", type=Path)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--beams", type=int, default=10)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument(
        "--combo-names",
        default="",
        help="Comma-separated subset of combination names to evaluate; empty means all.",
    )
    return p.parse_args()


def canon(value: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(value).replace(" ", "").strip())
        return None if mol is None else Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def load_state(model: Any, path: Path, *, base_checkpoint: bool = False) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not base_checkpoint and isinstance(payload, dict) and "student_state_dict" in payload:
        state = payload["student_state_dict"]
    model.load_state_dict(state, strict=True)


def metric(rows: list[list[str]], targets: list[str]) -> dict[str, int | float]:
    if len(rows) != len(targets):
        raise ValueError(
            f"prediction/target length mismatch: {len(rows)} != {len(targets)}"
        )
    target = [canon(x) for x in targets]
    hits = {}
    for k in (1, 5, 10):
        count = 0
        for pred, tgt in zip(rows, target):
            if tgt is not None and any(canon(x) == tgt for x in pred[:k]):
                count += 1
        hits[f"top{k}"] = count
        hits[f"top{k}_rate"] = count / max(1, len(targets))
    return {"n": len(targets), **hits}


def main() -> None:
    a = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = SimpleNamespace(
        data_path=a.data_path, preprocessor=a.preprocessor, checkpoint=a.base_checkpoint,
        model_config=a.model_config, data_config=a.data_config, batch_size=a.batch_size,
        num_workers=a.num_workers, beams=a.beams, limit=a.limit, precision=a.precision,
    )
    data_config, preprocessors, raw, loader = base._build_input_only_loader(common)
    target_col = next(str(v["column"]) for v in data_config.values() if v.get("target", False))
    targets = [str(x) for x in load_dataset("parquet", data_files=str(a.data_path), split="train", columns=[target_col])[target_col]]
    targets = targets[: len(raw)]
    input_modalities = frozenset(name for name, v in data_config.items() if not v.get("target", False))
    if a.combo_names.strip():
        requested = [value.strip() for value in a.combo_names.split(",") if value.strip()]
        unknown = sorted(set(requested).difference(COMBOS))
        if unknown:
            raise ValueError(f"unknown combination names: {unknown}; choices={sorted(COMBOS)}")
        selected_combos = {name: COMBOS[name] for name in requested}
    else:
        selected_combos = COMBOS
    results: dict[str, Any] = {
        "num_samples": len(targets),
        "beams": a.beams,
        "precision": a.precision,
        "combinations": list(selected_combos),
        "stages": {},
    }
    checkpoints = (
        (("single", a.single_checkpoint),)
        if a.single_checkpoint is not None
        else (("stage1", a.stage1_checkpoint), ("stage2", a.stage2_checkpoint))
    )
    if a.save_predictions:
        results["predictions"] = {}
    for stage_name, ckpt in checkpoints:
        model = base._build_model(common, data_config, preprocessors, device, trainable=False).eval()
        load_state(model, ckpt)
        stage: dict[str, Any] = {}
        for combo_name, keep in selected_combos.items():
            excluded = frozenset(input_modalities.difference(keep))
            model.excluded_input_modalities = excluded
            predictions: list[list[str]] = []
            for raw_batch in loader:
                batch = base._move(raw_batch, device)
                decoded = base._ordinary_generate_with_sequences(common, model, batch, beams=a.beams)[0]
                predictions.extend(decoded)
            stage[combo_name] = metric(predictions, targets)
            if a.save_predictions:
                results["predictions"].setdefault(stage_name, {})[combo_name] = predictions
        results["stages"][stage_name] = stage
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "output": str(a.output), "n": len(targets)}), flush=True)


if __name__ == "__main__":
    main()

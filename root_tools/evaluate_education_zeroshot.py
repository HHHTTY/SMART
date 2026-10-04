#!/usr/bin/env python3
"""Evaluate the official checkpoint on the Chemical Education parquet."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

from analytical_fm.data.datasets import build_dataset_multimodal
import run_appmb as appmb
from analytical_fm.utils import clean_sample, seed_everything


def canonical_smiles(value: str) -> str | None:
    try:
        molecule = Chem.MolFromSmiles(value)
        return Chem.MolToSmiles(molecule) if molecule is not None else None
    except Exception:
        return None


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                sub_key: sub_value.to(device) if isinstance(sub_value, torch.Tensor) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def update_metrics(rows: list[dict], beams: int) -> dict:
    count = len(rows)
    summary = {"count": count, "beams": beams}
    for k in (1, 5, 10):
        k = min(k, beams)
        raw = canonical = valid = 0
        for row in rows:
            predictions = row["predictions"][:k]
            target = row["target"]
            target_canonical = row["target_canonical"]
            raw += target in predictions
            canonical += target_canonical is not None and target_canonical in {
                prediction for prediction in (canonical_smiles(p) for p in predictions) if prediction is not None
            }
            valid += any(canonical_smiles(prediction) is not None for prediction in predictions)
        summary[f"raw_top{k}"] = raw / count if count else 0.0
        summary[f"canonical_top{k}"] = canonical / count if count else 0.0
        summary[f"valid_smiles_top{k}"] = valid / count if count else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--beams", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--precision", choices=("32", "16-mixed", "bf16-mixed"), default="16-mixed")
    parser.add_argument(
        "--legacy-generation-normalization",
        action="store_true",
        help="Apply the generation-config normalization used by the earlier education evaluator.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Optional encoder modality subset, e.g. Formula Carbon.",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    seed_everything()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    data_config, preprocessors = appmb._load_preprocessors(args.preprocessor)
    data_config = appmb._merge_preprocessor_metadata(
        appmb._load_yaml(args.data_config), data_config
    )
    data_config, dataset = build_dataset_multimodal(
        data_config,
        data_path=str(args.data),
        cv_split=0,
        splitting="test_only",
        augment_config=None,
        num_cpu=2,
        mixture_config=None,
    )
    tokenizer = preprocessors["Smiles"]
    tokenizer_audit = {
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "rebuilt_smiles_vocab_size": None,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    args.data_config = args.data_config or (args.root / "configs/data/multimodal/paper_multitask_ms_generation.yaml")
    args.model_config = args.model_config or (args.root / "configs/model/custom_model_paper_multitask_ms.yaml")
    args.num_workers = 2
    model = appmb._build_model(args, data_config, preprocessors, device)
    available = set(dataset["train"].column_names)
    requested = set(args.modalities or available)
    missing = requested.difference(available)
    if missing:
        raise ValueError(f"Requested modalities are absent from data: {sorted(missing)}")
    model.excluded_input_modalities = frozenset(
        name
        for name, metadata in data_config.items()
        if not metadata.get("target", False) and name not in requested
    )
    generation_config_audit = {}
    for name, target in (("wrapper", model), ("hf_model", model.hf_model)):
        generation_config = target.generation_config
        if args.legacy_generation_normalization:
            generation_config = generation_config.from_dict(generation_config.to_dict())
            generation_config.forced_bos_token_id = None
            generation_config.no_repeat_ngram_size = 0
            generation_config.early_stopping = False
            target.generation_config = generation_config
        generation_config.num_beams = args.beams
        generation_config.num_return_sequences = args.beams
        generation_config_audit[name] = generation_config.to_dict()

    from analytical_fm.data.datamodules import MultiModalDataModule

    data_module = MultiModalDataModule(
        dataset=dataset,
        preprocessors=preprocessors,
        data_config=data_config,
        model_type=model.model_type,
        batch_size=args.batch_size,
        num_workers=2,
    )
    loader = data_module.val_dataloader()
    if args.precision == "16-mixed":
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
    elif args.precision == "bf16-mixed":
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    rows: list[dict] = []
    total = len(loader.dataset) if hasattr(loader, "dataset") else None
    with torch.inference_mode(), autocast_context:
        for batch_index, batch in enumerate(loader, 1):
            target_smiles = list(batch["target_smiles"])
            if args.limit and len(rows) >= args.limit:
                break
            take = min(len(target_smiles), args.limit - len(rows)) if args.limit else len(target_smiles)
            batch_device = move_batch(batch, device)
            generated = model.generate(batch_device, n_beams=args.beams)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for index, target in enumerate(target_smiles[:take]):
                predictions = [
                    clean_sample(value, canonicalise=False)
                    for value in decoded[index * args.beams : (index + 1) * args.beams]
                ]
                rows.append(
                    {
                        "index": len(rows),
                        "target": clean_sample(target, canonicalise=False),
                        "target_canonical": canonical_smiles(clean_sample(target, canonicalise=False)),
                        "predictions": predictions,
                        "prediction_canonical": [canonical_smiles(value) for value in predictions],
                    }
                )
            if batch_index % 2 == 0:
                print(f"evaluated {len(rows)}" + (f"/{total}" if total else ""), flush=True)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": None,
        "checkpoint_global_step": None,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "data": str(args.data),
        "precision": args.precision,
        "limit": args.limit,
        "input_modalities": args.modalities,
        "tokenizer": tokenizer_audit,
        "legacy_generation_normalization": args.legacy_generation_normalization,
        "generation_config": generation_config_audit,
        "metrics": update_metrics(rows, args.beams),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate predictions for the routed SDBS500 TTT checkpoint by modality view.

This inference pass only loads observable spectral inputs. Ground-truth SMILES
are intentionally left for a separate, post-inference scoring step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import __main__
from tools import build_chemotion_compatible_preprocessor as _chemotion_compat

for _name in (
    "ChemotionFormulaPreprocessor",
    "ChemotionCarbonPreprocessor",
    "ChemotionMultipletPreprocessor",
    "ChemotionMSMSTextPreprocessor",
):
    setattr(__main__, _name, getattr(_chemotion_compat, _name))

import run_batchwise_pseudoreward_tta as routed


VIEWS = {
    "C": frozenset(("Formula", "CNMR")),
    "H": frozenset(("Formula", "HNMR")),
    "MS": frozenset(("Formula", "MSMS")),
    "IR": frozenset(("Formula", "IR")),
    "H_C": frozenset(("Formula", "HNMR", "CNMR")),
    "H_C_MS": frozenset(("Formula", "HNMR", "CNMR", "MSMS")),
    "H_C_IR": frozenset(("Formula", "HNMR", "CNMR", "IR")),
    "ALL": frozenset(("Formula", "HNMR", "CNMR", "MSMS", "IR")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", default=",".join(VIEWS))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def load_student_state(model: Any, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "student_state_dict" not in payload:
        raise ValueError(f"not a Stage 2 student checkpoint: {checkpoint}")
    model.load_state_dict(payload["student_state_dict"], strict=True)
    return payload


def run(args: argparse.Namespace) -> None:
    if args.limit < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("limit, batch size, and max-new-tokens must be positive")
    requested = [value.strip() for value in args.views.split(",") if value.strip()]
    unknown = sorted(set(requested).difference(VIEWS))
    if unknown or not requested:
        raise ValueError(f"unknown/empty views {unknown}; choices={list(VIEWS)}")
    if len(set(requested)) != len(requested):
        raise ValueError("views must be unique")

    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TTT checkpoint evaluation")
    device = torch.device("cuda")
    common = SimpleNamespace(
        data_path=args.data_path,
        preprocessor=args.preprocessor,
        checkpoint=args.base_checkpoint,
        model_config=args.model_config,
        data_config=args.data_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        beams=1,
        limit=args.limit,
        precision=args.precision,
        sample_manifest_json=None,
        stream_order_json=None,
    )
    data_config, preprocessors, raw, loader = routed._build_input_only_loader(common)
    model = routed._build_model(common, data_config, preprocessors, device, trainable=False)
    payload = load_student_state(model, args.student_checkpoint)
    model.eval()
    model.generation_config.max_new_tokens = args.max_new_tokens
    input_modalities = frozenset(
        name for name, metadata in data_config.items() if not metadata.get("target", False)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifests = []
    for view_name in requested:
        output = args.output_dir / f"{view_name.lower()}_predictions.jsonl"
        manifest_path = output.with_suffix(".manifest.json")
        if output.exists() or manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite existing prediction: {output}")
        keep = VIEWS[view_name]
        if not keep.issubset(input_modalities):
            raise ValueError(f"view {view_name} requests unavailable inputs: {keep - input_modalities}")
        model.excluded_input_modalities = frozenset(input_modalities.difference(keep))
        view_started = time.time()
        written = 0
        with output.open("w", encoding="utf-8") as handle:
            for raw_batch in loader:
                batch = routed._move(routed.adaptation_view(raw_batch), device)
                with routed._autocast(common, device):
                    decoded, _sequences = routed._ordinary_generate_with_sequences(
                        common, model, batch, beams=1
                    )
                row_ids = [int(value) for value in raw_batch["source_row_indices"]]
                if len(row_ids) != len(decoded):
                    raise RuntimeError("prediction count does not match source rows")
                for row_id, hypotheses in zip(row_ids, decoded):
                    handle.write(json.dumps({
                        "source_row_index": row_id,
                        "prediction": str(hypotheses[0]),
                        "view": view_name,
                        "checkpoint": str(args.student_checkpoint),
                    }) + "\n")
                    written += 1
        if written != len(raw):
            raise RuntimeError(f"{view_name} wrote {written} predictions for {len(raw)} inputs")
        manifest = {
            "complete": True,
            "view": view_name,
            "retained_modalities": sorted(keep),
            "excluded_modalities": sorted(model.excluded_input_modalities),
            "num_rows": written,
            "beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "checkpoint": str(args.student_checkpoint),
            "checkpoint_epoch": payload.get("epoch"),
            "prediction_jsonl": str(output),
            "ground_truth_loaded": False,
            "elapsed_seconds": time.time() - view_started,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests.append(manifest)
        print(json.dumps({"view": view_name, "num_rows": written,
                          "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)

    summary = {
        "complete": True,
        "data_path": str(args.data_path),
        "num_rows": len(raw),
        "views": manifests,
        "ground_truth_loaded": False,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "complete.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"complete": True, "num_rows": len(raw),
                      "views": requested, "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    run(parse_args())

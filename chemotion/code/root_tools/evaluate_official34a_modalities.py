#!/usr/bin/env python
"""Evaluate one official 34a checkpoint across H/C/IR modality combinations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

try:
    import runner
except ModuleNotFoundError:
    import run_official34a_router_ttt as runner


COMBINATIONS = {
    "H": ("Multiplets",),
    "C": ("Carbon",),
    "HC": ("Multiplets", "Carbon"),
    "IR": ("IR",),
    "HIR": ("Multiplets", "IR"),
    "CIR": ("Carbon", "IR"),
    "HCIR": ("Multiplets", "Carbon", "IR"),
}


def evaluate(model, dataloader, device, precision: str, keep: tuple[str, ...]):
    total = top1 = top5 = top10 = valid = 0
    with torch.inference_mode():
        for batch in dataloader:
            batch = runner.move(batch, device)
            tokens = runner.generate(model, batch, keep, 10, precision).cpu()
            candidates = runner.decoded_beams(model, tokens, 10)
            for choices, target in zip(candidates, batch["target_smiles"]):
                target_can = runner.canonical(target)
                candidate_can = [runner.canonical(value) for value in choices]
                valid += int(candidate_can[0] is not None)
                total += 1
                top1 += int(target_can in candidate_can[:1])
                top5 += int(target_can in candidate_can[:5])
                top10 += int(target_can in candidate_can[:10])
    return {
        "count": total,
        "top1": top1 / max(total, 1),
        "top5": top5 / max(total, 1),
        "top10": top10 / max(total, 1),
        "valid_top1": valid / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--education-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--seed", type=int, default=3247)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    temp, data_config, preprocessors, education = runner.education_dataset(
        args.root, args.preprocessor, args.education_data, args.workers
    )
    try:
        education = education.add_column("row_id", list(range(len(education))))
        dataloader = runner.loader(
            education,
            data_config,
            preprocessors,
            batch_size=args.batch_size,
            shuffle=False,
            workers=args.workers,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = runner.make_model(data_config, preprocessors, args.root, lr=8e-6).to(device)
        runner.load_checkpoint(model, args.checkpoint)
        model.eval()
        args.output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for label, modalities in COMBINATIONS.items():
            metrics = evaluate(model, dataloader, device, args.precision, modalities)
            results[label] = metrics
            print(json.dumps({"combination": label, **metrics}), flush=True)
        results["evaluation"] = {
            "checkpoint": str(args.checkpoint),
            "beam_size": 10,
            "seed": args.seed,
            "precision": args.precision,
        }
        (args.output_dir / "modality_metrics.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    main()

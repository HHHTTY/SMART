#!/usr/bin/env python3
"""Score matched zero-shot and TTT predictions on the locked SDBS500 split."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


VIEWS = {
    "C": ("c_500_prediction/generated_predictions.jsonl", "c_predictions.jsonl"),
    "H": ("h_500_prediction/generated_predictions.jsonl", "h_predictions.jsonl"),
    "MS": ("ms_500_prediction/generated_predictions.jsonl", "ms_predictions.jsonl"),
    "IR": ("ir_500_prediction/generated_predictions.jsonl", "ir_predictions.jsonl"),
    "H_C": ("c_h_500_prediction/generated_predictions.jsonl", "h_c_predictions.jsonl"),
    "H_C_MS": ("c_h_ms_500_prediction/generated_predictions.jsonl", "h_c_ms_predictions.jsonl"),
    "H_C_IR": ("c_h_ir_500_prediction.jsonl", "h_c_ir_predictions.jsonl"),
    "ALL": ("c_h_ms_ir_500_prediction.jsonl", "all_predictions.jsonl"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 500:
        raise ValueError(f"expected 500 rows in {path}, got {len(rows)}")
    return rows


def extract_smiles(value: Any) -> str:
    match = re.search(r"##SMILES:\s*([^\s<]+)", str(value))
    return match.group(1) if match else ""


def compact_ttt(value: Any) -> str:
    return "".join(str(value).split())


def write_pairs(path: Path, predictions: list[str], labels: list[str]) -> None:
    if len(predictions) != 500 or len(labels) != 500:
        raise ValueError("paired score inputs must both contain 500 rows")
    with path.open("w", encoding="utf-8") as handle:
        for prediction, label in zip(predictions, labels):
            row = {"predict": f"##SMILES: {prediction}", "label": f"##SMILES: {label}"}
            handle.write(json.dumps(row) + "\n")


def metric_subset(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": payload["n"],
        "validity": payload["validity"],
        **payload["paper_metrics_all_rows_invalid_as_zero"],
        "mces_timeout_fallback_rows": payload["mces_timeout_fallback_rows"],
        "fraggle_timeout_fallback_rows": payload["fraggle_timeout_fallback_rows"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot-root", type=Path, required=True)
    parser.add_argument("--ttt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "n": 500,
        "checkpoint": "stage2_epoch_10.pt",
        "generated_max_new_tokens": 1024,
        "metrics": {},
    }
    for view, (zero_rel, ttt_name) in VIEWS.items():
        zero_rows = read_jsonl(args.zero_shot_root / zero_rel)
        ttt_rows = read_jsonl(args.ttt_root / ttt_name)
        zero_preds: list[str] = []
        ttt_preds: list[str] = []
        labels: list[str] = []
        for row_index, (zero, ttt) in enumerate(zip(zero_rows, ttt_rows)):
            if "source_row_index" in ttt and int(ttt["source_row_index"]) != row_index:
                raise ValueError(f"TTT row order mismatch in {view}: {row_index}")
            label = extract_smiles(zero.get("label", ""))
            if not label:
                raise ValueError(f"missing zero-shot target label in {view}, row {row_index}")
            zero_preds.append(extract_smiles(zero.get("predict", "")))
            ttt_preds.append(compact_ttt(ttt.get("prediction", "")))
            labels.append(label)

        view_result: dict[str, Any] = {}
        for system, predictions in (("zero_shot", zero_preds), ("ttt", ttt_preds)):
            pair_path = args.output_dir / f"{view.lower()}_{system}_paired.jsonl"
            metrics_path = args.output_dir / f"{view.lower()}_{system}_metrics.json"
            if args.resume and metrics_path.is_file():
                pass
            else:
                write_pairs(pair_path, predictions, labels)
                subprocess.run(
                    [
                        "python",
                        str(args.scorer),
                        str(pair_path),
                        str(metrics_path),
                        "--workers",
                        str(args.workers),
                    ],
                    check=True,
                )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            view_result[system] = metric_subset(metrics)
        view_result["delta_ttt_minus_zero_shot"] = {
            key: view_result["ttt"][key] - view_result["zero_shot"][key]
            for key in (
                "validity",
                "tanimoto_ecfp4",
                "cosine",
                "mces_edge_distance_lower_better",
                "functional_group",
                "tanimoto_maccs",
                "fraggle",
            )
        }
        results["metrics"][view] = view_result
        print(json.dumps({"view": view, **view_result}), flush=True)

    summary_path = args.output_dir / "paired_metrics.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "output": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()

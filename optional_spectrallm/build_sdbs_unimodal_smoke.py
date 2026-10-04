#!/usr/bin/env python3
"""Build four SpectraLLM single-modality smoke-test JSONL files."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


def load_converter(path: Path):
    spec = importlib.util.spec_from_file_location("sdbs_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()
    conv = load_converter(args.converter)
    frame = pd.read_parquet(args.input, columns=[
        "smiles", "canonical_smiles", "h_nmr_peaks", "c_nmr_peaks", "ms_peaks",
        "h_nmr_raw_json", "c_nmr_raw_json", "ms_raw_json", "ir_1800", "ir_valid_mask",
    ]).head(args.max_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = {
        "hnmr": lambda row: conv.h_spec(row, 0.10),
        "cnmr": lambda row: conv.c_spec(row, 0.10),
        "ir": lambda row: conv.ir_spec(row, 0.03, 0.10, 60),
        "ms": lambda row: conv.ms_spec(row, 0.10, 80),
    }
    for mode, make_spec in modes.items():
        out = args.output_dir / f"{mode}_test32.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for _, row in frame.iterrows():
                smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
                spec = make_spec(row)
                record = {
                    "system": conv.SYSTEM,
                    "prompt": (
                        "Given " + spec + ". Based on the information provided, predict "
                        "which compound the spectra correspond to and give the SMILES of "
                        "that compound. Please answer strictly in the format ##SMILES: ."
                    ),
                    "response": "##SMILES: " + smiles,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(mode, out, sum(1 for _ in out.open(encoding="utf-8")))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a paper-format, high-to-low 1H NMR SpectraLLM probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd


def load_converter(path: Path):
    spec = importlib.util.spec_from_file_location("sdbs_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reverse_h_spec(spec: str) -> str:
    pattern = r'("H-shifts"\s*:\s*")([^"]*)("\s*,\s*"Intensities"\s*:\s*")([^"]*)(")'
    match = re.search(pattern, spec)
    if not match:
        return spec
    shifts = ",".join(reversed(match.group(2).split(",")))
    intensities = ",".join(reversed(match.group(4).split(",")))
    replacement = match.group(1) + shifts + match.group(3) + intensities + match.group(5)
    return spec[:match.start()] + replacement + spec[match.end():]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conv = load_converter(args.converter)
    frame = pd.read_parquet(args.input, columns=["canonical_smiles", "smiles", "h_nmr_raw_json", "h_nmr_peaks"]).head(32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for _, row in frame.iterrows():
            smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
            spec = reverse_h_spec(conv.h_spec(row, 0.01))
            prompt = (
                "Given " + spec + ", the spectra data includes the Chemical Shift "
                "positions in ppm as H-shifts and corresponding intensities as "
                "Intensities. Based on the information provided, predict which "
                "compound the spectra correspond to and give the SMILES of that "
                "compound. Please answer strictly in the format ##SMILES: ."
            )
            handle.write(json.dumps({
                "system": conv.SYSTEM,
                "prompt": prompt,
                "response": "##SMILES: " + smiles,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

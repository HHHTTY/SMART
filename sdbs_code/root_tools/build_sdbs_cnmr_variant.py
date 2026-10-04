#!/usr/bin/env python3
"""Build a 13C-only SpectraLLM probe with a selectable peak threshold."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--paper-single-prompt", action="store_true")
    parser.add_argument("--descending", action="store_true")
    args = parser.parse_args()
    conv = load_converter(args.converter)
    frame = pd.read_parquet(args.input, columns=["canonical_smiles", "smiles", "c_nmr_raw_json", "c_nmr_peaks"]).head(32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for _, row in frame.iterrows():
            smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
            spec = conv.c_spec(row, args.threshold)
            if args.descending:
                # SpectraLLM's released MSD prompts list NMR shifts high-to-low.
                pattern = r'("C-shifts"\s*:\s*")([^"]*)("\s*,\s*"Intensities"\s*:\s*")([^"]*)(")'
                match = re.search(pattern, spec)
                if match:
                    shifts = ",".join(reversed(match.group(2).split(",")))
                    intensities = ",".join(reversed(match.group(4).split(",")))
                    replacement = match.group(1) + shifts + match.group(3) + intensities + match.group(5)
                    spec = spec[:match.start()] + replacement + spec[match.end():]
            if args.paper_single_prompt:
                prompt = (
                    "Given " + spec + ", the spectra data includes the Chemical Shift "
                    "positions in ppm as C-shifts and corresponding intensities as "
                    "Intensities. Based on the information provided, predict which "
                    "compound the spectra correspond to and give the SMILES of that "
                    "compound. Please answer strictly in the format ##SMILES: ."
                )
            else:
                prompt = (
                    "Given " + spec + ". Based on the information provided, predict "
                    "which compound the spectra correspond to and give the SMILES of "
                    "that compound. Please answer strictly in the format ##SMILES: ."
                )
            handle.write(json.dumps({
                "system": conv.SYSTEM,
                "prompt": prompt,
                "response": "##SMILES: " + smiles,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

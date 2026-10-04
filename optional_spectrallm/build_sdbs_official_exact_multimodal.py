#!/usr/bin/env python3
"""Build full C/H/IR/MS prompts using SpectraLLM's original notebook format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_sdbs_official_exact import (
    SYSTEM,
    generate_spec_str,
    ir_peaks,
    peak_table,
)


def record(row: pd.Series) -> dict[str, str]:
    c = generate_spec_str(peak_table(row["c_nmr_raw_json"], x_index=0, y_index=1), "cnmr", False)
    h = generate_spec_str(peak_table(row["h_nmr_raw_json"], x_index=1, y_index=2), "hnmr", False)
    ir = generate_spec_str(ir_peaks(row), "ir", False)
    ms = generate_spec_str(peak_table(row["ms_raw_json"], x_index=0, y_index=1, reverse=False), "msms", False)
    specs = ", ".join([c, h, ir, ms])
    prompt = (
        "Given multiple spectra, they are " + specs + ". All of these spectra are determined by the same compound,  "
        "with the wavenumber postions in reciprocal centimeters as Wavenumbers, the energy postions in eV as Energies "
        "and corresponding intensities as Intensities. Based on the information provided by these spectra, predict which "
        "compound the spectra correspond to and give the SMILES of that compound. Please answer strictly in the format ##SMILES: ."
    )
    smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
    return {"system": SYSTEM, "prompt": prompt, "response": "##SMILES: " + smiles}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=32)
    args = ap.parse_args()
    cols = ["smiles", "canonical_smiles", "h_nmr_raw_json", "c_nmr_raw_json", "ms_raw_json", "ir_1800"]
    frame = pd.read_parquet(args.input, columns=cols)
    if args.max_samples > 0:
        frame = frame.head(args.max_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for _, row in frame.iterrows():
            f.write(json.dumps(record(row), ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(frame), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

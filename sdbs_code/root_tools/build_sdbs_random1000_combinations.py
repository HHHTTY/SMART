#!/usr/bin/env python3
"""Build identical random SDBS subsets for several SpectraLLM modality combinations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_sdbs_official_exact import generate_spec_str, ir_peaks, peak_table, SYSTEM


COMBINATIONS = {
    "c_h_ms_ir": ["cnmr", "hnmr", "msms", "ir"],
    "c_h_ir": ["cnmr", "hnmr", "ir"],
    "c_h_ms": ["cnmr", "hnmr", "msms"],
    "c_h": ["cnmr", "hnmr"],
    "c": ["cnmr"],
    "h": ["hnmr"],
    "ms": ["msms"],
    "ir": ["ir"],
}

PROMPT_SUFFIX = (
    ". All of these spectra are determined by the same compound,  with the wavenumber "
    "postions in reciprocal centimeters as Wavenumbers, the energy postions in eV as "
    "Energies and corresponding intensities as Intensities. Based on the information "
    "provided by these spectra, predict which compound the spectra correspond to and give "
    "the SMILES of that compound. Please answer strictly in the format ##SMILES: ."
)


def spectrum(row: pd.Series, mode: str) -> str:
    if mode == "cnmr":
        peaks = peak_table(row["c_nmr_raw_json"], x_index=0, y_index=1)
    elif mode == "hnmr":
        peaks = peak_table(row["h_nmr_raw_json"], x_index=1, y_index=2)
    elif mode == "msms":
        peaks = peak_table(row["ms_raw_json"], x_index=0, y_index=1, reverse=False)
    elif mode == "ir":
        peaks = ir_peaks(row)
    else:
        raise ValueError(mode)
    # generate_all_prompt uses describe=False for each component.
    return generate_spec_str(peaks, mode, False)


def make_record(row: pd.Series, modes: list[str]) -> dict[str, str]:
    specs = ", ".join(spectrum(row, mode) for mode in modes)
    prompt = "Given multiple spectra, they are " + specs + PROMPT_SUFFIX
    smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
    return {"system": SYSTEM, "prompt": prompt, "response": "##SMILES: " + smiles}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--sample-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()
    cols = ["global_row_id", "sdbs_no", "smiles", "canonical_smiles", "h_nmr_raw_json", "c_nmr_raw_json", "ms_raw_json", "ir_1800"]
    frame = pd.read_parquet(args.input, columns=cols)
    if args.sample_size > len(frame):
        raise ValueError(f"sample-size {args.sample_size} > rows {len(frame)}")
    sampled = frame.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input": str(args.input),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "global_row_ids": [int(x) for x in sampled["global_row_id"]],
        "sdbs_no": [str(x) for x in sampled["sdbs_no"]],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, modes in COMBINATIONS.items():
        path = args.output_dir / f"{name}_test{args.sample_size}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for _, row in sampled.iterrows():
                handle.write(json.dumps(make_record(row, modes), ensure_ascii=False) + "\n")
        print(name, path, len(sampled))


if __name__ == "__main__":
    main()

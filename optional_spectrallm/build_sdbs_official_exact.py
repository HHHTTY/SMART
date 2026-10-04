#!/usr/bin/env python3
"""Build SDBS prompts matching SpectraLLM's released MSD notebook.

The notebook serializes extracted peak dictionaries with generate_spec_str:
values below 0.1 are removed, values are rounded to two decimals, and the
default json.dumps spacing is retained.  SDBS supplies peak tables for H/C/MS
and a continuous 1800-point IR array; the latter is processed with the exact
find_peaks/peak_widths helper used by the notebook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


SYSTEM = (
    "You are a chemist. Given the description of spectras extracted from a compound, "
    "you are skilled at analyzing this spectra to infer whether the compound contains "
    "certain fragments or functional groups, and then combining information to deduce "
    "which compound these spectra correspond to, and accurately provide the SMILES of "
    "the compound."
)


def extract_peaks_info_width(x: Any, y: Any, *, prominence: float | None = None) -> list[dict[str, float]]:
    """The notebook's helper, with only the arguments used by its preprocessing cells."""
    x = np.asarray(x)
    y = np.asarray(y)
    peaks, _ = find_peaks(y, prominence=prominence)
    widths_result = peak_widths(y, peaks, rel_height=0.5)
    left_ips, right_ips = widths_result[2], widths_result[3]
    out: list[dict[str, float]] = []
    for i, peak in enumerate(peaks):
        left = np.interp(left_ips[i], np.arange(len(x)), x)
        right = np.interp(right_ips[i], np.arange(len(x)), x)
        out.append({
            "position": float(x[peak]),
            "intensity": float(y[peak]),
            "width": float(right - left),
        })
    return out


def parse_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def peak_table(value: Any, *, x_index: int, y_index: int, reverse: bool = True) -> list[dict[str, float]]:
    pairs: list[tuple[float, float]] = []
    for item in parse_json(value):
        if not isinstance(item, (list, tuple)) or len(item) <= max(x_index, y_index):
            continue
        try:
            x, y = float(item[x_index]), float(item[y_index])
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            pairs.append((x, y))
    if not pairs:
        return []
    maximum = max(y for _, y in pairs)
    if maximum > 0:
        pairs = [(x, y / maximum) for x, y in pairs]
    # The official NMR peak extraction follows a descending ppm axis.
    if reverse:
        pairs = list(reversed(pairs))
    return [{"position": x, "intensity": y, "width": 0.0} for x, y in pairs]


def generate_spec_str(spec_data: list[dict[str, float]], type_: str, describe: bool = True) -> str:
    x_list: list[str] = []
    y_list: list[str] = []
    for spec in spec_data:
        y = spec["intensity"]
        if y < 0.1:
            continue
        x_list.append(str(round(spec["position"], 2)))
        y_list.append(str(round(y, 2)))
    if type_ == "cnmr":
        obj = {"C-shifts": ",".join(x_list), "Intensities": ",".join(y_list)}
        text = "Carbon-13 Nuclear Magnetic Resonance " + json.dumps(obj)
        return text + ", the spectra data includes the Chemical Shift positions in ppm as C-shifts and corresponding intensities as Intensities" if describe else text
    if type_ == "hnmr":
        obj = {"H-shifts": ",".join(x_list), "Intensities": ",".join(y_list)}
        text = "Proton Nuclear Magnetic Resonance " + json.dumps(obj)
        return text + ", the spectra data includes the Chemical Shift positions in ppm as H-shifts and corresponding intensities as Intensities" if describe else text
    if type_ == "ir":
        obj = {"Wavenumbers": ",".join(x_list), "Intensities": ",".join(y_list)}
        text = "Infrared Spectrum " + json.dumps(obj)
        return text + ", the spectra data includes the wavenumber positions in reciprocal centimeters as Wavenumbers and corresponding intensities as Intensities" if describe else text
    if type_ == "msms":
        obj = {"mzs": ",".join(x_list), "Intensities": ",".join(y_list)}
        text = "Mass spectrum data " + json.dumps(obj)
        return text + ", the spectra data includes mass-to-charge ratios (m/z) as mzs and their corresponding relative intensities as Intensities." if describe else text
    raise ValueError(type_)


def ir_peaks(row: pd.Series) -> list[dict[str, float]]:
    values = np.asarray(parse_json(row["ir_1800"]), dtype=float)
    if values.size != 1800 or np.all(values == 0):
        return []
    # The notebook explicitly uses this decreasing axis. No baseline removal,
    # renormalization, peak truncation, or width field is done during prompting.
    # SDBS stores the resampled grid in ascending 400 -> 4000 cm^-1 order;
    # the paper notebook's axis is descending 4000 -> 400, so reverse the
    # signal before applying the notebook helper to keep x/y paired.
    values = values[::-1]
    wavenumbers = np.linspace(4000, 400, len(values))
    return extract_peaks_info_width(wavenumbers, values, prominence=0.03)


def record(row: pd.Series, type_: str) -> dict[str, str]:
    if type_ == "hnmr":
        peaks = peak_table(row["h_nmr_raw_json"], x_index=1, y_index=2)
    elif type_ == "cnmr":
        peaks = peak_table(row["c_nmr_raw_json"], x_index=0, y_index=1)
    elif type_ == "msms":
        # SDBS is EI-MS, but this preserves its supplied m/z ordering for the
        # requested smoke test; the paper's training source calls this msms.
        peaks = peak_table(row["ms_raw_json"], x_index=0, y_index=1, reverse=False)
    elif type_ == "ir":
        peaks = ir_peaks(row)
    else:
        raise ValueError(type_)
    spec = generate_spec_str(peaks, type_)
    prompt = (
        f"Given {spec}. Based on the information provided, predict which compound "
        "the spectra correspond to and give the SMILES of that compound. "
        "Please answer strictly in the format ##SMILES: ."
    )
    smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
    return {"system": SYSTEM, "prompt": prompt, "response": "##SMILES: " + smiles}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=32)
    args = ap.parse_args()
    cols = ["smiles", "canonical_smiles", "h_nmr_raw_json", "c_nmr_raw_json", "ms_raw_json", "ir_1800"]
    frame = pd.read_parquet(args.input, columns=cols).head(args.max_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for type_ in ("hnmr", "cnmr", "ir", "msms"):
        out = args.output_dir / f"{type_}_test{len(frame)}_official_exact.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for _, row in frame.iterrows():
                f.write(json.dumps(record(row, type_), ensure_ascii=False) + "\n")
        print(type_, out, "rows", len(frame))


if __name__ == "__main__":
    main()

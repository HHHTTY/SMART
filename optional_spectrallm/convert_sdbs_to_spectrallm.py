#!/usr/bin/env python3
"""Convert SDBS_final_ir1800 parquet rows to SpectraLLM JSONL prompts.

The official SpectraLLM MSD preprocessing represents each spectrum as a
small list of peaks inside a text prompt.  This converter keeps that contract
while using SDBS's measured H/C/MS peak tables and its 1800-point IR grid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


SYSTEM = (
    "You are a chemist. Given the description of spectras extracted from a compound, "
    "you are skilled at analyzing this spectra to infer whether the compound contains "
    "certain fragments or functional groups, and then combining those information to deduce "
    "which compound these spectra correspond to, and accurately provide the SMILES of "
    "the compound."
)
PROMPT_SUFFIX = (
    ". All of these spectra are determined by the same compound,  with the wavenumber "
    "postions in reciprocal centimeters as Wavenumbers, the energy postions in eV as "
    "Energies and corresponding intensities as Intensities. Based on the information "
    "provided by these spectra, predict which compound the spectra correspond to and "
    "give the SMILES of that compound. Please answer strictly in the format ##SMILES: ."
)


def parse_json(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return value.tolist() if isinstance(value, np.ndarray) else value
    if value is None:
        return []
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def finite_pairs(value: Any) -> list[tuple[float, float]]:
    raw = parse_json(value)
    result: list[tuple[float, float]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            result.append((x, y))
    return result


def normalize_peaks(
    pairs: list[tuple[float, float]],
    *,
    min_relative: float,
    max_peaks: int = 0,
) -> list[tuple[float, float]]:
    if not pairs:
        return []
    maximum = max(y for _, y in pairs)
    if not math.isfinite(maximum) or maximum <= 0:
        return []
    normalized = [(x, max(0.0, y) / maximum) for x, y in pairs]
    normalized = [(x, y) for x, y in normalized if y >= min_relative]
    if max_peaks and len(normalized) > max_peaks:
        normalized = sorted(normalized, key=lambda item: (-item[1], item[0]))[:max_peaks]
    return sorted(normalized, key=lambda item: item[0])


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def spec_text(name: str, fields: dict[str, str]) -> str:
    # Match the repository's prompt generator, which uses json.dumps default spacing.
    return f"{name} " + json.dumps(fields, ensure_ascii=False)


def h_spec(row: pd.Series, min_relative: float) -> str:
    # Raw SDBS H rows are [frequency, ppm, intensity].
    pairs = [(item[1], item[2]) for item in parse_json(row.get("h_nmr_raw_json"))
             if isinstance(item, (list, tuple)) and len(item) >= 3]
    pairs = normalize_peaks(pairs, min_relative=min_relative)
    if not pairs:
        raw = parse_json(row.get("h_nmr_peaks"))
        pairs = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, dict):
                try:
                    ppm = float(item.get("centroid", item.get("rangeMax")))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(ppm):
                    pairs.append((ppm, 1.0))
    return spec_text(
        "Proton Nuclear Magnetic Resonance",
        {"H-shifts": ",".join(fmt(x) for x, _ in pairs),
         "Intensities": ",".join(fmt(y) for _, y in pairs)},
    )


def c_spec(row: pd.Series, min_relative: float) -> str:
    pairs = normalize_peaks(finite_pairs(row.get("c_nmr_raw_json")), min_relative=min_relative)
    if not pairs:
        raw = parse_json(row.get("c_nmr_peaks"))
        pairs = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, dict):
                try:
                    ppm = float(item.get("delta (ppm)"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(ppm):
                    pairs.append((ppm, 1.0))
    return spec_text(
        "Carbon-13 Nuclear Magnetic Resonance",
        {"C-shifts": ",".join(fmt(x) for x, _ in pairs),
         "Intensities": ",".join(fmt(y) for _, y in pairs)},
    )


def ms_spec(row: pd.Series, min_relative: float, max_peaks: int) -> str:
    pairs = normalize_peaks(
        finite_pairs(row.get("ms_raw_json")),
        min_relative=min_relative,
        max_peaks=max_peaks,
    )
    return spec_text(
        "Mass spectrum data",
        {"mzs": ",".join(fmt(x) for x, _ in pairs),
         "Intensities": ",".join(fmt(y) for _, y in pairs)},
    )


def ir_spec(row: pd.Series, prominence: float, min_relative: float, max_peaks: int) -> str:
    values = np.asarray(parse_json(row.get("ir_1800")), dtype=np.float64)
    mask = np.asarray(parse_json(row.get("ir_valid_mask")), dtype=bool)
    if values.size != 1800:
        raise ValueError(f"expected 1800 IR values, got {values.size}")
    if mask.size != values.size:
        mask = np.ones(values.size, dtype=bool)
    # SDBS stores the grid increasing (400 -> 4000); the official MSD
    # preprocessor emits wavenumbers decreasing (4000 -> 400).
    x_full = np.linspace(4000.0, 400.0, values.size)
    values = values[::-1]
    mask = mask[::-1]
    valid = mask & np.isfinite(values)
    if valid.sum() < 8:
        valid = np.isfinite(values)
    x, y = x_full[valid], values[valid]
    peaks, properties = find_peaks(y, prominence=prominence, distance=3)
    if len(peaks) == 0:
        peaks = np.asarray([int(np.argmax(y))])
        properties = {"prominences": np.asarray([float(np.max(y))])}
    intensities = np.asarray(y[peaks], dtype=np.float64)
    baseline = float(np.min(y))
    intensities = np.maximum(intensities - baseline, 0.0)
    peak_max = float(np.max(intensities)) if len(intensities) else 0.0
    if peak_max > 0:
        intensities /= peak_max
    selected = [(float(x[p]), float(v)) for p, v in zip(peaks, intensities) if v >= min_relative]
    selected.sort(key=lambda item: (-item[1], item[0]))
    if max_peaks and len(selected) > max_peaks:
        selected = selected[:max_peaks]
    selected.sort(key=lambda item: item[0], reverse=True)
    # Widths are accepted by the checkpoint's IR prompt format and preserve
    # useful line-shape information without including the 1800 raw samples.
    width_values = peak_widths(y, peaks, rel_height=0.5)[0]
    step = abs(float(np.median(np.diff(x)))) if len(x) > 1 else 1.0
    width_by_position = {float(x[p]): float(w * step) for p, w in zip(peaks, width_values)}
    return spec_text(
        "Infrared Spectrum",
        {"Wavenumbers": ",".join(fmt(x) for x, _ in selected),
         "Intensities": ",".join(fmt(y) for _, y in selected),
         "Widths": ",".join(fmt(width_by_position[x]) for x, _ in selected)},
    )


def make_record(
    row: pd.Series,
    *,
    ir_prominence: float,
    min_relative: float,
    max_ms_peaks: int,
    max_ir_peaks: int,
) -> dict[str, str]:
    specs = [
        c_spec(row, min_relative),
        h_spec(row, min_relative),
        ir_spec(row, ir_prominence, min_relative, max_ir_peaks),
        ms_spec(row, min_relative, max_ms_peaks),
    ]
    smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
    if not smiles:
        raise ValueError("missing canonical_smiles/smiles")
    return {
        "system": SYSTEM,
        "prompt": "Given multiple spectra, they are " + ", ".join(specs) + PROMPT_SUFFIX,
        "response": "##SMILES: " + smiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--ir-prominence", type=float, default=0.03)
    parser.add_argument("--min-relative", type=float, default=0.10)
    parser.add_argument("--max-ms-peaks", type=int, default=80)
    parser.add_argument("--max-ir-peaks", type=int, default=60)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    args = parser.parse_args()
    columns = [
        "smiles", "canonical_smiles", "h_nmr_peaks", "c_nmr_peaks", "ms_peaks",
        "h_nmr_raw_json", "c_nmr_raw_json", "ms_raw_json", "ir_1800", "ir_valid_mask",
    ]
    frame = pd.read_parquet(args.input, columns=columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"input_rows": len(frame), "written": 0, "skipped": 0, "prompt_chars": []}
    with args.output.open("w", encoding="utf-8") as handle:
        for index, (_, row) in enumerate(frame.iterrows()):
            if args.max_samples and stats["written"] >= args.max_samples:
                break
            try:
                record = make_record(
                    row, ir_prominence=args.ir_prominence, min_relative=args.min_relative,
                    max_ms_peaks=args.max_ms_peaks, max_ir_peaks=args.max_ir_peaks,
                )
            except (ValueError, TypeError, KeyError) as exc:
                stats["skipped"] += 1
                print(f"skip row {index}: {exc}")
                continue
            chars = len(record["prompt"])
            if args.max_prompt_chars and chars > args.max_prompt_chars:
                stats["skipped"] += 1
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            stats["prompt_chars"].append(chars)
    lengths = np.asarray(stats.pop("prompt_chars"), dtype=float)
    stats["prompt_chars_summary"] = {
        "min": int(lengths.min()) if len(lengths) else None,
        "median": float(np.median(lengths)) if len(lengths) else None,
        "p95": float(np.percentile(lengths, 95)) if len(lengths) else None,
        "max": int(lengths.max()) if len(lengths) else None,
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

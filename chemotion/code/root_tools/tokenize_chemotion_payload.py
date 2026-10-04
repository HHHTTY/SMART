#!/usr/bin/env python3
"""Tokenize Chemotion's integrated payload into the simulated OpenNMT format.

The source payload is read-only.  IR transmittance is converted to the
simulated dataset's absorbance/intensity-like positive-peak representation
before the legacy 400-bin tokenization; this is not physical -log10(T).
Reflectance, unknown, and arbitrary units are kept in the audit but are not
admitted to the main dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds


MODS = (("nmr_1h", "h"), ("nmr_13c", "c"), ("ir", "ir"), ("ms", "ms"))
SPLITS = ("train", "val", "test")
TASK = "T_1H_13C_IR_MS"


def load_legacy(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("legacy_real_tokenizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = value
    else:
        try:
            raw = json.loads(str(value))
        except Exception:
            return []
    out: list[float] = []
    for item in raw or []:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def split_for_key(key: str) -> str:
    value = int(hashlib.sha1(f"chemotion-v1|{key}".encode()).hexdigest()[:16], 16) / 16**16
    return "train" if value < 0.8 else "val" if value < 0.9 else "test"


def clean_line(text: str) -> str:
    return " ".join(str(text).replace("\t", " ").replace("\n", " ").split())


def classify_transmittance(y: np.ndarray) -> str:
    """Classify the stored scale without letting one spike set the unit."""
    if y.size == 0 or not np.isfinite(y).all():
        return "invalid"
    p1, _p50, p99 = np.percentile(y, [1.0, 50.0, 99.0])
    if p1 >= -0.1 and p99 <= 1.25:
        return "fraction"
    if p1 >= -5.0 and p99 <= 105.0:
        return "percent"
    return "out_of_range"


def legacy_ir_400bin_tokens(x: np.ndarray, y: np.ndarray) -> str:
    """The project's legacy 400-bin interpolation/scaling convention."""
    target_x = np.linspace(400.0, 4000.0, num=400)
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if unique_x.size != x_sorted.size:
        summed = np.zeros_like(unique_x, dtype=float)
        counts = np.zeros_like(unique_x, dtype=float)
        np.add.at(summed, inverse, y_sorted)
        np.add.at(counts, inverse, 1.0)
        x_sorted, y_sorted = unique_x, summed / np.maximum(counts, 1.0)
    y_interp = np.interp(target_x, x_sorted, y_sorted)
    y_interp = y_interp + abs(float(np.nanmin(y_interp)))
    max_y = float(np.nanmax(y_interp))
    if max_y <= 0:
        return ""
    values = np.clip(np.rint(y_interp / max_y * 100.0), 0, 100).astype(int)
    return "IR " + " ".join(str(int(value)) for value in values)


def one_h_shifts_plausible(row: dict[str, Any], legacy: Any) -> bool:
    """Reject parser artifacts outside a chemically plausible 1H range."""
    peaks = legacy.parse_json_list(row.get("peaks"))
    shifts: list[float] = []
    for peak in peaks:
        if not isinstance(peak, dict):
            continue
        value = None
        for key in ("ppm", "shift"):
            try:
                value = float(peak.get(key)) if peak.get(key) is not None else None
            except (TypeError, ValueError):
                value = None
            if value is not None:
                break
        if value is None:
            try:
                start = float(peak.get("rangeMin", peak.get("start")))
                end = float(peak.get("rangeMax", peak.get("end")))
                value = (start + end) / 2.0
            except (TypeError, ValueError):
                continue
        if math.isfinite(value):
            shifts.append(value)
    return bool(shifts) and all(-1.0 <= value <= 16.0 for value in shifts)


def ir_tokens(row: dict[str, Any], legacy: Any) -> tuple[str, str]:
    """Return legacy IR tokens plus the applied representation label."""
    x = np.asarray(parse_list(row.get("x_values")), dtype=float)
    y = np.asarray(parse_list(row.get("intensities")), dtype=float)
    if x.size < 16 or y.size != x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        return "", "invalid_shape"
    unit = str(row.get("x_unit") or "").strip().upper().replace("−", "-")
    if "CM" not in unit:
        return "", "non_cm_inverse"
    lo, hi = float(np.min(x)), float(np.max(x))
    overlap = max(0.0, min(hi, 4000.0) - max(lo, 400.0))
    if hi - lo < 2000.0 or overlap < 2000.0:
        return "", "insufficient_wavenumber_range"
    y_type = str(row.get("y_type") or "").strip().upper()
    if y_type == "TRANSMITTANCE":
        scale = classify_transmittance(y)
        if scale in {"out_of_range", "invalid"}:
            return "", "transmittance_out_of_range"
        # Match simulated/SDBS preprocessing: convert transmittance to an
        # absorbance/intensity-like positive deviation before legacy scaling.
        if scale == "fraction":
            y = 1.0 - np.clip(y, 0.0, 1.0)
        else:
            y = 100.0 - np.clip(y, 0.0, 100.0)
        semantic = f"transmittance_{scale}_to_complement"
    elif y_type == "ABSORBANCE":
        if float(np.min(y)) < -1.0 or float(np.max(y)) > 5.0:
            return "", "absorbance_out_of_range"
        semantic = "absorbance_native"
    else:
        return "", f"excluded_y_type:{y_type or 'MISSING'}"
    # Interpolation and scaling happen after T -> A conversion, so valleys in
    # transmittance become peaks in the model's absorbance convention.
    text = legacy_ir_400bin_tokens(x, y)
    return text, semantic if text else "empty_after_conversion"


def choose_row(rows: list[dict[str, Any]], modality: str, legacy: Any) -> tuple[dict[str, Any] | None, str]:
    """Choose one deterministic observation for a record/modality."""
    candidates: list[tuple[tuple[Any, ...], dict[str, Any], str]] = []
    for row in rows:
        if modality == "h" and not one_h_shifts_plausible(row, legacy):
            continue
        if modality == "ir":
            block, reason = ir_tokens(row, legacy)
        else:
            args = type("Args", (), {"max_c_peaks": 256, "ms_top_k": 256})()
            block, stats = legacy.modality_tokens(modality, row, args)
            if not stats.get("valid") or not block:
                continue
            reason = "valid"
        if not block or reason.startswith("excluded_y_type:"):
            continue
        smiles = str(row.get("canonical_smiles") or "").strip()
        inchikey = str(row.get("inchikey") or "").strip()
        # Prefer an explicit structure, then richer tokenized observations;
        # lexical tie-breaking makes the result reproducible across parquet
        # partition order.
        rank = (0 if smiles else 1, -len(block), inchikey, smiles, str(row.get("observation_id") or ""))
        candidates.append((rank, {**row, "_block": block, "_reason": reason}, reason))
    if not candidates:
        return None, "no_valid_observation"
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--legacy-script", type=Path, required=True)
    parser.add_argument("--no-split", action="store_true", help="write one src.txt/tgt.txt pair")
    args = parser.parse_args()
    legacy = load_legacy(args.legacy_script)

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    audit = Counter()
    for dataset_name, modality in MODS:
        path = args.payload_root / dataset_name
        # The integrated payload contains partitions where optional columns
        # are null in one file and strings in another; use the project's
        # promoted-schema helper so Arrow does not attempt a null->string cast.
        dataset = legacy.dataset_with_promoted_schema(path)
        columns = [
            "observation_id", "canonical_smiles", "inchikey", "source_dataset",
            "source_record_id", "source_subdataset", "payload_status", "source_observation_ordinal",
            "peaks", "raw_text", "x_unit", "y_type", "x_values", "intensities",
            "mz_values", "relative_intensities", "spectrum_type", "ms_level", "polarity",
            "precursor_mz", "adduct", "collision_energy", "ms_category", "ms_level_inferred",
            "collision_energy_raw", "collision_energy_kind", "collision_energy_unit",
            "fragmentation_mode", "solvent", "frequency_mhz",
        ]
        columns = [name for name in columns if name in dataset.schema.names]
        for batch in dataset.to_batches(columns=columns, batch_size=2048):
            for row in batch.to_pylist():
                if str(row.get("source_dataset") or "") != "Chemotion_IR_PENDING_AUTH":
                    continue
                if modality == "ir":
                    y_type = str(row.get("y_type") or "MISSING").strip().upper()
                    audit[f"ir_input_y_type:{y_type}"] += 1
                    x_values = parse_list(row.get("x_values"))
                    intensities = parse_list(row.get("intensities"))
                    if len(x_values) != len(intensities) or len(x_values) < 16:
                        audit["ir_input_shape_invalid"] += 1
                    else:
                        x_lo, x_hi = min(x_values), max(x_values)
                        overlap = max(0.0, min(x_hi, 4000.0) - max(x_lo, 400.0))
                        range_ok = x_hi - x_lo >= 2000.0 and overlap >= 2000.0
                        if not range_ok:
                            audit["ir_input_insufficient_wavenumber_range"] += 1
                        if y_type == "TRANSMITTANCE":
                            scale = classify_transmittance(np.asarray(intensities, dtype=float))
                            if scale == "fraction":
                                audit["ir_input_transmittance_fraction"] += 1
                            elif scale == "percent":
                                audit["ir_input_transmittance_percent"] += 1
                            else:
                                audit["ir_input_transmittance_out_of_range"] += 1
                record_id = str(row.get("source_record_id") or "").strip()
                if not record_id:
                    audit[f"missing_record_id:{modality}"] += 1
                    continue
                grouped[record_id][modality].append(row)
                audit[f"payload_rows:{modality}"] += 1

    out_data = args.out_root / "opennmt" / TASK / "data"
    out_data.mkdir(parents=True, exist_ok=True)
    handles = {}
    output_splits = ("all",) if args.no_split else SPLITS
    for split in output_splits:
        src_name = "src.txt" if args.no_split else f"src-{split}.txt"
        tgt_name = "tgt.txt" if args.no_split else f"tgt-{split}.txt"
        handles[(split, "src")] = (out_data / src_name).open("w", encoding="utf-8")
        handles[(split, "tgt")] = (out_data / tgt_name).open("w", encoding="utf-8")

    rows_written = Counter()
    target_seen: set[tuple[str, str]] = set()
    split_records: dict[str, list[str]] = defaultdict(list)
    for record_id in sorted(grouped):
        selected: dict[str, dict[str, Any]] = {}
        reasons: dict[str, str] = {}
        for _dataset_name, modality in MODS:
            row, reason = choose_row(grouped[record_id].get(modality, []), modality, legacy)
            if row is None:
                reasons[modality] = reason
            else:
                selected[modality] = row
        if len(selected) != len(MODS):
            audit["incomplete_or_invalid_record"] += 1
            for modality in dict.fromkeys(reasons):
                audit[f"missing_or_invalid:{modality}"] += 1
            continue
        inchikeys = {str(row.get("inchikey") or "").strip() for row in selected.values() if row.get("inchikey")}
        if len(inchikeys) != 1:
            audit["cross_modality_inchikey_conflict"] += 1
            continue
        smiles_values = [str(row.get("canonical_smiles") or "").strip() for row in selected.values()]
        smiles = next((value for value in smiles_values if value), "")
        if not smiles:
            audit["missing_smiles"] += 1
            continue
        try:
            target, _tokens, canonical = legacy.target_for_smiles(smiles)
            formula = legacy.formula_for_smiles(canonical)
        except Exception:
            audit["target_or_formula_error"] += 1
            continue
        key = next(iter(inchikeys)) or legacy.rdkit_inchikey_for_smiles(canonical) or canonical
        dedupe_key = (key, canonical)
        if dedupe_key in target_seen:
            audit["duplicate_key_canonical"] += 1
            continue
        target_seen.add(dedupe_key)
        blocks = " ".join(str(selected[mod]["_block"]) for _dataset_name, mod in MODS)
        source = clean_line(f"{formula} {blocks}")
        target = clean_line(target)
        if not source or not target:
            audit["empty_source_or_target"] += 1
            continue
        split = "all" if args.no_split else split_for_key(key)
        handles[(split, "src")].write(source + "\n")
        handles[(split, "tgt")].write(target + "\n")
        rows_written[split] += 1
        split_records[split].append(record_id)
        audit["complete_records"] += 1
        audit[f"complete_by_subdataset:{str(selected['ir'].get('source_subdataset') or '')}"] += 1

    for handle in handles.values():
        handle.close()
    manifest = {
        "format": "simulated_open_nmt",
        "task": TASK,
        "split": "none" if args.no_split else "inchikey_random_v1",
        "source_dataset": "Chemotion_IR_PENDING_AUTH",
        "payload_root": str(args.payload_root),
        "ir_policy": {
            "TRANSMITTANCE_0_1": "y = 1 - clip(T, 0, 1), matching simulated/SDBS preprocessing",
            "TRANSMITTANCE_0_100": "y = 100 - clip(T, 0, 100), matching simulated/SDBS preprocessing",
            "TRANSMITTANCE_scale_detection": "robust p1/p99 thresholds; isolated spikes do not set the scale",
            "ABSORBANCE": "use native absorbance",
            "REFLECTANCE": "excluded from main dataset",
            "ARBITRARY_UNITS": "excluded from main dataset",
            "UNKNOWN": "excluded from main dataset",
            "grid": "legacy IR 400 bins on 400-4000 cm-1, then 0-100 integer scaling",
        },
        "counts": dict(rows_written),
        "audit": dict(audit),
        "record_ids": {split: split_records[split] for split in output_splits},
    }
    (args.out_root / "opennmt" / TASK / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": dict(rows_written), "audit": dict(audit)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

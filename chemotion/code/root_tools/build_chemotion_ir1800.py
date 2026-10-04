#!/usr/bin/env python3
"""Build a Chemotion Parquet view with the SDBS IR1800 contract.

The source is the integrated Chemotion payload.  H/C/MS selection and text
tokenization are reused from the existing Chemotion OpenNMT pipeline, while
IR is regenerated from the raw x/y arrays on a continuous 1800-point grid.
The representation matches SDBS_final_ir1800's simulated-complement mode:
native absorbance is used directly, and transmittance is converted to an
absorbance-like positive signal before per-spectrum min-shift/max-scaling.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


MODS = (("nmr_1h", "h"), ("nmr_13c", "c"), ("ir", "ir"), ("ms", "ms"))
SOURCE_DATASET = "Chemotion_IR_PENDING_AUTH"
GRID_SIZE = 1800
GRID_START = 400.0
GRID_END = 4000.0
GRID = np.linspace(GRID_START, GRID_END, GRID_SIZE, dtype=np.float64)

# The integrated payload can contain several representations of one 13C
# spectrum.  Prefer explicit peak-picked observations over peak tables parsed
# from the dense JDX trace; the latter can contain hundreds of line-shape
# samples rather than distinct carbon resonances.
C_SUBMODALITY_PRIORITY = {
    "bruker_peaklist_xml": 0,
    "peak_txt": 1,
    "jcamp_peak_table_1d": 2,
    "nmrium_peak_nodes": 3,
    "jcamp_peak_table": 4,
}
# The Chemotion C peak payloads frequently retain the CDCl3 triplet.  SDBS
# peak lists used for checkpoint training do not emit that solvent signal, so
# remove the narrow triplet window before materialising the model field.
C_SOLVENT_TRIPLET_RANGE = (76.45, 77.55)
C_COMMON_C_SOLVENT_RANGES = (
    (38.5, 41.0),  # DMSO-d6
    (76.0, 79.0),  # CDCl3, including the broadened JCAMP triplet
)
C_DENSE_SUBMODALITIES = frozenset(
    {
        "jcamp_peak_table",
        "jcamp_peak_table_1d",
        "nmrium_peak_nodes",
        "nmrium_peak_nodes_1d",
    }
)
H_SUBMODALITY_PRIORITY = {
    "peak_txt": 0,
    "bruker_peaklist_xml": 1,
    "jcamp_peak_table": 2,
    "nmrium_peak_nodes": 3,
    "nmrium_peak_nodes_1d": 4,
    "jcamp_peak_table_1d": 5,
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean_line(text: object) -> str:
    return " ".join(str(text).replace("\t", " ").replace("\n", " ").split())


def parse_json(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def parse_float_list(value: Any) -> list[float]:
    raw = parse_json(value)
    if not isinstance(raw, list):
        return []
    values: list[float] = []
    for item in raw:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _parse_h_block(block: object) -> list[dict[str, Any]]:
    """Convert the legacy Chemotion H block to the model's multiplet schema."""
    text = clean_line(block)
    if text.startswith("1HNMR"):
        text = text[len("1HNMR") :].strip()
    peaks: list[dict[str, Any]] = []
    for segment in text.split("|"):
        tokens = segment.strip().split()
        if len(tokens) < 3:
            continue
        try:
            range_max = float(tokens[0])
            range_min = float(tokens[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(range_max) or not math.isfinite(range_min):
            continue
        category = str(tokens[2] or "m")
        n_h: int | float = 1
        j_values: list[str] = []
        reading_j = False
        for token in tokens[3:]:
            if token == "J":
                reading_j = True
                continue
            if reading_j:
                try:
                    value = float(token)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    j_values.append(f"{value:g}")
                continue
            if token.endswith("H"):
                try:
                    value = float(token[:-1])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    n_h = int(round(value)) if abs(value - round(value)) < 1e-6 else value
        peaks.append(
            {
                "rangeMax": range_max,
                "rangeMin": range_min,
                "centroid": (range_max + range_min) / 2.0,
                "category": category,
                "nH": n_h,
                "j_values": "_".join(j_values) if j_values else "None",
            }
        )
    return peaks


def normalize_h_peaks(value: Any, block: object = "") -> list[dict[str, Any]]:
    peaks = _parse_h_block(block)
    if peaks:
        return peaks
    raw = parse_json(value)
    peaks = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            ppm = float(item.get("ppm", item.get("shift", item.get("centroid"))))
            start = float(item.get("rangeMin", item.get("start", ppm)))
            end = float(item.get("rangeMax", item.get("end", ppm)))
            intensity = float(item.get("intensity", 1.0))
            n_h_raw = item.get("nH", item.get("integration", 1))
            n_h_value = float(n_h_raw)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(number) for number in (ppm, start, end, intensity, n_h_value)):
            continue
        n_h: int | float = int(round(n_h_value)) if abs(n_h_value - round(n_h_value)) < 1e-6 else n_h_value
        peaks.append(
            {
                "rangeMax": max(start, end),
                "rangeMin": min(start, end),
                "centroid": ppm,
                "category": str(item.get("category") or item.get("multiplicity") or "m"),
                "nH": n_h,
                "j_values": str(item.get("j_values") or "None"),
                "intensity": intensity,
            }
        )
    return peaks


def _raw_c_points(value: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    raw = parse_json(value)
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            ppm = float(
                item.get(
                    "ppm",
                    item.get("shift", item.get("delta (ppm)", item.get("delta"))),
                )
            )
            intensity = float(
                item.get("intensity", item.get("height", item.get("y", 1.0)))
            )
        except (TypeError, ValueError):
            continue
        if math.isfinite(ppm) and math.isfinite(intensity) and -10.0 <= ppm <= 230.0:
            points.append((ppm, max(0.0, intensity)))
    return points


def _collapse_c_points(points: list[tuple[float, float]], spacing: float = 0.08) -> list[tuple[float, float]]:
    """Collapse line-shape duplicates while retaining the strongest point."""
    if not points:
        return []
    ordered = sorted(points, key=lambda item: item[0])
    groups: list[list[tuple[float, float]]] = [[ordered[0]]]
    for point in ordered[1:]:
        if point[0] - groups[-1][-1][0] <= spacing:
            groups[-1].append(point)
        else:
            groups.append([point])
    return [max(group, key=lambda item: item[1]) for group in groups]


def _drop_obvious_c_solvents(
    points: list[tuple[float, float]], *, dense: bool
) -> list[tuple[float, float]]:
    """Remove solvent clusters without deleting isolated analyte shifts."""
    if not points:
        return []
    remove: set[int] = set()
    for lo, hi in C_COMMON_C_SOLVENT_RANGES:
        indices = [index for index, (ppm, _intensity) in enumerate(points) if lo <= ppm <= hi]
        # CDCl3 is a very recognizable triplet and occasionally survives as a
        # single already-picked line at the edge of the broadened window.
        is_cdcl3 = (lo, hi) == (76.0, 79.0)
        if is_cdcl3 or dense or len(indices) >= 2:
            remove.update(indices)
    return [point for index, point in enumerate(points) if index not in remove]


def _peak_pick_dense_c(
    points: list[tuple[float, float]],
    max_peaks: int,
    min_spacing: float = 0.18,
) -> list[dict[str, float]]:
    """Pick resonance maxima from dense JCAMP/Bruker line tables.

    The payload stores both real peak lists and dense trace-derived samples.
    Sorting dense samples by ppm is not peak picking: it preferentially keeps
    the low-ppm baseline.  A 0.1-ppm envelope, adaptive noise floor, local
    maxima and a representation-level safety cap produce a discrete list
    compatible with the SDBS C representation. Peak count is not inferred
    from molecular formula because overlap and non-equivalent environments
    make that a lossy assumption.
    """
    points = _drop_obvious_c_solvents(points, dense=True)
    if not points:
        return []
    # Keep one maximum per 0.1-ppm bin before looking for local maxima.  This
    # also removes repeated line-shape rows at effectively identical shifts.
    bins: dict[int, tuple[float, float]] = {}
    for ppm, intensity in points:
        index = int(round(ppm * 10.0))
        previous = bins.get(index)
        if previous is None or intensity > previous[1]:
            bins[index] = (ppm, intensity)
    ordered = [bins[index] for index in sorted(bins)]
    intensities = np.asarray([item[1] for item in ordered], dtype=np.float64)
    maximum = float(intensities.max(initial=0.0))
    if maximum <= 0.0:
        return []
    median = float(np.median(intensities))
    mad = float(np.median(np.abs(intensities - median)))
    threshold = max(median + 6.0 * mad, maximum * 0.01)
    candidates: list[tuple[float, float]] = []
    for index, point in enumerate(ordered):
        intensity = point[1]
        if intensity < threshold:
            continue
        left = ordered[max(0, index - 2) : index]
        right = ordered[index + 1 : index + 3]
        neighbours = [item[1] for item in left + right]
        if neighbours and intensity < max(neighbours):
            continue
        candidates.append(point)
    if not candidates:
        candidates = [max(ordered, key=lambda item: item[1])]

    budget = max(1, max_peaks if max_peaks > 0 else len(candidates))
    # Select the strongest maxima while preventing one broad resonance from
    # contributing several adjacent bins.
    selected: list[tuple[float, float]] = []
    for point in sorted(candidates, key=lambda item: (-item[1], item[0])):
        if any(abs(point[0] - old[0]) < min_spacing for old in selected):
            continue
        selected.append(point)
        if len(selected) >= budget:
            break
    selected.sort(key=lambda item: item[0])
    return [
        {"delta (ppm)": round(ppm, 1), "intensity": float(intensity)}
        for ppm, intensity in selected
    ]


def normalize_c_peaks(
    value: Any,
    block: object = "",
    *,
    submodality: str = "",
    max_peaks: int = 256,
    dense_min_spacing: float = 0.18,
) -> list[dict[str, float]]:
    """Materialize discrete C shifts, peak-picking dense trace payloads."""
    raw_points = _raw_c_points(value)
    normalized_submodality = str(submodality or "").strip().lower()
    dense = normalized_submodality in C_DENSE_SUBMODALITIES
    budget = max_peaks if max_peaks > 0 else 256
    # Representation metadata is authoritative here: an unusually long
    # explicit peak list is still a reported list, not a dense trace.  Only
    # trace-derived submodalities may be peak-picked with the dense spacing
    # policy; otherwise an explicit Bruker/peak_txt list could be silently
    # altered by its row count.
    if raw_points and dense:
        picked = _peak_pick_dense_c(raw_points, max_peaks, dense_min_spacing)
        if picked:
            return picked

    if raw_points:
        raw_points = _drop_obvious_c_solvents(raw_points, dense=False)
        collapsed = _collapse_c_points(raw_points)
        if collapsed:
            collapsed.sort(key=lambda item: item[0])
            return [
                {"delta (ppm)": round(ppm, 1), "intensity": float(intensity)}
                for ppm, intensity in collapsed[:budget]
            ]

    # Last-resort parsing keeps compatibility with older payloads that only
    # expose the already-tokenized block.
    text = clean_line(block)
    fallback: list[dict[str, float]] = []
    if text.startswith("13CNMR"):
        for token in text[len("13CNMR") :].strip().split():
            try:
                ppm = float(token)
            except (TypeError, ValueError):
                continue
            if math.isfinite(ppm) and -10.0 <= ppm <= 230.0:
                fallback.append({"delta (ppm)": round(ppm, 1), "intensity": 1.0})
    if not fallback:
        return []
    filtered = [
        peak
        for peak in fallback
        if not (C_SOLVENT_TRIPLET_RANGE[0] <= peak["delta (ppm)"] <= C_SOLVENT_TRIPLET_RANGE[1])
    ]
    return (filtered or fallback)[:budget]


def normalize_ms_peaks(
    value: Any,
    max_peaks: int = 256,
    mz_values: Any = None,
    relative_intensities: Any = None,
) -> list[list[float]]:
    """Normalize MS peaks with the same max-intensity/top-k policy as export."""
    peaks: list[list[float]] = []
    raw = parse_json(value)
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            mz = float(item.get("mz"))
            if item.get("relative_intensity") is not None:
                intensity = float(item["relative_intensity"])
            else:
                intensity = float(item.get("intensity", 0.0))
                if abs(intensity) <= 1.5:
                    intensity *= 100.0
        except (TypeError, ValueError):
            continue
        if math.isfinite(mz) and math.isfinite(intensity) and mz > 0.0 and intensity >= 0.0:
            peaks.append([mz, intensity])
    if not peaks:
        masses = parse_float_list(mz_values)
        intensities = parse_float_list(relative_intensities)
        for mz, intensity in zip(masses, intensities):
            if math.isfinite(mz) and math.isfinite(intensity) and mz > 0.0 and intensity >= 0.0:
                peaks.append([mz, intensity])
    if not peaks:
        return peaks
    maximum = max(intensity for _, intensity in peaks)
    if maximum <= 0.0:
        return []
    peaks = [[mz, intensity / maximum * 100.0] for mz, intensity in peaks]
    if max_peaks > 0 and len(peaks) > max_peaks:
        peaks = sorted(peaks, key=lambda item: (-item[1], item[0]))[:max_peaks]
    return sorted(peaks, key=lambda item: item[0])


def ms_text(peaks: list[list[float]]) -> str:
    values: list[str] = []
    for mz, intensity in peaks:
        values.extend([f"{mz:.1f}", f"{intensity:.1f}"])
    return " ".join(values)


def compact_formula(formula: str) -> str:
    return "".join(str(formula).split())


def transmittance_scale(y: np.ndarray) -> str:
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError("transmittance is empty or non-finite")
    p99 = float(np.percentile(y, 99.0))
    return "fraction" if p99 <= 1.5 else "percent"


def convert_ir(row: dict[str, Any], tokenizer: Any) -> tuple[dict[str, Any], str]:
    x = np.asarray(parse_float_list(row.get("x_values")), dtype=np.float64)
    y = np.asarray(parse_float_list(row.get("intensities")), dtype=np.float64)
    if x.size < 16 or y.size != x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid IR x/y arrays")
    unit = str(row.get("x_unit") or "").strip().upper().replace("-", "")
    if "CM" not in unit:
        raise ValueError(f"unsupported IR x unit: {row.get('x_unit')!r}")
    x_lo, x_hi = float(x.min()), float(x.max())
    overlap = max(0.0, min(x_hi, GRID_END) - max(x_lo, GRID_START))
    if x_hi - x_lo < 2000.0 or overlap < 2000.0:
        raise ValueError("insufficient IR wavenumber range")

    y_type = str(row.get("y_type") or "").strip().upper()
    clipped_low = clipped_high = 0
    if y_type == "TRANSMITTANCE":
        scale = transmittance_scale(y)
        fraction = y if scale == "fraction" else y / 100.0
        clipped_low = int(np.count_nonzero(fraction < 0.0))
        clipped_high = int(np.count_nonzero(fraction > 1.0))
        signal = 100.0 * (1.0 - fraction)
        transform = f"T_{scale.upper()}_TO_100_MINUS_T_THEN_IR1800"
    elif y_type == "ABSORBANCE":
        if float(y.min()) < -1.0 or float(y.max()) > 5.0:
            raise ValueError("absorbance outside accepted range")
        scale = ""
        signal = y
        transform = "A_DIRECT_THEN_IR1800"
    else:
        raise ValueError(f"excluded IR y type: {y_type or 'MISSING'}")

    finite = np.isfinite(x) & np.isfinite(signal)
    x, signal = x[finite], signal[finite]
    order = np.argsort(x, kind="mergesort")
    x, signal = x[order], signal[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size != x.size:
        summed = np.zeros(unique_x.size, dtype=np.float64)
        counts = np.zeros(unique_x.size, dtype=np.int64)
        np.add.at(summed, inverse, signal)
        np.add.at(counts, inverse, 1)
        x, signal = unique_x, summed / np.maximum(counts, 1)

    grid_signal = np.interp(GRID, x, signal)
    valid_mask = (GRID >= float(x.min())) & (GRID <= float(x.max()))
    shifted = grid_signal + abs(float(grid_signal.min()))
    maximum = float(shifted.max(initial=0.0))
    values = np.zeros(GRID_SIZE, dtype=np.float32) if maximum <= 0.0 else np.clip(
        shifted / maximum, 0.0, 1.0
    ).astype(np.float32)
    metadata = {
        "ir_spectra": values.tolist(),
        "ir_1800": values.tolist(),
        "ir_valid_mask": valid_mask.tolist(),
        "ir_coverage": float(valid_mask.mean()),
        "ir_raw_points": int(x.size),
        "ir_raw_x_min": float(x.min()),
        "ir_raw_x_max": float(x.max()),
        "ir_raw_y_min": float(signal.min()),
        "ir_raw_y_max": float(signal.max()),
        "ir_jdx_member": str(row.get("observation_id") or ""),
        "ir_xunits": str(row.get("x_unit") or ""),
        "ir_yunits": y_type,
        "ir_transform": transform,
        "ir_transmittance_scale": scale,
        "ir_transmittance_clipped_low": clipped_low,
        "ir_transmittance_clipped_high": clipped_high,
    }
    return metadata, transform


def selected_columns(dataset: Any) -> list[str]:
    names = set(dataset.schema.names)
    wanted = [
        "observation_id", "canonical_smiles", "inchikey", "source_dataset",
        "source_record_id", "source_subdataset", "payload_status",
        "source_observation_ordinal", "submodality", "peaks", "raw_text", "x_unit", "y_type",
        "x_values", "intensities", "mz_values", "relative_intensities",
        "spectrum_type", "ms_level", "polarity", "precursor_mz", "adduct",
        "collision_energy", "ms_category", "ms_level_inferred",
        "collision_energy_raw", "collision_energy_kind", "collision_energy_unit",
        "fragmentation_mode", "solvent", "frequency_mhz",
    ]
    return [name for name in wanted if name in names]


def collect_rows(payload_root: Path, legacy: Any, tokenizer: Any) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], Counter[str]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    audit: Counter[str] = Counter()
    for dataset_name, modality in MODS:
        dataset = legacy.dataset_with_promoted_schema(payload_root / dataset_name)
        columns = selected_columns(dataset)
        for batch in dataset.to_batches(columns=columns, batch_size=2048):
            for row in batch.to_pylist():
                if str(row.get("source_dataset") or "") != SOURCE_DATASET:
                    continue
                audit[f"payload_rows:{modality}"] += 1
                record_id = str(row.get("source_record_id") or "").strip()
                if not record_id:
                    audit[f"missing_record_id:{modality}"] += 1
                    continue
                grouped[record_id][modality].append(row)
    return grouped, audit


def choose_c_row_peak_picked(
    rows: list[dict[str, Any]], legacy: Any
) -> tuple[dict[str, Any] | None, str]:
    """Choose a chemically peak-picked 13C observation deterministically.

    The legacy chooser ranks by token-string length.  That is appropriate for
    retaining information in ordinary text observations, but it makes a dense
    JDX-derived line table win over an explicit Bruker peak list.  Keep the
    same validity checks and tie-breaking after applying the representation
    priority above.
    """
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    args = type("Args", (), {"max_c_peaks": 256, "ms_top_k": 256})()
    for row in rows:
        block, stats = legacy.modality_tokens("c", row, args)
        if not stats.get("valid") or not block:
            continue
        submodality = str(row.get("submodality") or "").strip().lower()
        priority = C_SUBMODALITY_PRIORITY.get(submodality, len(C_SUBMODALITY_PRIORITY))
        candidate = {**row, "_block": block, "_reason": "valid"}
        rank = (
            priority,
            -len(block),
            str(row.get("inchikey") or ""),
            str(row.get("canonical_smiles") or ""),
            str(row.get("observation_id") or ""),
        )
        candidates.append((rank, candidate))
    if not candidates:
        return None, "no_valid_observation"
    candidates.sort(key=lambda item: item[0])
    selected = candidates[0][1]
    return selected, f"peak_picked:{str(selected.get('submodality') or 'unknown')}"


def choose_h_row_peak_picked(
    rows: list[dict[str, Any]], legacy: Any, tokenizer: Any
) -> tuple[dict[str, Any] | None, str]:
    """Prefer explicit 1H multiplets over dense trace-derived line tables."""
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    args = type("Args", (), {"max_c_peaks": 256, "ms_top_k": 256})()
    for row in rows:
        if not tokenizer.one_h_shifts_plausible(row, legacy):
            continue
        block, stats = legacy.modality_tokens("h", row, args)
        if not stats.get("valid") or not block:
            continue
        submodality = str(row.get("submodality") or "").strip().lower()
        priority = H_SUBMODALITY_PRIORITY.get(submodality, len(H_SUBMODALITY_PRIORITY))
        peak_mode = str(stats.get("peak_mode") or "")
        explicit = 0 if peak_mode == "reported_multiplets" else 1
        candidate = {**row, "_block": block, "_reason": "peak_picked", "_h_peak_mode": peak_mode}
        rank = (
            priority,
            explicit,
            -len(block),
            str(row.get("inchikey") or ""),
            str(row.get("canonical_smiles") or ""),
            str(row.get("observation_id") or ""),
        )
        candidates.append((rank, candidate))
    if not candidates:
        return None, "no_valid_observation"
    candidates.sort(key=lambda item: item[0])
    selected = candidates[0][1]
    return selected, f"peak_picked:{str(selected.get('submodality') or 'unknown')}:{selected.get('_h_peak_mode', '')}"


def build_rows(
    grouped: dict[str, dict[str, list[dict[str, Any]]]],
    legacy: Any,
    tokenizer: Any,
    audit: Counter[str],
    limit: int | None = None,
    c_selection_policy: str = "existing",
    h_selection_policy: str = "existing",
    dense_min_spacing: float = 0.18,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_seen: set[tuple[str, str]] = set()
    for record_id in sorted(grouped):
        if limit is not None and len(rows) >= limit:
            break
        selected: dict[str, dict[str, Any]] = {}
        structure_selected: dict[str, dict[str, Any]] = {}
        reasons: dict[str, str] = {}
        for modality in ("h", "c", "ir", "ms"):
            existing_row, existing_reason = tokenizer.choose_row(
                grouped[record_id].get(modality, []), modality, legacy
            )
            if existing_row is not None:
                structure_selected[modality] = existing_row
            if modality == "h" and h_selection_policy == "peak_picked":
                row, reason = choose_h_row_peak_picked(grouped[record_id].get(modality, []), legacy, tokenizer)
            elif modality == "c" and c_selection_policy == "peak_picked":
                row, reason = choose_c_row_peak_picked(grouped[record_id].get(modality, []), legacy)
            else:
                row, reason = existing_row, existing_reason
            if row is None:
                reasons[modality] = reason
            else:
                selected[modality] = row
        if len(selected) != 4:
            audit["incomplete_or_invalid_record"] += 1
            for modality in reasons:
                audit[f"missing_or_invalid:{modality}"] += 1
            continue

        inchikeys = {str(row.get("inchikey") or "").strip() for row in selected.values() if row.get("inchikey")}
        if len(inchikeys) != 1:
            audit["cross_modality_inchikey_conflict"] += 1
            continue
        # Keep the structure/target representation from the existing chooser;
        # peak-picked C rows can carry equivalent but differently written
        # metal/aromatic SMILES strings.
        structure_rows = structure_selected if len(structure_selected) == 4 else selected
        smiles = next(
            (
                str(structure_rows[m].get("canonical_smiles") or "").strip()
                for m in ("h", "c", "ir", "ms")
                if structure_rows[m].get("canonical_smiles")
            ),
            "",
        )
        if not smiles:
            audit["missing_smiles"] += 1
            continue
        try:
            target, _tokens, canonical = legacy.target_for_smiles(smiles)
            formula = legacy.formula_for_smiles(canonical)
            ir_meta, _transform = convert_ir(selected["ir"], tokenizer)
        except Exception as error:
            audit[f"conversion_error:{type(error).__name__}"] += 1
            continue

        key = next(iter(inchikeys)) or legacy.rdkit_inchikey_for_smiles(canonical) or canonical
        dedupe_key = (key, canonical)
        if dedupe_key in target_seen:
            audit["duplicate_key_canonical"] += 1
            continue
        target_seen.add(dedupe_key)

        h_block = clean_line(selected["h"]["_block"])
        c_block = clean_line(selected["c"]["_block"])
        ms_block = clean_line(selected["ms"]["_block"])
        h_peaks = normalize_h_peaks(selected["h"].get("peaks"), h_block)
        c_peaks = normalize_c_peaks(
            selected["c"].get("peaks"),
            c_block,
            submodality=str(selected["c"].get("submodality") or ""),
            dense_min_spacing=dense_min_spacing,
        )
        ms_peaks = normalize_ms_peaks(
            selected["ms"].get("peaks"),
            max_peaks=256,
            mz_values=selected["ms"].get("mz_values"),
            relative_intensities=selected["ms"].get("relative_intensities"),
        )
        if not h_peaks or not c_peaks or not ms_peaks:
            audit["normalized_peak_payload_missing"] += 1
            continue
        c_block = "13CNMR " + " ".join(f"{peak['delta (ppm)']:.1f}" for peak in c_peaks)
        spectrum = ms_text(ms_peaks)
        ir_text = "IR " + " ".join(f"{value:.8g}" for value in ir_meta["ir_spectra"])
        source_line = clean_line(f"{formula} {h_block} {c_block} {ir_text} {spectrum}")
        target_line = clean_line(target)
        compact = compact_formula(formula)
        source_subdataset = str(selected["ir"].get("source_subdataset") or "")
        ms_semantics = "/".join(
            str(selected["ms"].get(key) or "").strip()
            for key in ("spectrum_type", "ms_level", "polarity")
            if str(selected["ms"].get(key) or "").strip()
        ) or "Chemotion_MS"
        rows.append(
            {
                "global_row_id": len(rows),
                "chemotion_record_id": record_id,
                "chemotion_observation_ids": json.dumps(
                    {mod: str(selected[mod].get("observation_id") or "") for mod in ("h", "c", "ir", "ms")},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "source_dataset": SOURCE_DATASET,
                "source_subdataset": source_subdataset,
                "name": record_id,
                "inchikey": key,
                "formula": formula,
                "formula_raw": compact,
                "molecular_formula": compact,
                "smiles": canonical,
                "canonical_smiles": canonical,
                "smiles_tokenized": target_line,
                "h_nmr_peaks": h_peaks,
                "c_nmr_peaks": c_peaks,
                "ms_peaks": ms_peaks,
                "spectrum": spectrum,
                "ms_spectrum": spectrum,
                "h_nmr_text": h_block,
                "c_nmr_text": c_block,
                "c_nmr_representation": str(selected["c"].get("submodality") or ""),
                "c_nmr_observation_id": str(selected["c"].get("observation_id") or ""),
                "ir_text": ir_text,
                **ir_meta,
                "ir_source_group": SOURCE_DATASET,
                "spectrum_semantics": ms_semantics,
                "h_nmr_representation": "Chemotion processed peak observations",
                "source_line": source_line,
                "src_line": source_line,
                "target_line": target_line,
                "tgt_line": target_line,
                "h_nmr_raw_json": json.dumps(h_peaks, separators=(",", ":")),
                "c_nmr_raw_json": json.dumps(c_peaks, separators=(",", ":")),
                "ms_raw_json": json.dumps(ms_peaks, separators=(",", ":")),
                "h_condition": "",
                "c_condition": "",
                "ms_condition": "",
                "ms_ion": "",
                "h_image": "",
                "c_image": "",
                "ms_image": "",
                "has_1HNMR": True,
                "has_13CNMR": True,
                "has_IR": True,
                "has_MS": True,
            }
        )
        audit["complete_records"] += 1
        audit[f"complete_by_subdataset:{source_subdataset}"] += 1
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-script", type=Path, required=True)
    parser.add_argument("--legacy-script", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="optional row limit for a validation run")
    parser.add_argument(
        "--c-selection-policy",
        choices=("existing", "peak_picked"),
        default="existing",
        help="13C selection policy; peak_picked prefers explicit peak-list representations",
    )
    parser.add_argument(
        "--h-selection-policy",
        choices=("existing", "peak_picked"),
        default="existing",
        help="1H selection policy; peak_picked prefers explicit multiplet representations",
    )
    parser.add_argument(
        "--dense-c-min-spacing",
        type=float,
        default=0.18,
        help="minimum ppm separation for dense trace-derived C peaks only",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    tokenizer = load_module("chemotion_legacy_tokenizer", args.tokenizer_script.resolve())
    legacy = tokenizer.load_legacy(args.legacy_script.resolve())
    grouped, audit = collect_rows(args.payload_root.resolve(), legacy, tokenizer)
    rows = build_rows(
        grouped,
        legacy,
        tokenizer,
        audit,
        args.limit,
        args.c_selection_policy,
        args.h_selection_policy,
        args.dense_c_min_spacing,
    )
    if not rows:
        raise RuntimeError("no complete Chemotion rows were produced")

    table = pa.Table.from_pylist(rows)
    test_path = output / "test.parquet"
    pq.write_table(table, test_path, compression="zstd")
    (output / "src.txt").write_text("\n".join(row["src_line"] for row in rows) + "\n", encoding="utf-8")
    (output / "tgt.txt").write_text("\n".join(row["tgt_line"] for row in rows) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "Chemotion.ir1800.v1",
        "dataset": output.name,
        "source_dataset": SOURCE_DATASET,
        "payload_root": str(args.payload_root.resolve()),
        "row_count": len(rows),
        "row_limit": args.limit,
        "unique_inchikey_count": len({row["inchikey"] for row in rows}),
        "files": {
            "test.parquet": {"bytes": test_path.stat().st_size, "sha256": sha256(test_path)},
            "src.txt": {"bytes": (output / "src.txt").stat().st_size},
            "tgt.txt": {"bytes": (output / "tgt.txt").stat().st_size},
        },
        "ir_representation": {
            "mode": "simulated_complement",
            "simulated_semantics": "absorbance/intensity-like positive peaks",
            "grid_cm1": [GRID_START, GRID_END, GRID_SIZE],
            "patch_contract": {"n_patches": 24, "patch_size": 75},
            "absorbance": "use native absorbance directly",
            "transmittance_fraction": "100 * (1 - T)",
            "transmittance_percent": "100 - T",
            "normalisation": "per-spectrum min-shift then max-scale to [0, 1]",
            "reflectance": "excluded",
            "arbitrary_units": "excluded",
            "unknown": "excluded",
            "physical_log_absorbance": False,
        },
        "selection": {
            "record_key": "source_record_id",
            "modality_selection": "existing Chemotion deterministic choose_row",
            "c_selection_policy": args.c_selection_policy,
            "h_selection_policy": args.h_selection_policy,
            "c_submodality_priority": C_SUBMODALITY_PRIORITY,
            "h_submodality_priority": H_SUBMODALITY_PRIORITY,
            "c_peak_materialization": (
                "adaptive 0.1-ppm local-max peak picking for dense trace payloads, "
                "using intensity/noise thresholds and minimum ppm spacing without "
                "a molecular-formula-derived peak-count cap; explicit lists are "
                "deduplicated at 0.08 ppm; obvious DMSO-d6/CDCl3 solvent clusters "
                "are removed"
            ),
            "dense_min_spacing_ppm": args.dense_c_min_spacing,
            "duplicate_policy": "drop duplicate (inchikey, canonical_smiles)",
        },
        "audit": dict(audit),
        "scripts": {
            "tokenizer_script": str(args.tokenizer_script.resolve()),
            "legacy_script": str(args.legacy_script.resolve()),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "columns": table.column_names, "audit": dict(audit)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

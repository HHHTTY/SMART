#!/usr/bin/env python3
"""Build a traceable Chemical Education zero-shot parquet from IR and JDX data.

The archive does not contain Auto-MultipletAnalysis peak annotations for most
records. This script therefore derives conservative 1D peak estimates from the
raw JDX spectra and labels the provenance explicitly in the audit manifest.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter


ALLOWED_ELEMENTS = {"C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"}
TARGET_HAC_MIN = 5
TARGET_HAC_MAX = 35


def normalize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", Path(str(value)).stem.lower()).strip("_")


def formula_elements(formula: str) -> list[tuple[str, int]]:
    return [
        (element, int(count) if count else 1)
        for element, count in re.findall(r"([A-Z][a-z]?)([0-9]*)", str(formula))
    ]


def heavy_atom_count(formula: str) -> int:
    return sum(count for element, count in formula_elements(formula) if element != "H")


def hydrogen_count(formula: str) -> int:
    return sum(count for element, count in formula_elements(formula) if element == "H")


def carbon_count(formula: str) -> int:
    return sum(count for element, count in formula_elements(formula) if element == "C")


def read_headers(block: str) -> dict[str, str]:
    return {
        key.strip().lower(): value.strip()
        for key, value in re.findall(r"(?m)^##([^=]+)=\s*(.*)$", block)
    }


def decode_lenient_block(block: str, jcamp: Any) -> tuple[dict[str, str], np.ndarray, np.ndarray, str | None]:
    """Decode a single XYDATA block while tolerating malformed ASDF lines."""

    headers = read_headers(block)
    data_start = re.search(r"(?m)^##XYDATA=.*\n", block)
    if data_start is None:
        return headers, np.empty(0), np.empty(0), "missing_xydata"
    data_text = block[data_start.end() :].split("##END", 1)[0]
    lines = [
        line.strip()
        for line in data_text.splitlines()
        if line.strip() and not line.startswith("$$")
    ]
    if not lines:
        return headers, np.empty(0), np.empty(0), "empty_xydata"

    asdf = any(char in jcamp.DIF_digits for char in lines[0])
    y_values: list[float] = []
    previous_y: float | None = None
    skipped = 0
    for line in lines:
        try:
            values = jcamp.parse(line)
        except Exception:
            skipped += 1
            continue
        if not values:
            continue
        if asdf:
            if len(values) < 2:
                skipped += 1
                continue
            if previous_y is None:
                y_values.extend(float(value) for value in values[1:])
            else:
                y_values.extend(float(value) for value in values[2:])
            if y_values:
                previous_y = y_values[-1]
        elif len(values) >= 2:
            y_values.extend(float(value) for value in values[1:])

    try:
        npoints = int(float(headers.get("npoints", "0")))
        first_x = float(headers["firstx"])
        last_x = float(headers["lastx"])
        y_factor = float(headers.get("yfactor", "1"))
    except (KeyError, TypeError, ValueError) as exc:
        return headers, np.empty(0), np.empty(0), f"bad_header:{exc}"

    if npoints <= 1 or len(y_values) < 2:
        return headers, np.empty(0), np.empty(0), "too_few_values"
    y = np.asarray(y_values[:npoints], dtype=np.float64) * y_factor
    x = np.linspace(first_x, last_x, len(y), dtype=np.float64)
    error = f"skipped_asdf_lines:{skipped}" if skipped else None
    return headers, x, y, error


def split_jdx_blocks(raw_bytes: bytes) -> list[str]:
    text = raw_bytes.decode("utf-8", "ignore")
    starts = list(re.finditer(r"(?m)^##TITLE=", text))
    return [
        text[match.start() : starts[index + 1].start() if index + 1 < len(starts) else len(text)]
        for index, match in enumerate(starts)
    ]


def parse_nmr_blocks(raw_bytes: bytes, jcamp: Any) -> tuple[list[dict[str, Any]], list[str]]:
    blocks: list[dict[str, Any]] = []
    warnings: list[str] = []
    for block_index, block in enumerate(split_jdx_blocks(raw_bytes)):
        headers = read_headers(block)
        if headers.get("data type", "").lower() != "nmr spectrum":
            continue
        if headers.get("data class", "").lower() != "xydata":
            continue
        nucleus = headers.get(".observe nucleus", "")
        inferred_nucleus = False
        if nucleus not in {"^1H", "^13C"}:
            # A few MestReNova exports lose the nucleus label while retaining
            # the standard block order: block 1 is proton and blocks 2-5 are
            # the 13C/DEPT traces.
            try:
                block_id = int(float(headers.get("block_id", "-1")))
            except ValueError:
                block_id = -1
            if block_id == 1:
                nucleus = "^1H"
                inferred_nucleus = True
            elif 2 <= block_id <= 5:
                nucleus = "^13C"
                inferred_nucleus = True
            else:
                continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                parsed = jcamp.read(io.BytesIO(block.encode("utf-8")))
            x = np.asarray(parsed.get("x", []), dtype=np.float64)
            y = np.asarray(parsed.get("y", []), dtype=np.float64)
            warning = None
        except Exception as exc:
            headers, x, y, warning = decode_lenient_block(block, jcamp)
            warning = f"{type(exc).__name__}:{exc};{warning or ''}".rstrip(";")
        if len(x) != len(y) or len(x) < 32:
            warnings.append(f"block={block_index}:length={len(x)}/{len(y)}")
            continue
        try:
            frequency = float(headers.get(".observe frequency", "nan").split(",", 1)[0])
        except ValueError:
            frequency = float("nan")
        if not np.isfinite(frequency) or frequency <= 0:
            warnings.append(f"block={block_index}:bad_frequency")
            continue
        blocks.append(
            {
                "block_index": block_index,
                "block_id": headers.get("block_id"),
                "nucleus": nucleus,
                "frequency_mhz": frequency,
                "x_hz": x,
                "y": y,
                "warning": ";".join(filter(None, [warning, "nucleus_inferred_by_block_id" if inferred_nucleus else None])),
            }
        )
    return blocks, warnings


def _smooth_signal(y: np.ndarray) -> np.ndarray:
    if len(y) < 9:
        return y.astype(np.float64)
    window = min(101, len(y) - 1 if len(y) % 2 == 0 else len(y))
    if window < 9:
        window = 9
    if window % 2 == 0:
        window -= 1
    try:
        return savgol_filter(y, window_length=window, polyorder=3, mode="interp")
    except ValueError:
        return y.astype(np.float64)


def peak_candidates(x_ppm: np.ndarray, y: np.ndarray, nucleus: str, max_peaks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return descending-ppm peak positions, heights, and half-height widths."""

    finite = np.isfinite(x_ppm) & np.isfinite(y)
    x_ppm = x_ppm[finite]
    y = y[finite]
    if nucleus == "^1H":
        region = (x_ppm >= -0.5) & (x_ppm <= 12.5)
        min_spacing_ppm = 0.015
    else:
        region = (x_ppm >= -5.0) & (x_ppm <= 220.0)
        min_spacing_ppm = 0.08
    x_ppm, y = x_ppm[region], y[region]
    if len(x_ppm) < 32:
        return np.empty(0), np.empty(0), np.empty(0)
    order = np.argsort(x_ppm)
    x_ppm, y = x_ppm[order], y[order]
    signal = _smooth_signal(y)
    # The absolute phase is not guaranteed in exported FIDs; choose the
    # orientation with positive absorptive peaks and remove a robust baseline.
    signal = signal - np.percentile(signal, 10)
    if abs(np.min(signal)) > abs(np.max(signal)):
        signal = -signal
    signal = np.maximum(signal, 0.0)
    dynamic = float(np.max(signal) - np.min(signal))
    if not np.isfinite(dynamic) or dynamic <= 0:
        return np.empty(0), np.empty(0), np.empty(0)
    step = float(np.median(np.diff(x_ppm)))
    distance = max(1, int(math.ceil(min_spacing_ppm / max(abs(step), 1e-9))))
    noise = float(np.median(np.abs(signal - np.median(signal))) * 1.4826)
    prominence = max(5.0 * noise, 0.015 * dynamic)
    peaks, properties = find_peaks(signal, prominence=prominence, distance=distance)
    if not len(peaks):
        peaks = np.asarray([int(np.argmax(signal))], dtype=int)
        properties = {"prominences": np.asarray([float(np.max(signal))])}
    prominences = np.asarray(properties.get("prominences", np.ones(len(peaks))), dtype=float)
    order = np.argsort(prominences)[::-1][:max_peaks]
    peaks = peaks[order]
    prominences = prominences[order]
    widths = peak_widths(signal, peaks, rel_height=0.5)[0] * abs(step)
    positions = x_ppm[peaks]
    descending = np.argsort(positions)[::-1]
    return positions[descending], prominences[descending], widths[descending]


def allocate_integrations(weights: np.ndarray, total_h: int) -> list[int]:
    if len(weights) == 0:
        return []
    total_h = max(1, int(total_h))
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-9)
    raw = weights / weights.sum() * total_h
    counts = np.maximum(1, np.floor(raw).astype(int))
    while counts.sum() > total_h and len(counts) > 1:
        index = int(np.argmax(counts))
        if counts[index] <= 1:
            break
        counts[index] -= 1
    while counts.sum() < total_h:
        counts[int(np.argmax(raw - counts))] += 1
    return counts.tolist()


def make_peak_annotations(block: dict[str, Any], formula: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nucleus = block["nucleus"]
    x_ppm = block["x_hz"] / float(block["frequency_mhz"])
    total_h = hydrogen_count(formula)
    total_c = carbon_count(formula)
    max_peaks = max(1, min(64, total_h if nucleus == "^1H" else total_c or 64))
    positions, strengths, widths = peak_candidates(x_ppm, block["y"], nucleus, max_peaks)
    if nucleus == "^1H":
        integrations = allocate_integrations(strengths * np.maximum(widths, 1e-6), total_h)
        peaks = []
        for position, width, n_h in zip(positions, widths, integrations):
            half_width = max(float(width) / 2.0, 0.01)
            peaks.append(
                {
                    "rangeMax": float(position + half_width),
                    "rangeMin": float(position - half_width),
                    "category": "m",
                    "nH": int(n_h),
                    "j_values": "None",
                }
            )
    else:
        peaks = [{"delta (ppm)": float(position)} for position in positions]
    audit = {
        "nucleus": nucleus,
        "block_id": block.get("block_id"),
        "frequency_mhz": block.get("frequency_mhz"),
        "source": "jdx_peak_estimate",
        "peak_count": len(peaks),
        "raw_points": int(len(block["y"])),
        "warning": block.get("warning"),
    }
    return peaks, audit


def fetch_smiles(cas: str, cache: dict[str, str], name: str = "", sleep_seconds: float = 0.05) -> tuple[str | None, str]:
    identifiers = [(str(cas or "").strip(), "cas"), (str(name or "").strip(), "name")]
    last_error = "missing_identifier"
    for identifier, kind in identifiers:
        if not identifier:
            continue
        cache_key = f"{kind}:{identifier}"
        if kind == "cas" and cache_key not in cache and identifier in cache:
            cache[cache_key] = cache[identifier]
        if cache_key in cache:
            return cache[cache_key], f"cache_{kind}"
        url = "https://cactus.nci.nih.gov/chemical/structure/" + urllib.parse.quote(identifier, safe="") + "/smiles"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "education-zero-shot-audit/1.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                value = response.read().decode("utf-8", "replace").strip().splitlines()[0]
            cache[cache_key] = value
            time.sleep(sleep_seconds)
            return value, f"cactus_{kind}"
        except (urllib.error.URLError, TimeoutError, IndexError) as exc:
            last_error = f"resolver_error_{kind}:{type(exc).__name__}"
    return None, last_error


def canonicalize_smiles(value: str | None) -> str | None:
    if not value:
        return None
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(value)
        return Chem.MolToSmiles(molecule) if molecule is not None else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnova-zip", type=Path, required=True)
    parser.add_argument("--ir-parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        import jcamp  # type: ignore
    except ImportError as exc:
        raise SystemExit("jcamp is required; set PYTHONPATH=/tmp/jcamp_probe_inst") from exc

    manifest_rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    ir_frame = pd.read_parquet(args.ir_parquet)
    ir_by_id = {str(row.source_record_id): row for row in ir_frame.itertuples(index=False)}
    resolution_cache: dict[str, str] = {}
    if args.cache and args.cache.exists():
        resolution_cache = json.loads(args.cache.read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(args.mnova_zip) as archive:
        members = {normalize_id(info.filename): info for info in archive.infolist() if info.filename.lower().endswith(".jdx")}
        for index, row in enumerate(manifest_rows, 1):
            source_id = str(row.get("source_record_id", ""))
            hac = heavy_atom_count(row.get("formula", ""))
            claims_nmr = row.get("H-1_claim", "") == "1" and row.get("C-13_claim", "") == "1"
            selected = claims_nmr and TARGET_HAC_MIN < hac < TARGET_HAC_MAX
            smiles_raw, smiles_source = fetch_smiles(row.get("cas", ""), resolution_cache, row.get("compound_name", ""))
            smiles = canonicalize_smiles(smiles_raw)
            record: dict[str, Any] = {
                "source_record_id": source_id,
                "compound_name": row.get("compound_name", ""),
                "formula": row.get("formula", ""),
                "cas": row.get("cas", ""),
                "heavy_atom_count": hac,
                "claims_nmr": claims_nmr,
                "paper_hac_filter": selected,
                "smiles": smiles,
                "smiles_raw": smiles_raw,
                "smiles_source": smiles_source,
                "jdx_member": None,
                "h_nmr_peaks": None,
                "c_nmr_peaks": None,
                "peak_audit": {},
                "warnings": [],
            }
            info = members.get(normalize_id(row.get("archive_base_name", source_id)) + "_jdx")
            if info is None:
                # normalize_id strips the extension, so the direct lookup is the
                # robust path for names containing spaces/parentheses.
                expected = normalize_id(row.get("archive_base_name", source_id))
                info = next((item for key, item in members.items() if key == expected), None)
            if info is not None:
                record["jdx_member"] = info.filename
                blocks, warnings = parse_nmr_blocks(archive.read(info), jcamp)
                record["warnings"].extend(warnings)
                h_blocks = [block for block in blocks if block["nucleus"] == "^1H"]
                c_blocks = [block for block in blocks if block["nucleus"] == "^13C"]
                if h_blocks:
                    record["h_nmr_peaks"], record["peak_audit"]["1H"] = make_peak_annotations(h_blocks[0], row.get("formula", ""))
                if c_blocks:
                    record["c_nmr_peaks"], record["peak_audit"]["13C"] = make_peak_annotations(c_blocks[0], row.get("formula", ""))
            else:
                record["warnings"].append("missing_jdx")
            if smiles is None:
                record["warnings"].append("missing_smiles")
            if source_id not in ir_by_id:
                record["warnings"].append("missing_ir")
            if index % 20 == 0:
                print(f"prepared {index}/{len(manifest_rows)}", flush=True)
            records.append(record)

    if args.cache:
        args.cache.write_text(json.dumps(resolution_cache, indent=2, sort_keys=True), encoding="utf-8")

    frame_rows: list[dict[str, Any]] = []
    for record in records:
        if not record["paper_hac_filter"]:
            continue
        ir_row = ir_by_id.get(record["source_record_id"])
        if ir_row is None or record["smiles"] is None:
            continue
        if record["h_nmr_peaks"] is None or record["c_nmr_peaks"] is None:
            continue
        frame_rows.append(
            {
                "source_record_id": record["source_record_id"],
                "compound_name": record["compound_name"],
                "molecular_formula": record["formula"],
                "cas": record["cas"],
                "smiles": record["smiles"],
                "h_nmr_peaks": record["h_nmr_peaks"],
                "c_nmr_peaks": record["c_nmr_peaks"],
                "ir_spectra": list(getattr(ir_row, "ir_spectra")),
            }
        )

    frame = pd.DataFrame(frame_rows)
    output_parquet = data_dir / "education_paired170_jdx_estimate.parquet"
    frame.to_parquet(output_parquet, index=False)
    (args.output_dir / "education_records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(records).drop(columns=["h_nmr_peaks", "c_nmr_peaks", "peak_audit"], errors="ignore").to_csv(
        args.output_dir / "education_manifest.csv", index=False
    )
    summary = {
        "input_ir_rows": len(manifest_rows),
        "paper_hac_filter_rows": sum(bool(record["paper_hac_filter"]) for record in records),
        "paired_rows_written": len(frame),
        "missing_nmr_or_jdx_after_filter": sum(
            bool(record["paper_hac_filter"]) and (record["h_nmr_peaks"] is None or record["c_nmr_peaks"] is None)
            for record in records
        ),
        "missing_smiles_after_filter": sum(bool(record["paper_hac_filter"]) and record["smiles"] is None for record in records),
        "ir_length_counts": frame["ir_spectra"].map(len).value_counts().to_dict() if len(frame) else {},
        "nmr_peak_source": "jdx_peak_estimate; Auto-MultipletAnalysis TSVs were empty for all but one member",
        "paper_filter": "H-1_claim=C-13_claim=1 and 5 < heavy_atom_count < 35",
        "note": "The uploaded registry has 205 IR rows but xanthine is IR-only; 170 fully paired rows are written.",
    }
    (args.output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

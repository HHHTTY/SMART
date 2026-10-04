#!/usr/bin/env python3
"""Build the reproducible, molecule-deduplicated SDBS_final source dataset.

The input is the directly downloaded SDBS.zip archive.  The builder joins the
three numeric AIST tables by SDBS number/InChIKey and joins numeric IR JDX
files through the InChI in nist_ir_info.csv.  It deliberately does not use
the scanned IR images as a numerical fallback and does not remove records
because of text or spectrum sequence length.

The resulting source contract is the one consumed by the simulated
HC--IR--MS preprocessor: formula, 1HNMR, 13CNMR, IR (400 integer values), and
an untagged EI-MS pair string.  SDBS contains EI spectra rather than the six
simulated collision-energy views, so the MS block is explicitly kept as
``MS`` and its EI provenance is recorded in the manifest.  The output is one
combined dataset; no train/validation/test split is generated here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


GRID = np.linspace(400.0, 4000.0, 400, dtype=np.float64)
IR_MODES = ("simulated_complement", "strict_absorbance")
TRANSMITTANCE_FLOOR = 1e-4
KEY_RE = re.compile(r"^[A-Z0-9]{14}-[A-Z0-9]{10}-[A-Z]$")
NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[EeDd][+-]?\d+)?$"
)
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%\d{2}|\d)"
)


class BuildError(RuntimeError):
    """Raised when the source cannot satisfy the explicit dataset contract."""


@dataclass
class JDXRecord:
    key: str
    filename: str
    xunits: str
    yunits: str
    xunits_raw: str
    yunits_raw: str
    collection: str
    origin: str
    inchi: str
    npoints_declared: int
    x: np.ndarray
    signal: np.ndarray
    grid_signal: np.ndarray
    valid_mask: np.ndarray
    coverage: float
    ir_transform: str
    transmittance_scale: str
    transmittance_clipped_low: int
    transmittance_clipped_high: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_dir", type=Path, required=True)
    parser.add_argument(
        "--expected-count", type=int, default=3669,
        help="Expected unique molecule count; set 0 only for exploratory runs.",
    )
    parser.add_argument(
        "--ir-mode", choices=IR_MODES, default="simulated_complement",
        help=(
            "IR transmittance representation. The default preserves the existing "
            "simulation-compatible complement; strict_absorbance applies -log10(T)."
        ),
    )
    parser.add_argument(
        "--ignore-images",
        action="store_true",
        help=(
            "Do not read IR image metadata or apply the IR-image presence gate; "
            "join H/C/MS tables directly and use only numeric JDX for IR."
        ),
    )
    return parser.parse_args()


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _finite(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BuildError(f"{context}: not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise BuildError(f"{context}: non-finite value")
    return result


def _literal(value: str, context: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise BuildError(f"{context}: invalid Python literal") from exc


def _key_and_inchi(row: list[str]) -> tuple[str, str]:
    key = next(
        (
            _text(cell).upper()
            for cell in row
            if KEY_RE.fullmatch(_text(cell).upper())
        ),
        "",
    )
    inchi = next((_text(cell) for cell in row if _text(cell).startswith("InChI=")), "")
    return key, inchi


def _read_csv(zf: zipfile.ZipFile, member: str) -> list[list[str]]:
    with zf.open(member, "r") as raw:
        wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
        return list(csv.reader(wrapper))


def _row_payload(
    row: list[str], kind: str, *, include_image_metadata: bool = True
) -> dict[str, Any] | None:
    """Normalize the archive's rows (H has one extra unquoted field in practice)."""
    if not row or not _text(row[0]):
        return None
    sid = _text(row[0])
    key, inchi = _key_and_inchi(row)
    # Some late SDBS rows contain a valid InChIKey but leave the InChI cell
    # blank.  Keep those rows; the numeric JDX metadata can supply the InChI
    # later, and the key remains the primary join identity.
    if not key:
        return None
    if kind == "h":
        # Actual data rows have assignment_data at 7, shift_data at 8, InChI at
        # 9, key at 10, and image at 11, despite the 11-column header.
        key_idx = next(
            i for i, cell in enumerate(row) if KEY_RE.fullmatch(_text(cell).upper())
        )
        shift_idx = key_idx - 2
        assignment_idx = key_idx - 3
        if shift_idx < 0 or assignment_idx < 0:
            return None
        payload = {
            "sid": sid,
            "key": key,
            "inchi": inchi,
            "name": _text(row[1]),
            "formula_raw": _text(row[2]),
            "frequency": _text(row[5]),
            "condition": _text(row[6]),
            "assignment_raw": _text(row[assignment_idx]),
            "shift_raw": _text(row[shift_idx]),
        }
        if include_image_metadata:
            payload["image"] = _text(row[-1])
        return payload
    if kind == "c":
        payload = {
            "sid": sid,
            "key": key,
            "inchi": inchi,
            "name": _text(row[1]),
            "formula_raw": _text(row[2]),
            "frequency": _text(row[5]),
            "condition": _text(row[6]),
            "shift_raw": _text(row[7]),
        }
        if include_image_metadata:
            payload["image"] = _text(row[-1])
        return payload
    if kind == "m":
        payload = {
            "sid": sid,
            "key": key,
            "inchi": inchi,
            "name": _text(row[1]),
            "formula_raw": _text(row[2]),
            "ion": _text(row[5]),
            "condition": _text(row[6]),
            "peak_raw": _text(row[7]),
        }
        if include_image_metadata:
            payload["image"] = _text(row[-1])
        return payload
    raise ValueError(kind)


def _numeric_h_peaks(raw: str, context: str) -> tuple[list[dict[str, Any]], list[list[float]]]:
    value = _literal(raw, context)
    if not isinstance(value, (list, tuple)):
        raise BuildError(f"{context}: expected a list")
    points: list[dict[str, Any]] = []
    triples: list[list[float]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        # SDBS H numeric data is (frequency, shift, intensity).  A few rows
        # contain a two-field form; preserving it is preferable to dropping it.
        ppm = _finite(item[1], f"{context}[{idx}].ppm")
        intensity = _finite(item[2], f"{context}[{idx}].intensity") if len(item) >= 3 else 1.0
        frequency = _finite(item[0], f"{context}[{idx}].frequency")
        points.append(
            {
                "rangeMax": ppm,
                "rangeMin": ppm,
                "centroid": ppm,
                "category": "m",
                "nH": 1,
                "j_values": "None",
            }
        )
        triples.append([frequency, ppm, intensity])
    points.sort(key=lambda peak: (float(peak["rangeMin"]), float(peak["rangeMax"])))
    triples.sort(key=lambda item: (item[1], item[0]))
    if not points:
        raise BuildError(f"{context}: no numeric H peaks")
    return points, triples


def _numeric_c_peaks(raw: str, context: str) -> tuple[list[dict[str, Any]], list[list[float]]]:
    value = _literal(raw, context)
    if not isinstance(value, (list, tuple)):
        raise BuildError(f"{context}: expected a list")
    peaks: list[dict[str, Any]] = []
    raw_values: list[list[float]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        ppm = _finite(item[0], f"{context}[{idx}].ppm")
        intensity = _finite(item[1], f"{context}[{idx}].intensity") if len(item) >= 2 else 1.0
        # CarbonPreprocessor itself rounds to one decimal, so materialize the
        # same representation while retaining the unrounded value in JSON.
        peaks.append({"delta (ppm)": round(ppm, 1)})
        raw_values.append([ppm, intensity])
    peaks.sort(key=lambda peak: float(peak["delta (ppm)"]))
    raw_values.sort(key=lambda item: item[0])
    if not peaks:
        raise BuildError(f"{context}: no numeric C peaks")
    return peaks, raw_values


def _numeric_ms_peaks(raw: str, context: str) -> tuple[list[list[float]], list[list[float]]]:
    value = _literal(raw, context)
    if not isinstance(value, (list, tuple)):
        raise BuildError(f"{context}: expected a list")
    peaks: list[list[float]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        mz = _finite(item[0], f"{context}[{idx}].mz")
        intensity = _finite(item[1], f"{context}[{idx}].intensity")
        peaks.append([mz, intensity])
    peaks.sort(key=lambda item: (item[0], item[1]))
    if not peaks:
        raise BuildError(f"{context}: no numeric MS peaks")
    return peaks, [list(item) for item in peaks]


def _spaced_formula(formula: str) -> str:
    tokens: list[str] = []
    for element, count in FORMULA_RE.findall(formula.replace(" ", "")):
        tokens.append(element)
        if count and int(count) != 1:
            tokens.append(count)
    if not tokens:
        raise BuildError(f"cannot tokenize molecular formula {formula!r}")
    return " ".join(tokens)


def _tokenized_smiles(smiles: str) -> str:
    pieces = SMILES_TOKEN_RE.findall(smiles)
    if "".join(pieces) != smiles:
        raise BuildError(f"SMILES tokenizer did not cover {smiles!r}")
    return " ".join(pieces)


def _h_text(peaks: list[dict[str, Any]]) -> str:
    tokens = ["1HNMR"]
    for peak in peaks:
        value = float(peak["rangeMax"])
        tokens.extend([f"{value:.2f}", f"{value:.2f}", "m", "1H", "|"])
    return " ".join(tokens[:-1])


def _c_text(peaks: list[dict[str, Any]]) -> str:
    return " ".join(["13CNMR"] + [str(peak["delta (ppm)"]) for peak in peaks])


def _ms_text(peaks: list[list[float]]) -> str:
    values: list[str] = []
    for mz, intensity in peaks:
        values.extend([f"{mz:.1f}", f"{intensity:.1f}"])
    return " ".join(values)


def _jdx_header(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.startswith("##") or "=" not in line:
            continue
        key, value = line[2:].split("=", 1)
        headers[key.strip().upper()] = value.strip()
    return headers


def _float_token(token: str) -> float | None:
    token = token.strip().rstrip(",;")
    if not NUMBER_RE.fullmatch(token):
        return None
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _accepted_y_unit(yunits: str) -> bool:
    normalized = yunits.lower().replace(" ", "")
    return "absorbance" in normalized or "transmittance" in normalized


def _accepted_x_unit(xunits: str) -> bool:
    normalized = xunits.lower().replace(" ", "")
    return "micrometer" in normalized or "1/cm" in normalized or "cm-1" in normalized


def _transmittance_scale(y: np.ndarray) -> str:
    """Infer fraction versus percent from robust percentiles."""
    p99 = float(np.percentile(y, 99.0))
    return "fraction" if p99 <= 1.5 else "percent"


def _parse_jdx_bytes(
    raw_bytes: bytes,
    *,
    key: str,
    filename: str,
    collection: str,
    origin: str,
    inchi: str,
    ir_mode: str,
) -> JDXRecord | None:
    raw = raw_bytes.decode("latin1", errors="replace")
    header = _jdx_header(raw)
    xunits_raw = header.get("XUNITS", "")
    yunits_raw = header.get("YUNITS", "")
    if not _accepted_x_unit(xunits_raw) or not _accepted_y_unit(yunits_raw):
        return None
    data_marker = next(
        (line.upper() for line in raw.splitlines() if line.upper().startswith("##XYDATA=")),
        "",
    )
    if not data_marker:
        return None
    try:
        xfactor = float(header.get("XFACTOR", "1").replace("D", "E").replace("d", "e"))
        yfactor = float(header.get("YFACTOR", "1").replace("D", "E").replace("d", "e"))
        deltax = float(header.get("DELTAX", "0").replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in (xfactor, yfactor, deltax)):
        return None
    npoints_declared = 0
    try:
        npoints_declared = int(float(header.get("NPOINTS", "0")))
    except ValueError:
        pass
    lines = raw.splitlines()
    in_data = False
    x_values: list[float] = []
    y_values: list[float] = []
    for line in lines:
        upper = line.upper()
        if upper.startswith("##XYDATA="):
            in_data = True
            continue
        if not in_data:
            continue
        if upper.startswith("##END=") or upper.startswith("##"):
            break
        fields = line.replace(",", " ").replace(";", " ").split()
        if len(fields) < 2:
            continue
        first_x = _float_token(fields[0])
        if first_x is None:
            continue
        y_row: list[float] = []
        for token in fields[1:]:
            number = _float_token(token)
            if number is not None:
                y_row.append(number * yfactor)
        if not y_row:
            continue
        base_x = first_x * xfactor
        step = deltax * xfactor
        # XYDATA=(X++(Y..Y)) is the format used throughout this archive.
        # If DELTAX is absent, infer the spacing from FIRSTX/LASTX/NPOINTS.
        if abs(step) < 1e-15:
            try:
                first = float(header["FIRSTX"])
                last = float(header["LASTX"])
                step = (last - first) / max(1, npoints_declared - 1)
            except (KeyError, ValueError):
                step = 0.0
        for offset, y in enumerate(y_row):
            x_values.append(base_x + step * offset)
            y_values.append(y)
    if len(x_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return None
    if "micrometer" in xunits_raw.lower():
        positive = x > 0
        x, y = 10000.0 / x[positive], y[positive]
    if x.size < 2:
        return None
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size != x.size:
        sums = np.zeros(unique_x.size, dtype=np.float64)
        counts = np.zeros(unique_x.size, dtype=np.int64)
        np.add.at(sums, inverse, y)
        np.add.at(counts, inverse, 1)
        x, y = unique_x, sums / np.maximum(counts, 1)
    transmittance_scale = ""
    transmittance_clipped_low = 0
    transmittance_clipped_high = 0
    if "transmittance" in yunits_raw.lower():
        # NIST's micrometer files encode T as 0..1; older wavenumber files
        # use percent. Clip small baseline excursions before converting so a
        # few negative/out-of-range instrument values cannot create NaNs or
        # invalid logarithms.
        scale = _transmittance_scale(y) if ir_mode == "strict_absorbance" else (
            "fraction" if float(np.nanmax(y)) <= 1.5 else "percent"
        )
        raw_fraction = y if scale == "fraction" else y / 100.0
        transmittance_scale = scale
        transmittance_clipped_low = int(np.count_nonzero(raw_fraction < 0.0))
        transmittance_clipped_high = int(np.count_nonzero(raw_fraction > 1.0))
        if ir_mode == "strict_absorbance":
            transmittance = np.clip(raw_fraction, 0.0, 1.0)
            signal = -np.log10(np.clip(transmittance, TRANSMITTANCE_FLOOR, 1.0))
            ir_transform = f"T_{scale.upper()}_TO_A_LOG10_FLOOR1E-4_THEN_LEGACY_IR400"
        else:
            # Preserve the original SDBS_final representation exactly in the
            # compatibility mode; its legacy path intentionally did not clip
            # baseline excursions before min-shift/max-scale.
            signal = 100.0 * (1.0 - raw_fraction)
            ir_transform = "T_TO_100_MINUS_T_THEN_LEGACY_IR400"
    else:
        signal = y
        ir_transform = "A_DIRECT_THEN_LEGACY_IR400"
    valid_mask = (GRID >= float(x.min())) & (GRID <= float(x.max()))
    grid_signal = np.interp(GRID, x, signal)
    coverage = float(valid_mask.mean())
    return JDXRecord(
        key=key,
        filename=filename,
        xunits=xunits_raw.lower(),
        yunits=yunits_raw.lower(),
        xunits_raw=xunits_raw,
        yunits_raw=yunits_raw,
        collection=collection,
        origin=origin,
        inchi=inchi,
        npoints_declared=npoints_declared,
        x=x,
        signal=signal,
        grid_signal=grid_signal,
        valid_mask=valid_mask,
        coverage=coverage,
        ir_transform=ir_transform,
        transmittance_scale=transmittance_scale,
        transmittance_clipped_low=transmittance_clipped_low,
        transmittance_clipped_high=transmittance_clipped_high,
    )


def _model_ir400(record: JDXRecord) -> list[int]:
    values = np.asarray(record.grid_signal, dtype=np.float64)
    if values.shape != (400,) or not np.isfinite(values).all():
        raise BuildError(f"IR {record.filename}: invalid interpolated vector")
    shifted = values + abs(float(values.min()))
    maximum = float(shifted.max(initial=0.0))
    if maximum <= 0.0:
        # Keep the molecule rather than introducing a length/quality filter;
        # the mask and flat-spectrum flag make this exceptional case visible.
        return [0] * 400
    return np.clip(np.rint(shifted / maximum * 100.0), 0, 100).astype(np.int16).tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_sid(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**12, value


def _load_aist(
    zf: zipfile.ZipFile,
    *,
    use_image_gate: bool = True,
    include_image_metadata: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Group by InChIKey, then require one shared SDBS number across all three
    # numeric AIST tables.  The shared number is the provenance anchor for the
    # IR image claim; allowing an H/C/MS cross-number splice would silently
    # combine separate source records that merely share a normalized key.
    tables: dict[str, dict[str, list[dict[str, Any]]]] = {}
    parse_failures: Counter[str] = Counter()
    for kind, member in (("h", "IR/hnmr_results.csv"), ("c", "IR/cnmr_results.csv"), ("m", "IR/ms_results.csv")):
        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _read_csv(zf, member)[1:]:
            try:
                payload = _row_payload(
                    row, kind, include_image_metadata=include_image_metadata
                )
                if payload is not None:
                    by_key[payload["key"]].append(payload)
            except (BuildError, StopIteration, IndexError) as exc:
                parse_failures[f"{kind}:{type(exc).__name__}"] += 1
        tables[kind] = by_key
    image_sids = set()
    if use_image_gate:
        image_sids = {
            _text(row[0])
            for row in _read_csv(zf, "IR/ir_results.csv")[1:]
            if len(row) >= 6 and _text(row[5])
        }
    candidates: list[dict[str, Any]] = []
    common_keys = set(tables["h"]) & set(tables["c"]) & set(tables["m"])
    image_key_count = 0
    shared_sid_key_count = 0
    cross_sid_only_key_count = 0

    def parsed_rows(kind: str, key: str) -> list[tuple[dict[str, Any], Any, Any]]:
        parsed: list[tuple[dict[str, Any], Any, Any]] = []
        for row in tables[kind][key]:
            try:
                if kind == "h":
                    values = _numeric_h_peaks(row["shift_raw"], f"H:{key}")
                elif kind == "c":
                    values = _numeric_c_peaks(row["shift_raw"], f"C:{key}")
                else:
                    values = _numeric_ms_peaks(row["peak_raw"], f"MS:{key}")
                parsed.append((row, values[0], values[1]))
            except BuildError as exc:
                parse_failures[f"joined:{kind}"] += 1
        return parsed

    for key in sorted(common_keys):
        parsed: dict[str, list[tuple[dict[str, Any], Any, Any]]] = {
            kind: parsed_rows(kind, key) for kind in ("h", "c", "m")
        }
        if not all(parsed.values()):
            continue
        shared_sids = set.intersection(
            *[{item[0]["sid"] for item in parsed[kind]} for kind in ("h", "c", "m")]
        )
        if not shared_sids:
            # A key can occur in all three tables while referring to different
            # SDBS records.  Do not manufacture a multimodal molecule by
            # joining those records across tables.
            cross_sid_only_key_count += 1
            continue
        shared_sid_key_count += 1
        if use_image_gate and not (shared_sids & image_sids):
            continue
        if use_image_gate:
            image_key_count += 1
        # If a table has a duplicate row, choose the one carrying the most
        # numeric observations, then the smallest numeric SDBS number for
        # reproducibility.  All selected rows are restricted to shared SIDs.

        def score(item: tuple[dict[str, Any], Any, Any]) -> tuple[int, tuple[int, str]]:
            row, primary, _ = item
            return (-len(primary), _numeric_sid(str(row["sid"])))

        chosen: dict[str, tuple[dict[str, Any], Any, Any]] = {}
        for kind in ("h", "c", "m"):
            options = [item for item in parsed[kind] if item[0]["sid"] in shared_sids]
            chosen[kind] = sorted(options, key=score)[0]
        h, h_peaks, h_raw = chosen["h"]
        c, c_peaks, c_raw = chosen["c"]
        m, ms_peaks, ms_raw = chosen["m"]
        inchi = next(
            (row["inchi"] for row in (h, c, m) if row.get("inchi")),
            "",
        )
        candidates.append(
            {
                "sid": min((h["sid"], c["sid"], m["sid"]), key=_numeric_sid),
                "key": key,
                "inchi": inchi,
                "name": h["name"] or c["name"] or m["name"],
                "formula_raw": h["formula_raw"] or c["formula_raw"] or m["formula_raw"],
                "h_peaks": h_peaks,
                "h_raw": h_raw,
                "c_peaks": c_peaks,
                "c_raw": c_raw,
                "ms_peaks": ms_peaks,
                "ms_raw": ms_raw,
                "h_condition": h["condition"],
                "c_condition": c["condition"],
                "ms_condition": m["condition"],
                "ms_ion": m["ion"],
            }
        )
        if include_image_metadata:
            candidates[-1].update(
                {
                    "h_image": h["image"],
                    "c_image": c["image"],
                    "ms_image": m["image"],
                }
            )
    return candidates, {
        "table_rows": {
            kind: sum(len(rows) for rows in tables[kind].values())
            for kind in ("h", "c", "m")
        },
        "image_gate_applied": use_image_gate,
        "image_metadata_included": include_image_metadata,
        "image_sids": len(image_sids) if use_image_gate else None,
        "common_keys_with_image": image_key_count if use_image_gate else None,
        "common_keys_with_shared_sid": shared_sid_key_count,
        "cross_sid_only_keys_excluded": cross_sid_only_key_count,
        "joined_candidates": len(candidates),
        "joined_candidate_keys": len({row["key"] for row in candidates}),
        "parse_failures": dict(parse_failures),
    }


def _load_jdx_index(zf: zipfile.ZipFile) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: Counter[str] = Counter()
    for row in _read_csv(zf, "IR/nist_ir_info.csv")[1:]:
        if len(row) < 7:
            failures["short_metadata_row"] += 1
            continue
        inchi = _text(row[2])
        filename = _text(row[6])
        if not inchi.startswith("InChI=") or not filename:
            failures["missing_inchi_or_filename"] += 1
            continue
        try:
            mol = Chem.MolFromInchi(inchi)
            key = Chem.MolToInchiKey(mol) if mol is not None else ""
        except Exception:
            key = ""
        if not key:
            failures["invalid_inchi"] += 1
            continue
        index[key].append(
            {
                "filename": filename,
                "collection": _text(row[12]) if len(row) > 12 else "",
                "origin": _text(row[13]) if len(row) > 13 else "",
                "cID": _text(row[0]),
                "inchi": inchi,
            }
        )
    return index, {
        "metadata_rows": sum(len(v) for v in index.values()) + sum(failures.values()),
        "metadata_keys": len(index),
        "metadata_failures": dict(failures),
    }


def _choose_jdx(
    zf: zipfile.ZipFile,
    key: str,
    candidates: Iterable[dict[str, Any]],
    cache: dict[str, JDXRecord | None],
    unit_counts: Counter[str],
    member_names: set[str],
    ir_mode: str,
) -> JDXRecord | None:
    parsed: list[JDXRecord] = []
    for item in candidates:
        filename = item["filename"]
        member = filename if filename.startswith("IR/") else f"IR/IR/{filename}"
        if member not in member_names:
            unit_counts["missing_member"] += 1
            continue
        if member not in cache:
            try:
                cache[member] = _parse_jdx_bytes(
                    zf.read(member),
                    key=key,
                    filename=filename,
                    collection=item.get("collection", ""),
                    origin=item.get("origin", ""),
                    inchi=item.get("inchi", ""),
                    ir_mode=ir_mode,
                )
            except Exception:
                cache[member] = None
        record = cache[member]
        if record is None:
            unit_counts["unusable"] += 1
        else:
            unit_counts[f"{record.xunits_raw}|{record.yunits_raw}"] += 1
            parsed.append(record)
    if not parsed:
        return None
    # Prefer the widest measured grid; ties are deterministic and independent
    # of ZIP member ordering.
    parsed.sort(key=lambda rec: (-rec.coverage, -rec.x.size, rec.filename))
    return parsed[0]


def _make_record(
    candidate: dict[str, Any],
    ir: JDXRecord,
    row_id: int,
    *,
    include_image_metadata: bool = True,
) -> dict[str, Any]:
    inchi = candidate.get("inchi") or ir.inchi
    molecule = Chem.MolFromInchi(inchi)
    if molecule is None:
        raise BuildError(f"{candidate['sid']}: RDKit cannot parse InChI")
    computed_key = Chem.MolToInchiKey(molecule)
    if computed_key and computed_key.upper() != str(candidate["key"]).upper():
        raise BuildError(
            f"{candidate['sid']}: InChIKey mismatch ({candidate['key']} vs {computed_key})"
        )
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    formula_spaced = _spaced_formula(formula)
    smiles_tokenized = _tokenized_smiles(smiles)
    h_text = _h_text(candidate["h_peaks"])
    c_text = _c_text(candidate["c_peaks"])
    ms_text = _ms_text(candidate["ms_peaks"])
    ir_values = _model_ir400(ir)
    src_line = " ".join(
        [formula_spaced, h_text, c_text, "IR", *map(str, ir_values), "MS", ms_text]
    )
    record = {
        "global_row_id": row_id,
        "sdbs_no": str(candidate["sid"]),
        "name": candidate["name"],
        "inchikey": candidate["key"],
        "inchi": inchi,
        "formula": formula_spaced,
        "formula_raw": candidate["formula_raw"],
        "molecular_formula": formula,
        "smiles": smiles,
        "canonical_smiles": smiles,
        "smiles_tokenized": smiles_tokenized,
        "h_nmr_peaks": candidate["h_peaks"],
        "c_nmr_peaks": candidate["c_peaks"],
        "ms_peaks": candidate["ms_peaks"],
        # The frozen generation config consumes `spectrum` as the MS text.
        "spectrum": ms_text,
        "ms_spectrum": ms_text,
        "h_nmr_text": h_text,
        "c_nmr_text": c_text,
        "ir_text": "IR " + " ".join(map(str, ir_values)),
        "ir_400": ir_values,
        "ir_valid_mask": ir.valid_mask.tolist(),
        "ir_coverage": ir.coverage,
        "ir_raw_points": int(ir.x.size),
        "ir_raw_x_min": float(ir.x.min()),
        "ir_raw_x_max": float(ir.x.max()),
        "ir_raw_y_min": float(ir.signal.min()),
        "ir_raw_y_max": float(ir.signal.max()),
        "ir_jdx_member": ir.filename,
        "ir_xunits": ir.xunits_raw,
        "ir_yunits": ir.yunits_raw,
        "ir_transform": ir.ir_transform,
        "ir_transmittance_scale": ir.transmittance_scale,
        "ir_transmittance_clipped_low": ir.transmittance_clipped_low,
        "ir_transmittance_clipped_high": ir.transmittance_clipped_high,
        "ir_source_group": "NIST_COBLENTZ_ACCEPTED_AS_SAME_SOURCE",
        "spectrum_semantics": "EI_75eV",
        "h_nmr_representation": "all_numeric_shift_data_as_singleton_multiplets",
        "source_line": src_line,
        "src_line": src_line,
        "target_line": smiles_tokenized,
        "tgt_line": smiles_tokenized,
        "h_nmr_raw_json": json.dumps(candidate["h_raw"], separators=(",", ":")),
        "c_nmr_raw_json": json.dumps(candidate["c_raw"], separators=(",", ":")),
        "ms_raw_json": json.dumps(candidate["ms_raw"], separators=(",", ":")),
        "h_condition": candidate["h_condition"],
        "c_condition": candidate["c_condition"],
        "ms_condition": candidate["ms_condition"],
        "ms_ion": candidate["ms_ion"],
        "has_1HNMR": True,
        "has_13CNMR": True,
        "has_IR": True,
        "has_MS": True,
    }
    if include_image_metadata:
        record.update(
            {
                "h_image": candidate["h_image"],
                "c_image": candidate["c_image"],
                "ms_image": candidate["ms_image"],
            }
        )
    return record


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="zstd")


def _write_lines(path: Path, records: list[dict[str, Any]], field: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(str(row[field]).rstrip("\n") + "\n")


def _write_readme(path: Path, manifest: dict[str, Any]) -> None:
    ir_policy = manifest["ir_representation"]
    if ir_policy["mode"] == "strict_absorbance":
        ir_description = (
            "Absorbance is used directly; transmittance is converted to a fraction "
            "and then to physical absorbance with "
            f"A=-log10(max(T, {TRANSMITTANCE_FLOOR:g}))."
        )
    else:
        ir_description = "Absorbance is used directly; transmittance is converted to 100 - T first."
    text = f"""# SDBS_final

This directory is a deterministic source-format view of the directly
downloaded `SDBS.zip` archive. It contains **{manifest['row_count']} unique
InChIKeys** in one combined dataset. This builder intentionally does not
generate train/validation/test split files; split design is left to the
downstream zero-shot evaluation.

## Input contract

* `src*.txt` lines are formula + `1HNMR` + `13CNMR` + `IR` (400 values) +
  `MS` (SDBS EI, 75 eV).
* H-NMR uses every numeric SDBS shift as a singleton `m 1H` multiplet. The
  original triples are retained in the parquet JSON sidecars.
* C-NMR is rounded to one decimal exactly as the frozen preprocessor does.
* IR is interpolated on 400..4000 cm-1, then shifted and max-scaled to 0..100,
  after the selected representation conversion. {ir_description} The raw
  coverage mask is retained.
* No row is removed for text length, peak count, or IR coverage. The fixed IR
  grid is a representation requirement, not a row-selection rule.

`spectrum`/`ms_spectrum` in parquet contain only the EI pair string because
the frozen generation configuration maps its MSMS column to `spectrum`.
`source_line` and `src_line` contain the complete OpenNMT input line.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    zip_path = args.zip_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates, aist_stats = _load_aist(
            zf,
            use_image_gate=not args.ignore_images,
            include_image_metadata=not args.ignore_images,
        )
        jdx_index, jdx_stats = _load_jdx_index(zf)
        candidate_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            candidate_by_key[str(candidate["key"])].append(candidate)
        cache: dict[str, JDXRecord | None] = {}
        unit_counts: Counter[str] = Counter()
        member_names = set(zf.namelist())
        selected: list[tuple[dict[str, Any], JDXRecord]] = []
        no_jdx_keys: list[str] = []
        for key in sorted(candidate_by_key):
            ir = _choose_jdx(
                zf, key, jdx_index.get(key, []), cache, unit_counts, member_names,
                args.ir_mode,
            )
            if ir is None:
                no_jdx_keys.append(key)
                continue
            # If multiple AIST rows represent one InChIKey, retain the row with
            # the most observed information and use SDBS number as a tie-break.
            options = sorted(
                candidate_by_key[key],
                key=lambda row: (
                    -(len(row["h_peaks"]) + len(row["c_peaks"]) + len(row["ms_peaks"])),
                    _numeric_sid(str(row["sid"])),
                ),
            )
            selected.append((options[0], ir))
        if args.expected_count and len(selected) != args.expected_count:
            raise BuildError(
                f"selected {len(selected)} unique keys, expected {args.expected_count}; "
                f"AIST candidates={len(candidates)}, no_jdx_keys={len(no_jdx_keys)}"
            )
        records = [
            _make_record(
                candidate,
                ir,
                idx,
                include_image_metadata=not args.ignore_images,
            )
            for idx, (candidate, ir) in enumerate(selected)
        ]
        records.sort(key=lambda row: str(row["inchikey"]))
        for idx, row in enumerate(records):
            row["global_row_id"] = idx
        if len({row["inchikey"] for row in records}) != len(records):
            raise BuildError("deduplication invariant failed")
        all_records = records
        manifest: dict[str, Any] = {
            "schema_version": f"SDBS_final.{args.ir_mode}.v1",
            "row_count": len(all_records),
            "unique_inchikey_count": len({row["inchikey"] for row in all_records}),
            "source_zip": str(zip_path),
            "source_zip_size_bytes": zip_path.stat().st_size,
            "source_zip_sha256": _sha256(zip_path),
            "split_policy": "none; downstream zero-shot evaluation assigns its own split",
            "sequence_filtering": "none",
            "row_selection": "one deterministic AIST H/C/MS + numeric JDX row per InChIKey",
            "image_policy": (
                "ignored: no IR image table read, no image gate, and no image metadata emitted"
                if args.ignore_images
                else "legacy: IR image presence gate and image metadata retained"
            ),
            "aist": aist_stats,
            "jdx": {
                **jdx_stats,
                # These counts cover every candidate JDX parsed while choosing
                # one spectrum per molecule; they are not final-row counts.
                "candidate_jdx_parse_counts": dict(unit_counts),
                "selected_jdx_count": len(selected),
                "selected_record_counts": {
                    "xunits": dict(Counter(row["ir_xunits"] for row in all_records)),
                    "yunits": dict(Counter(row["ir_yunits"] for row in all_records)),
                    "transforms": dict(Counter(row["ir_transform"] for row in all_records)),
                    "transmittance_scales": dict(
                        Counter(row["ir_transmittance_scale"] for row in all_records if row["ir_transmittance_scale"])
                    ),
                    "transmittance_clipped_points": {
                        "low": int(sum(row["ir_transmittance_clipped_low"] for row in all_records)),
                        "high": int(sum(row["ir_transmittance_clipped_high"] for row in all_records)),
                    },
                },
                "no_numeric_jdx_keys": len(no_jdx_keys),
                "coverage": {
                    "min": float(min((row["ir_coverage"] for row in all_records), default=0.0)),
                    "median": float(np.median([row["ir_coverage"] for row in all_records])) if all_records else 0.0,
                    "p10": float(np.percentile([row["ir_coverage"] for row in all_records], 10)) if all_records else 0.0,
                    "p90": float(np.percentile([row["ir_coverage"] for row in all_records], 90)) if all_records else 0.0,
                    "full_grid_rows": int(sum(row["ir_coverage"] >= 0.999999 for row in all_records)),
                },
            },
            "ir_representation": {
                "mode": args.ir_mode,
                "simulated_semantics": "absorbance/intensity-like positive peaks (A)",
                "grid_cm1": [400.0, 4000.0, 400],
                "absorbance": "direct",
                "transmittance": (
                    "T fraction/percent normalized, then A=-log10(max(T, 1e-4)) before "
                    "legacy min-shift/max-scale"
                    if args.ir_mode == "strict_absorbance"
                    else "T converted to 100-T before legacy min-shift/max-scale"
                ),
                "transmittance_floor": TRANSMITTANCE_FLOOR if args.ir_mode == "strict_absorbance" else None,
                "transmittance_scale_detection": "99th percentile <= 1.5 means fraction; otherwise percent",
                "edge_interpolation": "nearest endpoint hold; ir_valid_mask records measured coverage",
                "integer_range": [0, 100],
            },
            "ms_representation": "SDBS EI at 75 eV; no fabricated E10/E20/E40 tags",
            "nmr_representation": {
                "h": "all numeric shift_data points, singleton m 1H tokens",
                "c": "all shift_data points rounded to one decimal",
                "j_values": "not present in SDBS numeric tables; represented as None",
            },
            "frozen_config_compatibility": {
                "config": "paper_vittt_ailab/configs/data/multimodal/hc_ir_ms_generation_irmlp.yaml",
                "required_columns": ["molecular_formula", "spectrum", "h_nmr_peaks", "c_nmr_peaks", "ir_400", "smiles"],
                "ir_patch_size": 16,
            },
        }

    # Build in a sibling temporary directory so an interrupted run never
    # leaves a directory that looks complete.
    temporary = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.building.", dir=str(output_dir.parent)))
    try:
        _write_lines(temporary / "src.txt", all_records, "src_line")
        _write_lines(temporary / "tgt.txt", all_records, "tgt_line")
        _write_parquet(temporary / "records.parquet", all_records)
        manifest["files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_readme(temporary / "README.md", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()

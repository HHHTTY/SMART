#!/usr/bin/env python3
"""Build Chemical Education zero-shot records from repicked MestReNova peaks.

The v2 archive contains the original JDX files plus explicit 1H multiplet and
13C peak tables.  This builder keeps the model-facing schema used by the
existing education evaluation, while recording that the NMR values came from
the repicked tables rather than from curve-derived estimates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from build_education_zeroshot import (
    canonicalize_smiles,
    fetch_smiles,
    heavy_atom_count,
    normalize_id,
)


TARGET_HAC_MIN = 5
TARGET_HAC_MAX = 35


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    text = archive.read(member).decode("utf-8-sig", "replace")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def primary_page_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], str | None]:
    """Keep the earliest page, which is the main Proton/Carbon spectrum export."""

    page_values = []
    for row in rows:
        try:
            page_values.append(int(str(row.get("page_item_index", "")).strip()))
        except (TypeError, ValueError):
            continue
    if not page_values:
        return rows, None
    page = min(page_values)
    return [row for row in rows if str(row.get("page_item_index", "")).strip() == str(page)], str(page)


def finite(value: object) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer_or_float(value: object) -> int | float:
    result = finite(value)
    if result is None:
        return 1
    rounded = round(result)
    return int(rounded) if abs(result - rounded) < 1e-6 else result


def read_multiplets(
    archive: zipfile.ZipFile,
    member: str,
    primary_page_only: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    peaks: list[dict[str, Any]] = []
    rows = read_tsv(archive, member)
    primary_page = None
    if primary_page_only:
        rows, primary_page = primary_page_rows(rows)
    for row in rows:
        range_min = finite(row.get("range_min"))
        range_max = finite(row.get("range_max"))
        if range_min is None or range_max is None:
            continue
        if range_min > range_max:
            range_min, range_max = range_max, range_min
        peaks.append(
            {
                "rangeMax": range_max,
                "rangeMin": range_min,
                "category": (row.get("category") or "m").strip() or "m",
                "nH": integer_or_float(row.get("nH")),
                # The official checkpoint uses j_values=False.  Keep the
                # field for schema compatibility without injecting J tokens.
                "j_values": "None",
            }
        )
    # MestReNova emits the multiplets in descending chemical-shift order;
    # retain that deterministic order for the positional tokenizer.
    peaks.sort(key=lambda peak: (-float(peak["rangeMax"]), -float(peak["rangeMin"])))
    return peaks, primary_page


def read_carbon_peaks(
    archive: zipfile.ZipFile,
    member: str,
    primary_page_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int], str | None]:
    peaks: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    rows = read_tsv(archive, member)
    primary_page = None
    if primary_page_only:
        rows, primary_page = primary_page_rows(rows)
    for row in rows:
        peak_type = (row.get("type") or "").strip()
        type_counts[peak_type or "<empty>"] = type_counts.get(peak_type or "<empty>", 0) + 1
        # Solvent, artifact, reference, and impurity resonances are present in
        # the export but are not molecular evidence for the target compound.
        if peak_type and peak_type.lower() != "compound":
            continue
        ppm = finite(row.get("ppm"))
        if ppm is None:
            continue
        peaks.append({"delta (ppm)": ppm})
    # Keep the source table's usual descending ppm convention.
    peaks.sort(key=lambda peak: -float(peak["delta (ppm)"]))
    return peaks, type_counts, primary_page


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmr-peak-zip", type=Path, required=True)
    parser.add_argument("--ir-parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument(
        "--primary-page-only",
        action="store_true",
        help="Use only the earliest page in each MestReNova peak table (main 1H/13C page).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    ir_frame = pd.read_parquet(args.ir_parquet)
    ir_by_id = {str(row.source_record_id): row for row in ir_frame.itertuples(index=False)}
    resolution_cache: dict[str, str] = {}
    if args.cache and args.cache.exists():
        resolution_cache = json.loads(args.cache.read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(args.nmr_peak_zip) as archive:
        members = {normalize_id(Path(info.filename).stem): info.filename for info in archive.infolist()}
        for index, row in enumerate(manifest_rows, 1):
            source_id = str(row.get("source_record_id", ""))
            hac = heavy_atom_count(row.get("formula", ""))
            claims_nmr = row.get("H-1_claim", "") == "1" and row.get("C-13_claim", "") == "1"
            selected = claims_nmr and TARGET_HAC_MIN < hac < TARGET_HAC_MAX
            smiles_raw, smiles_source = fetch_smiles(
                row.get("cas", ""), resolution_cache, row.get("compound_name", "")
            )
            smiles = canonicalize_smiles(smiles_raw)
            source_key = normalize_id(source_id)
            h_member = members.get(source_key + "_1h_multiplets")
            c_member = members.get(source_key + "_13c_peaks")
            h_peaks, h_primary_page = (
                read_multiplets(archive, h_member, args.primary_page_only)
                if h_member
                else ([], None)
            )
            c_peaks, c_type_counts, c_primary_page = (
                read_carbon_peaks(archive, c_member, args.primary_page_only)
                if c_member
                else ([], {}, None)
            )
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
                "h_peak_member": h_member,
                "c_peak_member": c_member,
                "h_primary_page": h_primary_page,
                "c_primary_page": c_primary_page,
                "h_multiplet_count": len(h_peaks),
                "c_peak_count_compound": len(c_peaks),
                "c_type_counts": c_type_counts,
                "missing_reason": None,
            }
            if h_member is None or c_member is None:
                record["missing_reason"] = "missing_peak_table_member"
            elif not h_peaks or not c_peaks:
                record["missing_reason"] = "empty_peak_table"
            if smiles is None:
                record["missing_reason"] = record["missing_reason"] or "missing_smiles"
            if source_id not in ir_by_id:
                record["missing_reason"] = record["missing_reason"] or "missing_ir"
            record["h_nmr_peaks"] = h_peaks
            record["c_nmr_peaks"] = c_peaks
            records.append(record)
            if index % 20 == 0:
                print(f"prepared {index}/{len(manifest_rows)}", flush=True)

    if args.cache:
        args.cache.write_text(json.dumps(resolution_cache, indent=2, sort_keys=True), encoding="utf-8")

    frame_rows: list[dict[str, Any]] = []
    for record in records:
        if not record["paper_hac_filter"]:
            continue
        ir_row = ir_by_id.get(record["source_record_id"])
        if ir_row is None or record["smiles"] is None:
            continue
        if not record["h_nmr_peaks"] or not record["c_nmr_peaks"]:
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
    output_parquet = data_dir / "education_paired_mnova_repicked_v2.parquet"
    frame.to_parquet(output_parquet, index=False)
    (args.output_dir / "education_records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(records).drop(columns=["h_nmr_peaks", "c_nmr_peaks"], errors="ignore").to_csv(
        args.output_dir / "education_manifest.csv", index=False
    )
    summary = {
        "input_ir_rows": len(manifest_rows),
        "paper_hac_filter_rows": sum(bool(record["paper_hac_filter"]) for record in records),
        "paired_rows_written": len(frame),
        "missing_peak_table_after_filter": sum(
            bool(record["paper_hac_filter"])
            and (not record["h_nmr_peaks"] or not record["c_nmr_peaks"])
            for record in records
        ),
        "missing_smiles_after_filter": sum(
            bool(record["paper_hac_filter"]) and record["smiles"] is None for record in records
        ),
        "ir_length_counts": frame["ir_spectra"].map(len).value_counts().to_dict() if len(frame) else {},
        "nmr_peak_source": "MestReNova repicked *_1H_multiplets.tsv and *_13C_peaks.tsv",
        "h_peak_policy": (
            "earliest page only; all exported MAA multiplets on that page; "
            "j_values disabled by official tokenizer"
            if args.primary_page_only
            else "all exported MAA multiplets; j_values disabled by official tokenizer"
        ),
        "c_peak_policy": (
            "earliest page only; type=Compound; solvent/artifact/reference/impurity excluded"
            if args.primary_page_only
            else "type=Compound only; solvent/artifact/reference/impurity excluded"
        ),
        "primary_page_only": args.primary_page_only,
        "paper_filter": "H-1_claim=C-13_claim=1 and 5 < heavy_atom_count < 35",
        "nmr_peak_zip": str(args.nmr_peak_zip),
        "nmr_peak_zip_sha256": sha256(args.nmr_peak_zip),
        "nmr_peak_zip_size_bytes": args.nmr_peak_zip.stat().st_size,
        "source_archive_records": len(records),
    }
    (args.output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

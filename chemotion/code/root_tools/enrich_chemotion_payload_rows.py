#!/usr/bin/env python3
"""Create a small, non-destructive copy of selected Chemotion payload rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
    index = sidecar["observation_index"]
    requested = {(r["chemotion_record_id"], r["observation_id"]): r for r in index}
    if len(requested) != len(index):
        raise ValueError("duplicate (chemotion_record_id, observation_id) in sidecar")

    payload_partitions = {
        ("nmr_1h", "p0sup_91cb45fd68dd9a5eb9ed29244866bf87e9eddafc"):
            "nmr_1h/part-000086.parquet",
        ("nmr_1h", "p0sup_31950e27325ac275b8aa6e385fbb83b1ac4140b8"):
            "nmr_1h/part-000086.parquet",
        ("nmr_13c", "434dc55543f77759cbcd8cc5dd7b0dd008b708a4"):
            "nmr_13c/part-aux-numeric-suppressed-20260706.parquet",
        ("nmr_13c", "p0sup_ca0eac6e68f04a3055b7592f63fc9ad03305e384"):
            "nmr_13c/part-000088.parquet",
        ("ms", "c65aca33f3a3501a7e0eccf27aafe7190b3c3bf3"):
            "ms/part-000067.parquet",
        ("ms", "p0sup_28fd5ce31cbe3900d38c14f5cda81814e17133ef"):
            "ms/part-000068.parquet",
    }

    found: dict[tuple[str, str], dict[str, Any]] = {}
    input_files: dict[str, str] = {}
    sidecar_modality = {"nmr_1h": "h", "nmr_13c": "c", "ms": "ms"}
    for (modality, observation_id), relative in payload_partitions.items():
        payload_path = args.payload_root / relative
        parquet_file = pq.ParquetFile(payload_path)
        table = None
        for row_group in range(parquet_file.num_row_groups):
            identity = parquet_file.read_row_group(
                row_group,
                columns=["source_record_id", "observation_id"],
            )
            matches = [
                i for i, (record_id, obs_id) in enumerate(
                    zip(identity["source_record_id"].to_pylist(), identity["observation_id"].to_pylist())
                )
                if obs_id == observation_id and (record_id, obs_id) in requested
            ]
            if matches:
                full_group = parquet_file.read_row_group(row_group)
                table = full_group.take(pa.array(matches, type=pa.int64()))
                break
        if table is None:
            continue
        for row in table.to_pylist():
            key = (row["source_record_id"], row["observation_id"])
            metadata = requested.get(key)
            if metadata is None:
                continue
            if metadata["modality"] != sidecar_modality[modality]:
                raise ValueError(f"modality mismatch for {key}: {metadata['modality']} != {sidecar_modality[modality]}")
            if key in found:
                raise ValueError(f"duplicate exact payload row: {key}")
            row["source_payload_file"] = relative
            row["global_row_id"] = metadata["global_row_id"]
            row["raw_package_id"] = metadata["raw_package_id"]
            row["sample_id"] = metadata["sample_id"]
            row["package_url"] = metadata["package_url"]
            row["raw_dataset_id"] = metadata["dataset_id"]
            row["raw_analysis_id"] = metadata["analysis_id"]
            row["raw_dataset_doi"] = metadata["dataset_doi"]
            row["raw_nucleus"] = metadata.get("nucleus")
            row["raw_solvent"] = metadata.get("solvent")
            row["raw_observe_frequency_mhz"] = metadata.get("observe_frequency_mhz")
            row["raw_instrument"] = metadata.get("instrument")
            row["raw_acquisition_json"] = json.dumps(metadata.get("acquisition"), ensure_ascii=False, sort_keys=True)
            row["raw_processing_json"] = json.dumps(metadata.get("processing"), ensure_ascii=False, sort_keys=True)
            row["raw_metadata_source_paths_json"] = json.dumps(metadata["metadata_source_paths"], ensure_ascii=False)
            found[key] = row
            input_files[relative] = sha256(payload_path)

    missing = set(requested) - set(found)
    if missing:
        raise ValueError(f"expected exact payload rows not found: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ordered = [found[(r["chemotion_record_id"], r["observation_id"])] for r in index]
    all_columns = list(dict.fromkeys(key for row in ordered for key in row))
    normalized = [{key: row.get(key) for key in all_columns} for row in ordered]
    output_table = pa.Table.from_pylist(normalized)
    parquet_path = args.output_dir / "enriched_payload_rows.parquet"
    csv_path = args.output_dir / "enriched_payload_rows.csv"
    pq.write_table(output_table, parquet_path, compression="zstd")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_columns)
        writer.writeheader()
        for row in normalized:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})

    verified = pq.read_table(parquet_path)
    identities = {(r["source_record_id"], r["observation_id"]) for r in verified.select(["source_record_id", "observation_id"]).to_pylist()}
    if identities != set(requested) or verified.num_rows != len(requested):
        raise ValueError("written payload-row copy failed identity/count validation")
    manifest = {
        "artifact": "selected Chemotion legacy payload rows enriched from raw publication packages",
        "schema_version": "chemotion.enriched_payload_rows.v1",
        "input_payload_root": str(args.payload_root),
        "input_sidecar": str(args.sidecar),
        "join_keys": ["source_record_id == chemotion_record_id", "observation_id"],
        "row_count": len(ordered),
        "rows_by_modality": {
            modality: sum(1 for r in ordered if r["modality"] == modality)
            for modality in sorted({r["modality"] for r in ordered})
        },
        "unique_record_count": len({r["source_record_id"] for r in ordered}),
        "exact_join_validated": True,
        "payload_input_files_sha256": input_files,
        "output_files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in (parquet_path, csv_path)
        },
        "preservation": "Only new output files were written; source payload partitions and existing datasets were read-only.",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(ordered), "modalities": manifest["rows_by_modality"], "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge only the corrected 13C fields into an existing Chemotion parquet.

The peak-picked builder is intentionally independent from the original
model-compatible export.  This merger keeps the original target and all
non-C modalities byte-for-byte at the row-value level while replacing the C
peak representation and the source text that contains it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


C_FIELDS = (
    "c_nmr_peaks",
    "c_nmr_raw_json",
    "c_nmr_text",
    "c_nmr_representation",
    "c_nmr_observation_id",
    "source_line",
    "src_line",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"empty parquet: {path}")
    return rows


def observation_ids(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return {str(key): str(item) for key, item in parsed.items()} if isinstance(parsed, dict) else {}


def merge_rows(base_rows: list[dict[str, Any]], corrected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_id = {str(row.get("chemotion_record_id")): row for row in base_rows}
    corrected_by_id = {str(row.get("chemotion_record_id")): row for row in corrected_rows}
    if set(base_by_id) != set(corrected_by_id):
        missing = sorted(set(base_by_id) - set(corrected_by_id))[:5]
        extra = sorted(set(corrected_by_id) - set(base_by_id))[:5]
        raise ValueError(f"record-id mismatch; missing={missing}, extra={extra}")

    ordered = sorted(base_rows, key=lambda row: int(row.get("global_row_id", 0)))
    merged_rows: list[dict[str, Any]] = []
    for base in ordered:
        record_id = str(base["chemotion_record_id"])
        corrected = corrected_by_id[record_id]
        merged = dict(base)
        for field in C_FIELDS:
            if field not in corrected:
                raise ValueError(f"corrected parquet lacks {field}: {record_id}")
            merged[field] = corrected[field]

        old_obs = observation_ids(base.get("chemotion_observation_ids"))
        new_obs = observation_ids(corrected.get("chemotion_observation_ids"))
        if not old_obs or not new_obs or "c" not in new_obs:
            raise ValueError(f"invalid observation-id payload: {record_id}")
        old_obs["c"] = new_obs["c"]
        merged["chemotion_observation_ids"] = json.dumps(
            old_obs, sort_keys=True, separators=(",", ":")
        )
        merged_rows.append(merged)
    return merged_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="existing model-compatible dataset directory")
    parser.add_argument("--corrected", type=Path, required=True, help="peak-picked builder output directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    base = args.base.expanduser().resolve()
    corrected = args.corrected.expanduser().resolve()
    base_rows = load_rows(base / "test.parquet")
    corrected_rows = load_rows(corrected / "test.parquet")
    merged_rows = merge_rows(base_rows, corrected_rows)

    output.mkdir(parents=True, exist_ok=False)
    table = pa.Table.from_pylist(merged_rows)
    test_path = output / "test.parquet"
    pq.write_table(table, test_path, compression="zstd")
    (output / "src.txt").write_text(
        "\n".join(str(row["src_line"]) for row in merged_rows) + "\n", encoding="utf-8"
    )
    (output / "tgt.txt").write_text(
        "\n".join(str(row["tgt_line"]) for row in merged_rows) + "\n", encoding="utf-8"
    )

    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    corrected_manifest = json.loads((corrected / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": base_manifest.get("schema_version", "Chemotion.ir1800.v1"),
        "dataset": output.name,
        "source_dataset": base_manifest.get("source_dataset"),
        "payload_root": corrected_manifest.get("payload_root"),
        "row_count": len(merged_rows),
        "row_limit": None,
        "unique_inchikey_count": len({row["inchikey"] for row in merged_rows}),
        "files": {
            "test.parquet": {"bytes": test_path.stat().st_size, "sha256": sha256(test_path)},
            "src.txt": {"bytes": (output / "src.txt").stat().st_size},
            "tgt.txt": {"bytes": (output / "tgt.txt").stat().st_size},
        },
        "ir_representation": base_manifest.get("ir_representation", {}),
        "selection": {
            "record_key": "source_record_id",
            "base_modality_selection": base_manifest.get("selection", {}).get("modality_selection"),
            "c_selection_policy": corrected_manifest.get("selection", {}).get("c_selection_policy"),
            "c_submodality_priority": corrected_manifest.get("selection", {}).get("c_submodality_priority", {}),
            "duplicate_policy": base_manifest.get("selection", {}).get("duplicate_policy"),
        },
        "merge": {
            "base_dataset": str(base),
            "corrected_dataset": str(corrected),
            "changed_columns": [*C_FIELDS, "chemotion_observation_ids"],
            "preserved_structure_and_target": True,
            "preserved_modalities": ["nmr_1h", "ir", "ms"],
        },
        "audit": corrected_manifest.get("audit", {}),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(merged_rows),
                "changed_columns": [*C_FIELDS, "chemotion_observation_ids"],
                "base_sha256": sha256(base / "test.parquet"),
                "corrected_sha256": sha256(corrected / "test.parquet"),
                "merged_sha256": sha256(test_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

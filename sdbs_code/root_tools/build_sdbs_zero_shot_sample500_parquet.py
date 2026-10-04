#!/usr/bin/env python3
"""Materialize the saved SpectraLLM 500-row zero-shot subset as a parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    row_ids = [int(value) for value in sample["global_row_ids"]]
    if len(row_ids) != 500 or len(set(row_ids)) != 500:
        raise ValueError("sample manifest must contain exactly 500 unique global_row_ids")

    frame = pd.read_parquet(args.input).set_index("global_row_id", drop=False)
    missing = [row_id for row_id in row_ids if row_id not in frame.index]
    if missing:
        raise ValueError(f"sample manifest IDs absent from input: {missing[:5]}")
    selected = frame.loc[row_ids].reset_index(drop=True)
    if selected["global_row_id"].astype(int).tolist() != row_ids:
        raise RuntimeError("materialized parquet row order does not match the saved manifest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    selected.to_parquet(temporary, index=False)
    temporary.replace(args.output)
    output_manifest = {
        "dataset": str(args.output),
        "rows": len(selected),
        "source_dataset": str(args.input),
        "source_sample_manifest": str(args.sample_manifest),
        "source_sample_manifest_sha256": sha256(args.sample_manifest),
        "seed": int(sample["seed"]),
        "sample_size": int(sample["sample_size"]),
        "global_row_ids": row_ids,
        "sdbs_no": [str(value) for value in selected["sdbs_no"]],
        "target_labels_included_for_offline_scoring_only": True,
        "row_order_matches_manifest": True,
        "sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(output_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: output_manifest[key] for key in (
        "dataset", "rows", "seed", "source_sample_manifest_sha256", "sha256"
    )}, indent=2))


if __name__ == "__main__":
    main()

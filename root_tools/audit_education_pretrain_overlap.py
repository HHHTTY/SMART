#!/usr/bin/env python3
"""Audit exact canonical-SMILES overlap between an evaluation set and parquet shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger


def canonicalize(value: object) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(value))
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-data", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*.parquet")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    evaluation = pd.read_parquet(args.evaluation_data)
    targets = {
        canonicalize(value): str(record_id)
        for value, record_id in zip(evaluation["smiles"], evaluation["source_record_id"])
    }
    targets.pop(None, None)
    counts: Counter[str] = Counter()
    matched_files: dict[str, list[str]] = {}
    files = sorted(args.source_dir.glob(args.pattern))
    total_rows = 0
    for index, path in enumerate(files, 1):
        frame = pd.read_parquet(path, columns=["smiles"])
        canonical_values = frame["smiles"].map(canonicalize)
        total_rows += len(canonical_values)
        for value in canonical_values.dropna():
            if value in targets:
                counts[value] += 1
                matched_files.setdefault(value, []).append(path.name)
        if index % 25 == 0 or index == len(files):
            print(f"scanned {index}/{len(files)} shards; rows={total_rows}; unique target matches={len(counts)}", flush=True)

    result = {
        "evaluation_data": str(args.evaluation_data),
        "source_dir": str(args.source_dir),
        "source_pattern": args.pattern,
        "evaluation_rows": len(evaluation),
        "unique_valid_evaluation_targets": len(targets),
        "source_shards": len(files),
        "source_rows": total_rows,
        "matched_unique_targets": len(counts),
        "matched_evaluation_source_ids": sorted(targets[value] for value in counts),
        "matched_canonical_smiles": sorted(counts),
        "source_occurrence_counts": {value: counts[value] for value in sorted(counts)},
        "source_files_by_target": {value: sorted(set(matched_files[value])) for value in sorted(matched_files)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"matched_canonical_smiles", "source_occurrence_counts", "source_files_by_target"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()

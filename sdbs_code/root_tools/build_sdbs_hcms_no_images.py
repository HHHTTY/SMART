#!/usr/bin/env python3
"""Build the maximum table-only SDBS H/C/MS view without image data.

This is deliberately separate from the IR-complete builder.  The archive's
H/C/MS CSV tables provide 9k+ records, while the numeric IR JDX files cover a
smaller subset.  No IR placeholder is fabricated here; rows are marked with
``has_IR=False`` so this view cannot be mistaken for a full-modality dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import build_sdbs_final as legacy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_record(candidate: dict, row_id: int) -> dict:
    inchi = candidate.get("inchi", "")
    molecule = legacy.Chem.MolFromInchi(inchi)
    if molecule is None:
        raise legacy.BuildError(f"{candidate['sid']}: RDKit cannot parse InChI")
    computed_key = legacy.Chem.MolToInchiKey(molecule)
    if computed_key.upper() != str(candidate["key"]).upper():
        raise legacy.BuildError(
            f"{candidate['sid']}: InChIKey mismatch ({candidate['key']} vs {computed_key})"
        )
    smiles = legacy.Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    formula = legacy.rdMolDescriptors.CalcMolFormula(molecule)
    formula_spaced = legacy._spaced_formula(formula)
    smiles_tokenized = legacy._tokenized_smiles(smiles)
    h_text = legacy._h_text(candidate["h_peaks"])
    c_text = legacy._c_text(candidate["c_peaks"])
    ms_text = legacy._ms_text(candidate["ms_peaks"])
    source_line = " ".join([formula_spaced, h_text, c_text, "MS", ms_text])
    return {
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
        "spectrum": ms_text,
        "ms_spectrum": ms_text,
        "h_nmr_text": h_text,
        "c_nmr_text": c_text,
        "ir_text": "",
        "spectrum_semantics": "EI_75eV",
        "h_nmr_representation": "all_numeric_shift_data_as_singleton_multiplets",
        "source_line": source_line,
        "src_line": source_line,
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
        "has_IR": False,
        "has_MS": True,
    }


def _write_parquet(path: Path, records: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")


def _write_lines(path: Path, records: list[dict], field: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(str(row[field]).rstrip("\n") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=9546)
    args = parser.parse_args()

    zip_path = args.zip_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates, aist_stats = legacy._load_aist(
            zf, use_image_gate=False, include_image_metadata=False
        )
        candidate_by_key: dict[str, list[dict]] = {}
        for candidate in candidates:
            candidate_by_key.setdefault(str(candidate["key"]), []).append(candidate)
        selected: list[dict] = []
        for key in sorted(candidate_by_key):
            options = sorted(
                candidate_by_key[key],
                key=lambda row: (
                    -(len(row["h_peaks"]) + len(row["c_peaks"]) + len(row["ms_peaks"])),
                    legacy._numeric_sid(str(row["sid"])),
                ),
            )
            selected.append(options[0])
        records: list[dict] = []
        excluded_invalid_structure: list[dict[str, str]] = []
        for candidate in selected:
            try:
                records.append(_make_record(candidate, len(records)))
            except legacy.BuildError as exc:
                if not any(
                    marker in str(exc)
                    for marker in ("RDKit cannot parse InChI", "InChIKey mismatch")
                ):
                    raise
                excluded_invalid_structure.append(
                    {
                        "sid": str(candidate["sid"]),
                        "inchikey": str(candidate["key"]),
                        "reason": str(exc),
                    }
                )
        if args.expected_count and len(records) != args.expected_count:
            raise legacy.BuildError(
                f"materialized {len(records)} valid structures, expected {args.expected_count}; "
                f"joined candidates={len(selected)}"
            )
        records.sort(key=lambda row: str(row["inchikey"]))
        for idx, row in enumerate(records):
            row["global_row_id"] = idx

        manifest = {
            "schema_version": "SDBS_hcms_no_images.v1",
            "row_count": len(records),
            "unique_inchikey_count": len({row["inchikey"] for row in records}),
            "source_zip": str(zip_path),
            "source_zip_size_bytes": zip_path.stat().st_size,
            "source_zip_sha256": _sha256(zip_path),
            "image_policy": "ignored: no IR image table read, no image metadata emitted",
            "row_selection": "one deterministic shared-SID H/C/MS row per InChIKey",
            "joined_candidate_count": len(selected),
            "excluded_invalid_or_mismatched_structure_count": len(excluded_invalid_structure),
            "excluded_invalid_or_mismatched_structure": excluded_invalid_structure,
            "aist": aist_stats,
            "ir_policy": "IR is intentionally absent; no PNG, IR image metadata, or JDX values are used",
            "modality_counts": {"1HNMR": len(records), "13CNMR": len(records), "MS": len(records), "IR": 0},
            "files": {},
        }

    temporary = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.building.", dir=str(output_dir.parent)))
    try:
        _write_lines(temporary / "src.txt", records, "src_line")
        _write_lines(temporary / "tgt.txt", records, "tgt_line")
        _write_parquet(temporary / "records.parquet", records)
        manifest["files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            "# SDBS_hcms_no_images\n\n"
            "This is the maximum table-only H/C/MS view from SDBS.zip. It does not "
            "contain numerical IR; use the separate IR1800 build for rows with a "
            "reliably matched numeric JDX spectrum.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

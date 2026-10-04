#!/usr/bin/env python3
"""Build an SDBS view with the continuous 1800-point IR contract.

The existing ``build_sdbs_final.py`` intentionally materialises a legacy
400-bin integer IR field.  The multimodal pretraining checkpoint, however,
was fitted on ``ir_spectra``: a continuous 1800-point spectrum consumed as
24 patches of width 75.  SDBS JDX files have heterogeneous native lengths, so
this script keeps the native JDX metadata and deterministically interpolates
each accepted spectrum onto the same 400--4000 cm^-1 grid before per-spectrum
min-shift/max scaling to the simulated [0, 1] intensity convention.

The legacy dataset is never overwritten.  This script reuses the audited AIST
and JDX join implemented by ``build_sdbs_final.py`` and only changes the IR
grid/output contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

import build_sdbs_final as legacy


GRID_SIZE = 1800
GRID_START = 400.0
GRID_END = 4000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scaled_ir(record: legacy.JDXRecord) -> list[float]:
    """Convert one interpolated JDX signal to the simulated float convention."""
    values = np.asarray(record.grid_signal, dtype=np.float64)
    if values.shape != (GRID_SIZE,) or not np.isfinite(values).all():
        raise legacy.BuildError(
            f"IR {record.filename}: expected a finite {GRID_SIZE}-point vector"
        )
    shifted = values + abs(float(values.min()))
    maximum = float(shifted.max(initial=0.0))
    if maximum <= 0.0:
        return [0.0] * GRID_SIZE
    scaled = np.clip(shifted / maximum, 0.0, 1.0)
    return scaled.astype(np.float32).tolist()


def _make_record(
    candidate: dict,
    ir: legacy.JDXRecord,
    row_id: int,
    *,
    include_image_metadata: bool = True,
) -> dict:
    """Materialise the same metadata as SDBS_final with ``ir_spectra``."""
    inchi = candidate.get("inchi") or ir.inchi
    molecule = legacy.Chem.MolFromInchi(inchi)
    if molecule is None:
        raise legacy.BuildError(f"{candidate['sid']}: RDKit cannot parse InChI")
    computed_key = legacy.Chem.MolToInchiKey(molecule)
    if computed_key and computed_key.upper() != str(candidate["key"]).upper():
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
    ir_values = _scaled_ir(ir)
    ir_text_values = " ".join(f"{value:.8g}" for value in ir_values)
    src_line = " ".join([formula_spaced, h_text, c_text, "IR", ir_text_values, "MS", ms_text])

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
        "spectrum": ms_text,
        "ms_spectrum": ms_text,
        "h_nmr_text": h_text,
        "c_nmr_text": c_text,
        "ir_text": "IR " + ir_text_values,
        "ir_spectra": ir_values,
        "ir_1800": ir_values,
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
    parser.add_argument("--expected-count", type=int, default=3669)
    parser.add_argument(
        "--ir-mode",
        choices=legacy.IR_MODES,
        default="simulated_complement",
        help="JDX transmittance conversion; keep the legacy default for checkpoint parity.",
    )
    parser.add_argument(
        "--ignore-images",
        action="store_true",
        help=(
            "Do not read IR image metadata or apply the IR-image presence gate; "
            "use only H/C/MS tables and numeric JDX values."
        ),
    )
    args = parser.parse_args()

    zip_path = args.zip_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # The reused parser reads this module-level grid when it creates each
    # JDXRecord.  Set it before opening the archive and never mutate it again.
    legacy.GRID = np.linspace(GRID_START, GRID_END, GRID_SIZE, dtype=np.float64)

    with legacy.zipfile.ZipFile(zip_path, "r") as zf:
        candidates, aist_stats = legacy._load_aist(
            zf,
            use_image_gate=not args.ignore_images,
            include_image_metadata=not args.ignore_images,
        )
        jdx_index, jdx_stats = legacy._load_jdx_index(zf)
        candidate_by_key: dict[str, list[dict]] = {}
        for candidate in candidates:
            candidate_by_key.setdefault(str(candidate["key"]), []).append(candidate)

        cache: dict[str, legacy.JDXRecord | None] = {}
        unit_counts: Counter[str] = Counter()
        member_names = set(zf.namelist())
        selected: list[tuple[dict, legacy.JDXRecord]] = []
        no_jdx_keys: list[str] = []
        for key in sorted(candidate_by_key):
            ir = legacy._choose_jdx(
                zf,
                key,
                jdx_index.get(key, []),
                cache,
                unit_counts,
                member_names,
                args.ir_mode,
            )
            if ir is None:
                no_jdx_keys.append(key)
                continue
            options = sorted(
                candidate_by_key[key],
                key=lambda row: (
                    -(len(row["h_peaks"]) + len(row["c_peaks"]) + len(row["ms_peaks"])),
                    legacy._numeric_sid(str(row["sid"])),
                ),
            )
            selected.append((options[0], ir))

        if args.expected_count and len(selected) != args.expected_count:
            raise legacy.BuildError(
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

        manifest = {
            "schema_version": "SDBS_final.ir1800.v1",
            "row_count": len(records),
            "unique_inchikey_count": len({row["inchikey"] for row in records}),
            "source_zip": str(zip_path),
            "source_zip_size_bytes": zip_path.stat().st_size,
            "source_zip_sha256": _sha256(zip_path),
            "split_policy": "none; reuse locked SDBS_final test IDs downstream",
            "sequence_filtering": "none",
            "row_selection": "same deterministic AIST H/C/MS + numeric JDX join as SDBS_final.v1",
            "image_policy": (
                "ignored: no IR image table read, no image gate, and no image metadata emitted"
                if args.ignore_images
                else "legacy: IR image presence gate and image metadata retained"
            ),
            "aist": aist_stats,
            "jdx": {
                **jdx_stats,
                "candidate_jdx_parse_counts": dict(unit_counts),
                "selected_jdx_count": len(selected),
                "selected_record_counts": {
                    "xunits": dict(Counter(row["ir_xunits"] for row in records)),
                    "yunits": dict(Counter(row["ir_yunits"] for row in records)),
                    "transforms": dict(Counter(row["ir_transform"] for row in records)),
                },
                "no_numeric_jdx_keys": len(no_jdx_keys),
                "coverage": {
                    "min": float(min((row["ir_coverage"] for row in records), default=0.0)),
                    "median": float(np.median([row["ir_coverage"] for row in records])) if records else 0.0,
                    "p10": float(np.percentile([row["ir_coverage"] for row in records], 10)) if records else 0.0,
                    "p90": float(np.percentile([row["ir_coverage"] for row in records], 90)) if records else 0.0,
                    "full_grid_rows": int(sum(row["ir_coverage"] >= 0.999999 for row in records)),
                },
            },
            "ir_representation": {
                "mode": args.ir_mode,
                "grid_cm1": [GRID_START, GRID_END, GRID_SIZE],
                "grid_spacing_cm1": (GRID_END - GRID_START) / (GRID_SIZE - 1),
                "native_x_axis_retained": True,
                "continuous_float_range": [0.0, 1.0],
                "normalisation": "per-spectrum min-shift then max-scale; no integer quantisation",
                "simulated_semantics": "absorbance/intensity-like positive peaks (A)",
                "patch_contract": {"patch_size": 75, "n_patches": 24},
                "transmittance": (
                    "T converted to 100-T before min-shift/max-scale"
                    if args.ir_mode == "simulated_complement"
                    else "T converted to physical absorbance before min-shift/max-scale"
                ),
            },
            "ms_representation": "SDBS EI at 75 eV; no fabricated E10/E20/E40 tags",
            "nmr_representation": {
                "h": "all numeric shift_data points, singleton m 1H tokens",
                "c": "all shift_data points rounded to one decimal",
                "j_values": "not present in SDBS numeric tables; represented as None",
            },
            "checkpoint_compatibility": {
                "ir_column": "ir_spectra",
                "ir_preprocessor": "PatchPreprocessor(patch_size=75, interpolation=False, masking=False)",
                "text_vocab_policy": "reuse simulated Formula/MSMS/HNMR/CNMR/Smiles vocabularies; no refit",
            },
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
            "# SDBS_final_ir1800\n\n"
            "This dataset preserves the SDBS_final AIST/JDX join while exposing "
            "continuous 1800-point `ir_spectra` values on a 400--4000 cm-1 grid. "
            "Use the simulated checkpoint preprocessor with IR patch_size=75. "
            "The legacy SDBS_final 400-bin dataset is left unchanged.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    # Keep imports working when copied as a standalone tool to the HPC.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

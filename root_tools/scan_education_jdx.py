#!/usr/bin/env python3
"""Audit 1D JDX blocks in the Chemical Education MNova export."""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path


def stem_key(value: str) -> str:
    value = Path(value).stem.lower()
    value = value.replace("benzene", "benzene")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: scan_education_jdx.py <mnova.zip> <ir_manifest.csv> <out.json>")
    zip_path, manifest_path, out_path = map(Path, sys.argv[1:])
    manifest = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")))
    by_key = {stem_key(r["source_record_id"]): r["source_record_id"] for r in manifest}
    by_key.update({stem_key(r["archive_base_name"]): r["source_record_id"] for r in manifest})

    # Import lazily so the audit can still report archive names if the parser is missing.
    try:
        import jcamp  # type: ignore
    except ImportError as exc:
        raise SystemExit("install/use jcamp 1.3.2 (PYTHONPATH=/tmp/jcamp_probe_inst)") from exc

    records: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = [i for i in archive.infolist() if i.filename.lower().endswith(".jdx")]
        for index, info in enumerate(members, 1):
            stem = Path(info.filename).stem
            record: dict[str, object] = {
                "member": info.filename,
                "stem": stem,
                "source_record_id": by_key.get(stem_key(stem)),
                "compressed_size": info.compress_size,
                "file_size": info.file_size,
                "parse_error": None,
                "blocks": [],
            }
            try:
                raw_bytes = archive.read(info)
                try:
                    parsed = jcamp.read(io.BytesIO(raw_bytes))
                except Exception:
                    # Some MestReNova ASDF blocks contain a short/corrupt line that
                    # makes jcamp 1.3.2 abort while the 1D headers and most data are
                    # still usable. Parse the 1D headers and blocks without enforcing
                    # the parser's x/y consistency checks.
                    parsed = {"children": _read_children_lenient(raw_bytes, jcamp)}
                blocks = []
                for child in parsed.get("children", []):
                    nucleus = str(child.get(".observe nucleus", ""))
                    if child.get("block_id") in (1, 2, "1", "2") or nucleus in {"^1H", "^13C"}:
                        blocks.append(
                            {
                                "block_id": child.get("block_id"),
                                "data_type": child.get("data type"),
                                "nucleus": nucleus,
                                "frequency_mhz": child.get(".observe frequency"),
                                "npoints": child.get("npoints"),
                                "x_first_hz": float(child["x"][0]) if len(child.get("x", [])) else None,
                                "x_last_hz": float(child["x"][-1]) if len(child.get("x", [])) else None,
                                "y_min": float(min(child["y"])) if len(child.get("y", [])) else None,
                                "y_max": float(max(child["y"])) if len(child.get("y", [])) else None,
                            }
                        )
                record["blocks"] = blocks
            except Exception as exc:  # keep the audit complete across malformed members
                record["parse_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            if index % 20 == 0:
                print(f"scanned {index}/{len(members)}", flush=True)

    summary = {
        "jdx_members": len(records),
        "matched_to_ir": sum(r["source_record_id"] is not None for r in records),
        "parse_errors": sum(bool(r["parse_error"]) for r in records),
        "records_with_1h": sum(any(str(b.get("nucleus")) == "^1H" for b in r["blocks"]) for r in records),
        "records_with_13c": sum(any(str(b.get("nucleus")) == "^13C" for b in r["blocks"]) for r in records),
        "records_with_both": sum(
            {str(b.get("nucleus")) for b in r["blocks"]} >= {"^1H", "^13C"} for r in records
        ),
        "matched_source_ids_with_both": sorted(
            r["source_record_id"]
            for r in records
            if r["source_record_id"]
            and {str(b.get("nucleus")) for b in r["blocks"]} >= {"^1H", "^13C"}
        ),
    }
    Path(out_path).write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _read_children_lenient(raw_bytes: bytes, jcamp_module):
    """Leniently decode LINK children, retaining 1D NMR blocks."""
    text = raw_bytes.decode("utf-8", "ignore")
    starts = list(re.finditer(r"(?m)^##TITLE=", text))
    children = []
    for pos, match in enumerate(starts):
        block = text[match.start() : starts[pos + 1].start() if pos + 1 < len(starts) else len(text)]
        if "##DATA TYPE=\tNMR SPECTRUM" not in block and "##DATA TYPE=NMR SPECTRUM" not in block:
            continue
        child: dict[str, object] = {}
        for key, value in re.findall(r"(?m)^##([^=]+)=\s*(.*)$", block):
            child[key.strip().lower()] = value.strip()
        if "##XYDATA" not in block:
            children.append(child)
            continue
        # Parse the compressed Y values with jcamp's codec, but skip malformed lines.
        data_start = re.search(r"(?m)^##XYDATA=.*\n", block)
        if data_start is None:
            children.append(child)
            continue
        data_text = block[data_start.end() :]
        data_text = data_text.split("##END", 1)[0]
        lines = [line.strip() for line in data_text.splitlines() if line.strip() and not line.startswith("$$")]
        y: list[float] = []
        asdf = bool(lines and any(c in jcamp_module.DIF_digits for c in lines[0]))
        previous_y: float | None = None
        for line in lines:
            try:
                vals = jcamp_module.parse(line)
            except Exception:
                continue
            if not vals:
                continue
            if asdf:
                if len(vals) < 2:
                    continue
                if previous_y is None:
                    y.extend(float(v) for v in vals[1:])
                else:
                    # vals[1] is the repeated previous ordinate; the rest are differences.
                    if len(vals) >= 3:
                        y.extend(float(v) for v in vals[2:])
                previous_y = y[-1] if y else previous_y
            elif len(vals) >= 2:
                y.extend(float(v) for v in vals[1:])
        try:
            npoints = int(float(child.get("npoints", 0)))
            firstx = float(child.get("firstx"))
            lastx = float(child.get("lastx"))
        except (TypeError, ValueError):
            npoints = 0
            firstx = lastx = 0.0
        child["npoints"] = npoints
        child["x"] = [firstx, lastx] if npoints else []
        child["y"] = y[:npoints] if npoints else y
        children.append(child)
    return children


if __name__ == "__main__":
    main()

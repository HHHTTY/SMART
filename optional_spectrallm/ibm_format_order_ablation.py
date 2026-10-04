#!/usr/bin/env python3
"""Small format-only ablation for the IBM/OpenNMT reproduction.

The script is intended to run on the remote IBM reproduction checkout.  It
keeps the checkpoint, decoder, beam settings, rows, and canonical evaluator
fixed, changing only the representation of the MS segment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        required=True,
        choices=(
            "no_ms", "marker_only", "marker_null", "ms_before_ir",
            "ms_intensity_ge_0p5", "ms_intensity_ge_1p0",
            "ms_gamma_2p4", "ms_gamma_3p2",
            "preserve",
        ),
    )
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--target", type=Path)
    p.add_argument("--run-name", type=str)
    args = p.parse_args()

    sys.path.insert(0, str(args.base / "pipeline_code_routerfix_20260925"))
    import run_iclr_open_nmt_router_pipeline as pipeline

    source = args.source or (args.base / "sdbs_zip_rebuilt_20260924/src.txt")
    target = args.target or (args.base / "sdbs_zip_rebuilt_20260924/tgt.txt")
    rows = pipeline._parse_rows(source, target, args.limit, read_targets=True, ms_energy_mode="all")
    for row in rows:
        if "MSMS" not in row.parts:
            continue
        if args.variant == "no_ms":
            del row.parts["MSMS"]
        elif args.variant == "marker_only":
            row.parts["MSMS"] = ["E0Pos"]
        elif args.variant == "marker_null":
            row.parts["MSMS"] = ["E0Pos", "null"]
        elif args.variant in {"ms_intensity_ge_0p5", "ms_intensity_ge_1p0"}:
            threshold = 0.5 if args.variant.endswith("0p5") else 1.0
            segment = row.parts["MSMS"]
            kept = [segment[0]]
            for offset in range(1, len(segment) - 1, 2):
                try:
                    intensity = float(segment[offset + 1])
                except ValueError:
                    continue
                if intensity >= threshold:
                    kept.extend(segment[offset:offset + 2])
            row.parts["MSMS"] = kept
        elif args.variant in {"ms_gamma_2p4", "ms_gamma_3p2"}:
            # Match the simulator's intensity remapping used by the weak and
            # medium profiles. Keep m/z and peak count unchanged.
            gamma = 2.4 if args.variant.endswith("2p4") else 3.2
            segment = row.parts["MSMS"]
            transformed = [segment[0]]
            for offset in range(1, len(segment) - 1, 2):
                transformed.append(segment[offset])
                try:
                    intensity = max(float(segment[offset + 1]), 0.1)
                    remapped = 100.0 * (intensity / 100.0) ** gamma
                    transformed.append(f"{max(0.1, remapped):.1f}")
                except ValueError:
                    transformed.append(segment[offset + 1])
            row.parts["MSMS"] = transformed
        elif args.variant == "preserve":
            # Keep the source marker and all peak pairs exactly as parsed.
            pass
        else:
            # The official formatter is HNMR -> CNMR -> IR -> MS.  This
            # variant deliberately moves MS before IR while keeping tokens.
            pass

    ckpt = args.base / "iclr/all.pt"
    official = args.base / "official-code"
    load_args = SimpleNamespace(official_code=official, checkpoint=ckpt, device="cuda")
    model, vocabs, device = pipeline._load_onmt(load_args)
    model.eval()
    translator = model._iclr_translator
    translator.beam_size = 10
    translator.n_best = 10
    translator.max_length = 128
    translator.min_length = 5
    support = tuple(pipeline.MODALITIES)

    original_groups = pipeline._source_token_groups
    if args.variant == "ms_before_ir":
        def reordered(row, keep):
            keep_set = set(keep)
            groups = [("FORMULA", row.formula.split())]
            for name in ("HNMR", "CNMR", "MSMS", "IR"):
                if name in keep_set and name in row.parts:
                    groups.append((name, list(row.parts[name])))
            return groups
        pipeline._source_token_groups = reordered

    predictions = []
    lengths = []
    unknown = 0
    tokens = 0
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            source_tensor, lens = pipeline._batch_src(
                batch, vocabs, [support] * len(batch), device, "exact"
            )
            lengths.extend(lens.detach().cpu().tolist())
            for row in batch:
                _, stats = pipeline._canonical_source_tokens(row, support, vocabs["src"], "exact")
                unknown += stats["unknown"]
                tokens += stats["tokens"]
            result = translator.translate_batch({"src": source_tensor, "srclen": lens}, False)
            predictions.extend([
                [pipeline._decode(seq.detach().cpu().tolist(), vocabs["tgt"]) for seq in beam[:10]]
                for beam in result["predictions"]
            ])
            print(json.dumps({"rows": min(start + len(batch), len(rows))}), flush=True)

    targets = [pipeline._canonical_smiles(row.target) for row in rows]
    hits = {1: 0, 5: 0, 10: 0}
    invalid = 0
    records = []
    for row, target_smiles, beam in zip(rows, targets, predictions):
        canonical = [pipeline._canonical_smiles(value) for value in beam]
        invalid += int(not canonical or canonical[0] is None)
        for k in hits:
            hits[k] += int(target_smiles is not None and target_smiles in canonical[:k])
        records.append({"row": row.index, "target": row.target, "predictions": beam})

    run = args.base / (args.run_name or f"format_all_{args.variant}_{args.limit}_20260925")
    run.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "all",
        "variant": args.variant,
        "rows": len(rows),
        "input_view": ["FORMULA", "HNMR", "CNMR", "MSMS", "IR"],
        "beams": 10,
        "top1": hits[1],
        "top5": hits[5],
        "top10": hits[10],
        "top1_rate": hits[1] / max(1, len(rows)),
        "top5_rate": hits[5] / max(1, len(rows)),
        "top10_rate": hits[10] / max(1, len(rows)),
        "unknown_tokens": unknown,
        "source_tokens": tokens,
        "unknown_rate": unknown / max(1, tokens),
        "mean_source_len": sum(lengths) / max(1, len(lengths)),
        "max_source_len": max(lengths),
        "invalid_top1": invalid,
        "checkpoint": str(ckpt),
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run / "predictions.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

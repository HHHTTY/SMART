#!/usr/bin/env python3
"""Evaluate one OpenNMT checkpoint/state on one explicit modality view.

The script deliberately keeps the input contract explicit so that base and
TTT checkpoints can be compared on exactly the same rows and tokenization.
Formula tokens are always included by ``pipeline._batch_src``; ``--view``
therefore names only spectral modalities (for example ``HNMR+CNMR``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import run_iclr_open_nmt_router_pipeline as pipeline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--official-code", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--state-checkpoint", type=Path,
        help="Optional Stage-1/2 state dict loaded on top of --checkpoint.",
    )
    parser.add_argument("--view", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3669)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beams", type=int, default=10)
    parser.add_argument("--max-target-length", type=int, default=250)
    parser.add_argument("--min-target-length", type=int, default=5)
    parser.add_argument("--numeric-token-policy", choices=("exact", "nearest"), default="exact")
    parser.add_argument("--ms-energy-mode", choices=("all", "e1"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    view = tuple(value for value in args.view.replace(",", "+").split("+") if value)
    unknown = set(view).difference(pipeline.MODALITIES)
    if unknown:
        raise ValueError(f"unknown modalities in --view: {sorted(unknown)}")
    support = pipeline.MODEL_SUPPORT[args.checkpoint.stem]
    unsupported = set(view).difference(support)
    if unsupported:
        raise ValueError(
            f"{args.checkpoint.stem} does not support modalities {sorted(unsupported)}"
        )

    load_args = SimpleNamespace(
        official_code=args.official_code,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    model, vocabs, device = pipeline._load_onmt(load_args)
    if args.state_checkpoint is not None:
        pipeline._load_state(model, args.state_checkpoint)
    model.eval()

    rows = pipeline._parse_rows(
        args.source,
        args.target,
        args.limit,
        read_targets=True,
        ms_energy_mode=args.ms_energy_mode,
    )
    missing = [row.index for row in rows if any(name not in row.parts for name in view)]
    if missing:
        raise ValueError(
            f"requested view is unavailable on {len(missing)} rows; first={missing[:10]}"
        )

    predictions: list[list[str]] = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            source, lengths = pipeline._batch_src(
                batch,
                vocabs,
                [view] * len(batch),
                device,
                args.numeric_token_policy,
            )
            predictions.extend(
                pipeline._beam_decode(
                    model,
                    source,
                    lengths,
                    vocabs,
                    args.beams,
                    args.max_target_length,
                    args.min_target_length,
                )
            )
            print(
                json.dumps({"rows": min(start + len(batch), len(rows)), "view": view}),
                flush=True,
            )

    target_canonical = [pipeline._canonical_smiles(row.target) for row in rows]
    hits = {cutoff: 0 for cutoff in (1, 5, 10)}
    invalid_top1 = 0
    formula_match_top1 = 0
    formula_valid_top1 = 0
    prediction_rows = []
    for row, target, candidates_raw in zip(rows, target_canonical, predictions):
        candidates = [pipeline._canonical_smiles(value) for value in candidates_raw]
        top1 = candidates[0] if candidates else None
        invalid_top1 += int(top1 is None)
        if top1 is not None:
            formula_valid_top1 += 1
            formula_match_top1 += int(
                pipeline._smiles_formula_signature(top1)
                == pipeline._formula_signature(row.formula)
            )
        for cutoff in hits:
            hits[cutoff] += int(target is not None and target in candidates[:cutoff])
        if args.save_predictions:
            prediction_rows.append(
                {
                    "source_row_index": row.index,
                    "target": row.target,
                    "predictions": candidates_raw,
                }
            )

    count = max(1, len(rows))
    result = {
        "complete": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "state_checkpoint": str(args.state_checkpoint) if args.state_checkpoint else None,
        "state_checkpoint_sha256": (
            _sha256(args.state_checkpoint) if args.state_checkpoint else None
        ),
        "source": str(args.source),
        "source_sha256": _sha256(args.source),
        "target": str(args.target),
        "target_sha256": _sha256(args.target),
        "rows": len(rows),
        "view": list(view),
        "formula_always_included": True,
        "beams": args.beams,
        "numeric_token_policy": args.numeric_token_policy,
        "ms_energy_mode": args.ms_energy_mode,
        "tokenization_summary": pipeline._tokenization_summary(
            rows,
            vocabs,
            frozenset(view),
            args.numeric_token_policy,
        ),
        "top1": hits[1],
        "top5": hits[5],
        "top10": hits[10],
        "top1_rate": hits[1] / count,
        "top5_rate": hits[5] / count,
        "top10_rate": hits[10] / count,
        "invalid_top1": invalid_top1,
        "invalid_top1_rate": invalid_top1 / count,
        "formula_match_top1": formula_match_top1,
        "formula_match_given_valid_top1_rate": (
            formula_match_top1 / max(1, formula_valid_top1)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_predictions:
        prediction_path = args.output.with_name(f"{args.output.stem}_predictions.json")
        prediction_path.write_text(
            json.dumps(prediction_rows, ensure_ascii=False), encoding="utf-8"
        )
        result["predictions"] = str(prediction_path)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

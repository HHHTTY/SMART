#!/usr/bin/env python3
"""Evaluate one 500-row MS-format variant with the IBM replica pipeline.

This is deliberately an exploratory diagnostic.  Formula, HNMR, CNMR and IR
come from the same parsed rows; only the MS segment or its position changes.
The script writes ``metrics.json`` and ``predictions.json`` under ``--run-dir``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--tgt", type=Path, required=True)
    p.add_argument("--official-code", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--variant", choices=("e0", "e0_null", "e0_e1_e2_duplicate", "no_ms", "ms_first"), required=True)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--numeric-token-policy", choices=("exact", "nearest"), default="exact")
    return p.parse_args()


def _load_pipeline(root: Path):
    path = root / "run_iclr_open_nmt_router_pipeline.py"
    spec = importlib.util.spec_from_file_location("iclr_format_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _group_tokens(pipeline, row, modality, vocab, numeric_policy):
    """Canonicalize one group using exactly the pipeline's vocabulary policy."""
    group = row.formula.split() if modality == "FORMULA" else list(row.parts.get(modality, ()))
    out = []
    stats = {"tokens": 0, "unknown": 0, "numeric_mapped": 0}
    for token in group:
        if modality == "MSMS" and token == "MS":
            token = "E0Pos"
        mapped = False
        if modality != "FORMULA":
            token, mapped = pipeline._nearest_numeric_token(token, vocab, numeric_policy)
        stats["tokens"] += 1
        stats["numeric_mapped"] += int(mapped)
        stats["unknown"] += int(not pipeline._vocab_has(vocab, token))
        out.append(token)
    return out, stats


def _variant_groups(pipeline, row, variant, vocab, numeric_policy):
    formula, fstats = _group_tokens(pipeline, row, "FORMULA", vocab, numeric_policy)
    groups = [("FORMULA", formula, fstats)]
    hnmr, hs = _group_tokens(pipeline, row, "HNMR", vocab, numeric_policy)
    cnmr, cs = _group_tokens(pipeline, row, "CNMR", vocab, numeric_policy)
    ir, is_ = _group_tokens(pipeline, row, "IR", vocab, numeric_policy)
    ms, ms_stats = _group_tokens(pipeline, row, "MSMS", vocab, numeric_policy)
    if variant == "no_ms":
        groups.extend((("HNMR", hnmr, hs), ("CNMR", cnmr, cs), ("IR", ir, is_)))
    elif variant == "ms_first":
        groups.extend((("MSMS", ms, ms_stats), ("HNMR", hnmr, hs), ("CNMR", cnmr, cs), ("IR", ir, is_)))
    else:
        if variant == "e0_null" and ms:
            ms = [ms[0], "MSMS", "null", *ms[1:]]
        elif variant == "e0_e1_e2_duplicate" and ms:
            body = ms[1:]
            ms = [ms[0], *body, "E1Pos", *body, "E2Pos", *body]
        groups.extend((("HNMR", hnmr, hs), ("CNMR", cnmr, cs), ("IR", ir, is_), ("MSMS", ms, ms_stats)))
    tokens = [token for _name, group, _stats in groups for token in group]
    totals = {key: sum(stat[key] for _name, _tokens, stat in groups) for key in ("tokens", "unknown", "numeric_mapped")}
    # The synthetic variants add copies or marker tokens after the original
    # group statistics were collected.  Recount token/unknown totals so the
    # audit describes the actual tensor sent to OpenNMT.
    totals["tokens"] = len(tokens)
    totals["unknown"] = sum(not pipeline._vocab_has(vocab, token) for token in tokens)
    if variant == "e0_e1_e2_duplicate":
        totals["numeric_mapped"] = sum(stat["numeric_mapped"] for _name, _tokens, stat in groups if _name != "MSMS") + 3 * ms_stats["numeric_mapped"]
    return tokens, totals


def _batch(pipeline, rows, variant, vocab, device, numeric_policy):
    sequences = []
    totals = {"tokens": 0, "unknown": 0, "numeric_mapped": 0}
    for row in rows:
        tokens, stats = _variant_groups(pipeline, row, variant, vocab, numeric_policy)
        sequences.append([pipeline._tok_id(vocab, token) for token in tokens])
        for key in totals:
            totals[key] += stats[key]
    pad = pipeline._tok_id(vocab, "<blank>")
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long, device=device)
    source = torch.full((len(sequences), int(lengths.max().item()), 1), pad, dtype=torch.long, device=device)
    for index, seq in enumerate(sequences):
        source[index, :len(seq), 0] = torch.tensor(seq, dtype=torch.long, device=device)
    return source, lengths, totals


def main() -> None:
    args = _args()
    pipeline = _load_pipeline(args.pipeline_root)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = pipeline._parse_rows(args.src, args.tgt, args.limit, read_targets=True, ms_energy_mode="all")
    load_args = SimpleNamespace(official_code=args.official_code, checkpoint=args.checkpoint, device="cuda")
    model, vocabs, device = pipeline._load_onmt(load_args)
    model.eval()
    translator = model._iclr_translator
    translator.beam_size = 10
    translator.n_best = 10
    translator.max_length = 128
    translator.min_length = 5
    vocab = vocabs["src"]
    predictions = []
    total = {"tokens": 0, "unknown": 0, "numeric_mapped": 0}
    lengths = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            source, lens, stats = _batch(pipeline, batch_rows, args.variant, vocab, device, args.numeric_token_policy)
            lengths.extend(lens.cpu().tolist())
            for key in total:
                total[key] += stats[key]
            result = translator.translate_batch({"src": source, "srclen": lens}, False)
            for row_result in result["predictions"]:
                predictions.append([pipeline._decode(seq.detach().cpu().tolist(), vocabs["tgt"]) for seq in row_result[:10]])
    hits = {1: 0, 5: 0, 10: 0}
    records = []
    for row, candidates in zip(rows, predictions):
        target = pipeline._canonical_smiles(row.target)
        canonical = [pipeline._canonical_smiles(candidate) for candidate in candidates]
        for rank in hits:
            hits[rank] += int(target is not None and target in canonical[:rank])
        records.append({"row": row.index, "target": row.target, "predictions": candidates, "canonical_predictions": canonical})
    metrics = {
        "model": args.checkpoint.stem,
        "checkpoint": str(args.checkpoint),
        "variant": args.variant,
        "rows": len(rows),
        "input_contract": "Formula + HNMR + CNMR + IR with only the MS syntax/order changed",
        "beams": 10,
        "min_length": 5,
        "max_target_length": 128,
        "top1": hits[1], "top5": hits[5], "top10": hits[10],
        "top1_rate": hits[1] / max(len(rows), 1),
        "top5_rate": hits[5] / max(len(rows), 1),
        "top10_rate": hits[10] / max(len(rows), 1),
        "unknown_tokens": total["unknown"],
        "source_tokens": total["tokens"],
        "unknown_rate": total["unknown"] / max(total["tokens"], 1),
        "numeric_mapped": total["numeric_mapped"],
        "mean_source_len": sum(lengths) / max(len(lengths), 1),
        "max_source_len": max(lengths) if lengths else 0,
        "formal_baseline": False,
    }
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.run_dir / "predictions.json").write_text(json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

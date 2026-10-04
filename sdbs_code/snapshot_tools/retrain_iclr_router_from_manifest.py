#!/usr/bin/env python3
"""Retrain the ICLR native Router from a frozen action-reward manifest.

This is a diagnostic utility: it keeps the previously computed counterfactual
rewards fixed and changes only the Router target mode (for example soft-KL vs
hard oracle).  The source rows and manifest are copied into a fresh output
directory by the caller, so the original run remains immutable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import run_iclr_open_nmt_router_pipeline as pipeline


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--official-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--ms-energy-mode", choices=("all", "e1"), required=True)
    parser.add_argument("--router-target-mode", choices=("hard_oracle", "soft_reward"), required=True)
    parser.add_argument("--router-feature-mode", choices=("source_reference_free", "token_stats"), default="source_reference_free")
    parser.add_argument("--numeric-token-policy", choices=("exact", "nearest"), default="nearest")
    parser.add_argument("--seed", type=int, default=3247)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-epochs", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _args()
    original = json.loads(args.route_manifest.read_text(encoding="utf-8"))
    routes = pipeline._load_routes(args.route_manifest)
    source = Path(original.get("router_source", ""))
    if not source.exists():
        raise FileNotFoundError(f"router source from manifest does not exist: {source}")
    if int(original.get("rows", -1)) != args.limit:
        raise ValueError("manifest row count does not match --limit")
    rows = pipeline._parse_rows(
        source, None, args.limit, ms_energy_mode=args.ms_energy_mode
    )
    load_args = SimpleNamespace(
        official_code=args.official_code,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    model, vocabs, device = pipeline._load_onmt(load_args)
    run_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        src=source,
        router_src=source,
        tgt=Path("/dev/null"),
        run_dir=args.output_dir,
        official_code=args.official_code,
        limit=args.limit,
        batch_size=args.batch_size,
        lr=3e-4,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_epochs=args.early_stop_min_epochs,
        early_stop_min_delta=args.early_stop_min_delta,
        seed=args.seed,
        device=args.device,
        max_target_length=128,
        numeric_token_policy=args.numeric_token_policy,
        router_numeric_token_policy=args.numeric_token_policy,
        router_feature_mode=args.router_feature_mode,
        router_target_mode=args.router_target_mode,
        router_oracle_temperature=0.20,
        router_reward=str(original.get("router_reward", "target_beam_rank")),
        ms_energy_mode=args.ms_energy_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    route_copy = args.output_dir / "route_manifest.json"
    route_copy.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")
    output = pipeline._train_native_router(
        run_args, rows, route_copy, routes, vocabs, model, device
    )
    print(json.dumps({"router_checkpoint": str(output), "route_manifest": str(route_copy)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

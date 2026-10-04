#!/usr/bin/env python3
"""Apply a router trained on simulated source episodes to unlabeled SDBS rows.

The router checkpoint is trained with target-aware rewards on simulated rows.
This utility intentionally never reads a target file and never recomputes a
self-NLL reward on the real domain.  It only extracts the same checkpoint-aware
source-reference-free features and applies the frozen classifier.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

import torch


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--official-code", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ms-energy-mode", choices=("all", "e1"), default="all")
    parser.add_argument("--numeric-token-policy", choices=("exact", "nearest"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _args()
    import importlib.util

    spec = importlib.util.spec_from_file_location("iclr_pipeline", args.pipeline)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pipeline module: {args.pipeline}")
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    payload = torch.load(args.router_checkpoint, map_location="cpu", weights_only=False)
    policy = args.numeric_token_policy or payload.get("tokenization_policy", {}).get(
        "numeric_token_policy", "exact"
    )
    if payload.get("feature_mode") not in (None, "source_reference_free"):
        raise ValueError(
            "router checkpoint was trained with token_stats features; apply-only "
            "requires source_reference_free features"
        )

    load_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        official_code=args.official_code,
        device=args.device,
    )
    model, vocabs, device = pipeline._load_onmt(load_args)
    rows = pipeline._parse_rows(
        args.src, None, args.limit, ms_energy_mode=args.ms_energy_mode
    )
    support = pipeline.MODEL_SUPPORT[args.checkpoint.stem]
    features = pipeline._router_encoder_features(
        model, rows, vocabs, support, policy, device, batch_size=args.batch_size
    )

    feature_dim = int(payload["feature_dim"])
    if int(features.shape[1]) != feature_dim:
        raise ValueError(
            f"router feature dimension mismatch: checkpoint={feature_dim}, "
            f"input={features.shape[1]}"
        )
    classifier = torch.nn.Sequential(
        torch.nn.LayerNorm(feature_dim),
        torch.nn.Linear(feature_dim, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, len(pipeline.ACTION_SUBSETS)),
    )
    classifier.load_state_dict(payload["state_dict"])
    classifier.eval()
    with torch.no_grad():
        logits = classifier(features)

    routes = []
    for index, row in enumerate(rows):
        valid = torch.tensor(
            [set(subset).issubset(row.parts) for subset in pipeline.ACTION_SUBSETS],
            dtype=torch.bool,
        )
        masked = logits[index].masked_fill(~valid, float("-inf"))
        action = int(masked.argmax())
        routes.append(
            {
                "source_row_index": row.index,
                "router_selected_action_index": action,
                "router_selected_modalities": list(pipeline.ACTION_SUBSETS[action]),
                "valid_action_indices": valid.nonzero(as_tuple=False).flatten().tolist(),
            }
        )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "complete": True,
        "method": "apply_frozen_iclr_native_router",
        "checkpoint": str(args.checkpoint),
        "checkpoint_name": args.checkpoint.stem,
        "router_checkpoint": str(args.router_checkpoint),
        "router_source": "simulated_medium_target_aware_training_only",
        "source": str(args.src),
        "rows": len(rows),
        "target_labels_used": False,
        "router_reward": None,
        "feature_mode": payload.get("feature_mode", "source_reference_free"),
        "feature_contract_version": payload.get(
            "feature_contract_version", "source_reference_free_v2"
        ),
        "tokenization_policy": {
            "ms_energy_mode": args.ms_energy_mode,
            "numeric_token_policy": policy,
            "derived_shift": "m",
            "generic_ms_marker": "E1Pos",
            "explicit_ms_markers": ["E0Pos", "E1Pos", "E2Pos"],
        },
        "selection_counts": {
            "+".join(pipeline.ACTION_SUBSETS[action]): count
            for action, count in collections.Counter(
                row["router_selected_action_index"] for row in routes
            ).items()
        },
        "routes": routes,
    }
    (args.run_dir / "route_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["selection_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()

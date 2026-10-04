#!/usr/bin/env python
"""Stable-checkpoint pseudo labels followed by pretraining-style modality dropout.

The prepare pass generates Full-view greedy predictions and preserves their raw
token IDs.  The train pass keeps only canonical predictions shared by all input
checkpoints, additionally requiring a valid structure and an observed-formula
match.  Ground-truth structures are never loaded by either pass.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import run_batchwise_pseudoreward_tta as routed
import __main__
try:
    from tools import build_chemotion_compatible_preprocessor as _chemotion_compat
except ImportError:
    _chemotion_compat = None
if _chemotion_compat is not None:
    for _compat_name in (
        "ChemotionFormulaPreprocessor",
        "ChemotionCarbonPreprocessor",
        "ChemotionMultipletPreprocessor",
        "ChemotionMSMSTextPreprocessor",
    ):
        setattr(__main__, _compat_name, getattr(_chemotion_compat, _compat_name))
from analytical_fm.modeling.direct_pseudoreward import (
    direct_pseudo_sequence_nll,
    pad_pseudo_token_sequences,
)
from analytical_fm.modeling.molecular_pseudo_reward import (
    canonical_smiles,
    formula_signature,
    molecular_formula,
)


SPECTRAL_MODALITIES = ("MSMS", "HNMR", "CNMR", "IR")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3247)
    parser.add_argument("--ms-column", default=None)
    parser.add_argument("--ir-column", default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate")
    _common(generate)
    generate.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train")
    _common(train)
    train.add_argument("--prediction-jsonl", type=Path, action="append", required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument(
        "--full-view-policy",
        choices=("full_only", "include_full", "exclude_full"),
        required=True,
        help=(
            "Use only the Full view, or sample 0..3 / 1..3 dropped spectra "
            "per batch, respectively."
        ),
    )
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Existing completed epoch number; new epochs start at start_epoch + 1.",
    )
    train.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Restore optimizer state from --checkpoint when available.",
    )
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--grad-clip", type=float, default=0.8)
    train.add_argument("--save-epochs", default="1,2,3")
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--early-stop-patience", type=int, default=0,
                       help="Stop after this many epochs without pseudo-CE improvement; 0 disables.")
    train.add_argument("--early-stop-min-epochs", type=int, default=1)
    train.add_argument("--early-stop-min-delta", type=float, default=0.002)
    return parser.parse_args()


def _autocast(args: argparse.Namespace, device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and args.precision == "bf16",
    )


def _load_student_state(model: Any, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "student_state_dict" in payload:
        state = payload["student_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    model.load_state_dict(state, strict=True)


def _load_optimizer_state(optimizer: Any, checkpoint: Path) -> bool:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("optimizer_state_dict") if isinstance(payload, Mapping) else None
    if state is None:
        return False
    optimizer.load_state_dict(state)
    return True


def _trim_sequence(model: Any, sequence: torch.Tensor) -> list[int]:
    values = [int(value) for value in sequence.detach().cpu().tolist()]
    pad = int(model.target_tokenizer.pad_token_id)
    while values and values[-1] == pad:
        values.pop()
    return values


def generate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config, preprocessors, raw, loader = routed._build_input_only_loader(args)
    model = routed._build_model(args, data_config, preprocessors, device, trainable=False)
    _load_student_state(model, args.checkpoint)
    model.eval()
    model.excluded_input_modalities = frozenset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for raw_batch in loader:
            batch = routed._move(routed.adaptation_view(raw_batch), device)
            with _autocast(args, device):
                decoded, sequences = routed._ordinary_generate_with_sequences(
                    args, model, batch, beams=1
                )
            rows = [int(value) for value in raw_batch["source_row_indices"]]
            formulas = [str(value) for value in raw_batch["input_formulas"]]
            if len(rows) != len(decoded) or len(rows) != sequences.shape[0]:
                raise RuntimeError("generation output does not align with source rows")
            for index, source_row in enumerate(rows):
                text = str(decoded[index][0])
                canonical = canonical_smiles(text)
                predicted_formula = molecular_formula(canonical) if canonical else None
                formula_match = bool(
                    predicted_formula
                    and formula_signature(predicted_formula)
                    == formula_signature(formulas[index])
                )
                handle.write(
                    json.dumps(
                        {
                            "source_row_index": source_row,
                            "prediction": text,
                            "canonical_smiles": canonical,
                            "token_ids": _trim_sequence(model, sequences[index]),
                            "observed_formula": formulas[index],
                            "valid": canonical is not None,
                            "formula_match": formula_match,
                            "checkpoint": str(args.checkpoint),
                        }
                    )
                    + "\n"
                )
                written += 1
    manifest = {
        "complete": True,
        "checkpoint": str(args.checkpoint),
        "prediction_jsonl": str(args.output),
        "num_rows": written,
        "view": "Full",
        "beams": 1,
        "ground_truth_loaded": False,
        "elapsed_seconds": time.time() - started,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


def _read_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["source_row_index"])
            if index in rows:
                raise ValueError(f"duplicate source row {index} in {path}")
            rows[index] = row
    return rows


def _stable_rows(paths: Sequence[Path]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if len(paths) < 2:
        raise ValueError("stable labels require at least two checkpoint predictions")
    all_rows = [_read_rows(path) for path in paths]
    common = set(all_rows[0])
    for rows in all_rows[1:]:
        common.intersection_update(rows)
    stable: dict[int, dict[str, Any]] = {}
    counts = Counter()
    middle = len(all_rows) // 2
    for index in sorted(common):
        current = [rows[index] for rows in all_rows]
        canonical = [row.get("canonical_smiles") for row in current]
        if any(value is None for value in canonical):
            counts["invalid"] += 1
            continue
        if len(set(canonical)) != 1:
            counts["checkpoint_disagreement"] += 1
            continue
        counts["canonical_consensus"] += 1
        if not all(bool(row.get("formula_match")) for row in current):
            counts["formula_mismatch"] += 1
            continue
        chosen = dict(current[middle])
        chosen["checkpoint_predictions"] = [row["prediction"] for row in current]
        chosen["checkpoint_sources"] = [str(path) for path in paths]
        stable[index] = chosen
        counts["accepted"] += 1
    summary = {
        "prediction_jsonl": [str(path) for path in paths],
        "num_checkpoint_views": len(paths),
        "num_common_rows": len(common),
        **dict(counts),
        "selection": "strict canonical consensus AND valid AND formula match",
        "ground_truth_loaded": False,
    }
    return stable, summary


def _select_batch(batch: Mapping[str, Any], indices: Sequence[int], batch_size: int):
    return routed._select_batch(batch, indices, batch_size)


def _sample_exclusions(
    rng: np.random.RandomState,
    policy: str,
    input_modalities: frozenset[str],
) -> tuple[frozenset[str], tuple[str, ...]]:
    if policy == "full_only":
        return frozenset(), tuple(name for name in SPECTRAL_MODALITIES if name in input_modalities)
    available = tuple(name for name in SPECTRAL_MODALITIES if name in input_modalities)
    if not available:
        raise ValueError("Stage 2 dropout requires an available spectral modality")
    minimum = 0 if policy == "include_full" else 1
    count = int(rng.randint(minimum, len(available)))
    dropped = tuple(
        str(value)
        for value in rng.choice(available, count, replace=False)
    )
    return routed._random_dropout_exclusions(dropped, input_modalities)


def _save_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    args: argparse.Namespace,
    epoch: int,
    stable_summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> None:
    torch.save(
        {
            "student_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "stable_pseudo_summary": dict(stable_summary),
            "history": list(history),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    if args.epochs < 0 or args.start_epoch < 0 or args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("invalid optimization settings")
    if args.epochs == 0 and args.early_stop_patience < 1:
        raise ValueError("unbounded Stage 2 requires --early-stop-patience > 0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stable, stable_summary = _stable_rows(args.prediction_jsonl)
    if not stable:
        raise RuntimeError("stable pseudo-label filter accepted no rows")
    (args.run_dir / "stable_pseudo_manifest.json").write_text(
        json.dumps(stable_summary, indent=2), encoding="utf-8"
    )

    data_config, preprocessors, raw, loader = routed._build_input_only_loader(args)
    if not set(stable).issubset(set(range(len(raw)))):
        raise ValueError("stable pseudo rows are outside the current dataset")
    input_modalities = frozenset(
        name for name, metadata in data_config.items() if not metadata.get("target", False)
    )
    student = routed._build_model(args, data_config, preprocessors, device, trainable=True)
    _load_student_state(student, args.checkpoint)
    student.train()
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay, eps=1e-8
    )
    optimizer_resumed = False
    if args.resume_optimizer:
        optimizer_resumed = _load_optimizer_state(optimizer, args.checkpoint)
    save_epochs = {int(value) for value in args.save_epochs.split(",") if value.strip()}
    history: list[dict[str, Any]] = []
    dropout_counts: Counter[str] = Counter()
    started = time.time()
    global_step = 0
    best_epoch_loss = float("inf")
    stale_epochs = 0
    completed_epoch = args.start_epoch

    epoch_iterator = (
        itertools.count(args.start_epoch + 1)
        if args.epochs == 0
        else range(args.start_epoch + 1, args.start_epoch + args.epochs + 1)
    )
    for epoch in epoch_iterator:
        rng = np.random.RandomState(args.seed + 1009 * epoch)
        epoch_losses: list[float] = []
        epoch_accepted = 0
        for batch_index, raw_batch in enumerate(loader, start=1):
            source_rows = [int(value) for value in raw_batch["source_row_indices"]]
            selected_indices = [
                local for local, source_row in enumerate(source_rows) if source_row in stable
            ]
            if not selected_indices:
                continue
            selected_raw = _select_batch(raw_batch, selected_indices, len(source_rows))
            batch = routed._move(routed.adaptation_view(selected_raw), device)
            selected_rows = [source_rows[index] for index in selected_indices]
            pseudo = [
                torch.tensor(stable[source_row]["token_ids"], dtype=torch.long)
                for source_row in selected_rows
            ]
            sequences = pad_pseudo_token_sequences(student, pseudo)
            excluded, retained = _sample_exclusions(rng, args.full_view_policy, input_modalities)
            student.excluded_input_modalities = excluded
            dropout_counts["+".join(retained)] += 1
            optimizer.zero_grad(set_to_none=True)
            with _autocast(args, device):
                loss, per_sequence = direct_pseudo_sequence_nll(student, batch, sequences)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"non-finite pseudo CE at epoch {epoch} batch {batch_index}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            global_step += 1
            epoch_accepted += len(selected_rows)
            value = float(loss.detach().cpu())
            epoch_losses.append(value)
            record = {
                "epoch": epoch,
                "batch": batch_index,
                "global_step": global_step,
                "pseudo_ce": value,
                "mean_sequence_nll": float(per_sequence.mean().detach().cpu()),
                "accepted_pseudo": len(selected_rows),
                "dropped_modalities": sorted(excluded),
                "retained_spectral_modalities": list(retained),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            }
            history.append(record)
            if global_step == 1 or global_step % args.log_every == 0:
                print(json.dumps(record), flush=True)
        summary = {
            "epoch_summary": {
                "epoch": epoch,
                "mean_pseudo_ce": float(np.mean(epoch_losses)),
                "steps": len(epoch_losses),
                "accepted_examples": epoch_accepted,
                "dropout_counts_cumulative": dict(dropout_counts),
                "elapsed_seconds": time.time() - started,
            }
        }
        print(json.dumps(summary), flush=True)
        completed_epoch = epoch
        epoch_loss = float(summary["epoch_summary"]["mean_pseudo_ce"])
        if epoch_loss < best_epoch_loss - args.early_stop_min_delta:
            best_epoch_loss = epoch_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch in save_epochs or (
            args.epochs > 0 and epoch == args.start_epoch + args.epochs
        ) or (
            args.early_stop_patience > 0 and epoch >= args.early_stop_min_epochs
            and stale_epochs >= args.early_stop_patience
        ):
            _save_checkpoint(
                args.run_dir / f"stage2_epoch_{epoch}.pt",
                student,
                optimizer,
                args,
                epoch,
                stable_summary,
                history,
            )
        if (args.early_stop_patience > 0 and epoch >= args.early_stop_min_epochs
                and stale_epochs >= args.early_stop_patience):
            print(json.dumps({"early_stop": True, "epoch": epoch,
                              "best_mean_pseudo_ce": best_epoch_loss,
                              "stale_epochs": stale_epochs}), flush=True)
            break
    student.excluded_input_modalities = frozenset()
    complete = {
        "complete": True,
        "run_dir": str(args.run_dir),
        "init_checkpoint": str(args.checkpoint),
        "full_view_policy": args.full_view_policy,
        "stable_pseudo_summary": stable_summary,
        "dropout_counts": dict(dropout_counts),
        "epochs": completed_epoch - args.start_epoch,
        "completed_epoch": completed_epoch,
        "start_epoch": args.start_epoch,
        "optimizer_resumed": optimizer_resumed,
        "ground_truth_loaded": False,
        "elapsed_seconds": time.time() - started,
    }
    (args.run_dir / "adaptation.complete.json").write_text(
        json.dumps(complete, indent=2), encoding="utf-8"
    )
    print(json.dumps(complete), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train a Full-view student on routed pseudo labels with unsupervised early stop.

Only observable inputs and frozen-teacher pseudo labels are read. Ground-truth
structure columns are never loaded. Periodic predictions use the Stage 2
stable-checkpoint JSONL schema.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import run_batchwise_pseudoreward_tta as routed
from analytical_fm.modeling.molecular_pseudo_reward import (
    canonical_smiles,
    formula_signature,
    molecular_formula,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True,
                        help="Completed routed TTT run containing first_pass_batches.jsonl.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sample-manifest-json", type=Path)
    parser.add_argument("--stream-order-json", type=Path)
    parser.add_argument("--seed", type=int, default=3247)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=20,
                        help="Safety ceiling; normal stopping is controlled by validation patience.")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=2)
    parser.add_argument("--log-every-steps", type=int, default=25)
    return parser.parse_args()


def _load_pseudo_targets(path: Path) -> dict[int, dict[str, Any]]:
    records_path = path / "first_pass_batches.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    targets: dict[int, dict[str, Any]] = {}
    expected_batch = 0
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["batch_index"]) != expected_batch:
            raise ValueError("source first-pass batches are missing or out of order")
        expected_batch += 1
        arrays = (
            record["source_row_indices"],
            record["pseudo_token_ids"],
            record["pseudo_smiles"],
            record["pseudo_valid"],
            record["pseudo_formula_match"],
            record["selected_subsets"],
        )
        if len({len(values) for values in arrays}) != 1:
            raise ValueError(f"misaligned pseudo-target fields in batch {record['batch_index']}")
        for row, tokens, smiles, valid, formula_match, route in zip(*arrays):
            row = int(row)
            if row in targets:
                raise ValueError(f"duplicate pseudo target for source row {row}")
            token_ids = [int(value) for value in tokens]
            if len(token_ids) < 2:
                raise ValueError(f"pseudo token sequence is empty for source row {row}")
            targets[row] = {
                "token_ids": token_ids,
                "prediction": str(smiles),
                "teacher_valid": bool(valid),
                "teacher_formula_match": bool(formula_match),
                "route": [str(value) for value in route],
            }
    if not targets:
        raise ValueError("source first-pass run has no pseudo targets")
    return targets


def _split_rows(
    eligible_rows: list[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0.05 <= validation_fraction < 0.5:
        raise ValueError("validation fraction must be in [0.05, 0.5)")
    if len(eligible_rows) < 10:
        raise ValueError("at least 10 valid, formula-matched teacher labels are required")
    rows = np.asarray(sorted(eligible_rows), dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(rows)
    n_validation = max(1, int(round(len(rows) * validation_fraction)))
    validation = sorted(int(value) for value in rows[:n_validation])
    train = sorted(int(value) for value in rows[n_validation:])
    return train, validation


def _tokens_for_rows(student: Any, targets: Mapping[int, Mapping[str, Any]],
                     rows: list[int], device: torch.device) -> torch.Tensor:
    return routed._pad_pseudo_token_sequences(
        student,
        [torch.tensor(targets[row]["token_ids"], dtype=torch.long, device=device) for row in rows],
    )


def _trim_tokens(model: Any, sequence: torch.Tensor) -> list[int]:
    values = [int(value) for value in sequence.detach().cpu().tolist()]
    pad = int(model.target_tokenizer.pad_token_id)
    while values and values[-1] == pad:
        values.pop()
    return values


def _make_loader(raw: Any, source_loader: Any, rows: list[int], args: argparse.Namespace,
                 *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(raw, rows),
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=source_loader.collate_fn,
        persistent_workers=args.num_workers > 0,
    )


def _save_checkpoint(path: Path, model: Any, epoch: int,
                     val_nll: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "routed pseudo-label Full-view TTT with pseudo-NLL early stopping",
            "student_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "epoch": int(epoch),
            "validation_pseudo_nll": float(val_nll),
            "source_checkpoint": str(args.checkpoint),
            "source_run": str(args.source_run),
            "teacher_labels_used_for_training": "valid and observed-formula-matched frozen-router Top-1",
            "ground_truth_loaded": False,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )


@torch.no_grad()
def _evaluate_pseudo_nll(args: argparse.Namespace, model: Any, loader: DataLoader,
                         targets: Mapping[int, Mapping[str, Any]], device: torch.device) -> float:
    model.eval()
    model.excluded_input_modalities = frozenset()
    losses: list[float] = []
    for raw_batch in loader:
        batch = routed._move(routed.adaptation_view(raw_batch), device)
        rows = [int(value) for value in batch["source_row_indices"]]
        pseudo = _tokens_for_rows(model, targets, rows, device)
        weights = torch.ones(len(rows), dtype=torch.float32, device=device)
        with routed._autocast(args, device):
            _loss, nll = routed._direct_pseudo_sequence_loss(model, batch, pseudo, weights)
        losses.extend(float(value) for value in nll.detach().cpu().tolist())
    if not losses:
        raise RuntimeError("pseudo-label validation set is empty")
    return float(np.mean(losses))


@torch.no_grad()
def _generate_stage2_labels(args: argparse.Namespace, model: Any, loader: Any,
                            output: Path, checkpoint: Path, device: torch.device,
                            epoch: int) -> int:
    model.eval()
    model.excluded_input_modalities = frozenset()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for raw_batch in loader:
            batch = routed._move(routed.adaptation_view(raw_batch), device)
            with routed._autocast(args, device):
                decoded, sequences = routed._ordinary_generate_with_sequences(
                    args, model, batch, beams=1
                )
            rows = [int(value) for value in raw_batch["source_row_indices"]]
            formulas = [str(value) for value in raw_batch["input_formulas"]]
            for index, source_row in enumerate(rows):
                prediction = str(decoded[index][0])
                canonical = canonical_smiles(prediction)
                predicted_formula = molecular_formula(canonical) if canonical else None
                formula_match = bool(
                    predicted_formula
                    and formula_signature(predicted_formula) == formula_signature(formulas[index])
                )
                handle.write(json.dumps({
                    "source_row_index": source_row,
                    "prediction": prediction,
                    "canonical_smiles": canonical,
                    "token_ids": _trim_tokens(model, sequences[index]),
                    "observed_formula": formulas[index],
                    "valid": canonical is not None,
                    "formula_match": formula_match,
                    "checkpoint": str(checkpoint),
                    "epoch": int(epoch),
                    "ground_truth_loaded": False,
                }) + "\n")
                written += 1
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "complete": True,
        "checkpoint": str(checkpoint),
        "prediction_jsonl": str(output),
        "num_rows": written,
        "view": "Full",
        "beams": 1,
        "ground_truth_loaded": False,
        "epoch": int(epoch),
    }, indent=2), encoding="utf-8")
    return written


def _run(args: argparse.Namespace) -> None:
    if args.lr <= 0 or args.grad_clip <= 0 or args.min_epochs < 1 or args.max_epochs < args.min_epochs:
        raise ValueError("invalid optimizer or early-stopping settings")
    if args.patience < 1 or args.checkpoint_interval_epochs < 1 or args.min_delta < 0:
        raise ValueError("patience/interval must be positive and min-delta non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TTT")
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if any(args.run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {args.run_dir}")

    targets = _load_pseudo_targets(args.source_run)
    eligible = sorted(
        row for row, value in targets.items()
        if value["teacher_valid"] and value["teacher_formula_match"]
    )
    train_rows, validation_rows = _split_rows(eligible, args.validation_fraction, args.seed)
    data_config, preprocessors, raw, loader = routed._build_input_only_loader(args)
    if set(targets) != set(range(len(raw))):
        raise ValueError("pseudo-target rows must exactly cover the ordered input-only data")
    train_loader = _make_loader(raw, loader, train_rows, args, shuffle=True, seed=args.seed + 17)
    validation_loader = _make_loader(raw, loader, validation_rows, args, shuffle=False, seed=args.seed)
    model = routed._build_model(args, data_config, preprocessors, device, trainable=True)
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

    args.run_dir.joinpath("pseudo_label_protocol.json").write_text(json.dumps({
        "source_run": str(args.source_run),
        "input_data": str(args.data_path),
        "rows": len(raw),
        "eligible_pseudo_rows": len(eligible),
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training_label_filter": "frozen routed-teacher Top-1 is valid and matches observed Formula",
        "validation_signal": "mean length-normalized NLL on held-out frozen-teacher pseudo labels",
        "validation_signal_is_ground_truth": False,
        "stage2_generation": "Full-view greedy predictions at baseline, checkpoint intervals, and final epoch",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs_safety_ceiling": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "checkpoint_interval_epochs": args.checkpoint_interval_epochs,
        "early_stopping_selects_checkpoint": "lowest held-out pseudo-label NLL; no structure labels read",
        "seed": args.seed,
    }, indent=2), encoding="utf-8")

    full_loader = loader
    checkpoints_dir = args.run_dir / "checkpoints"
    pseudo_dir = args.run_dir / "stage2_pseudo_labels"
    checkpoints_dir.mkdir()
    pseudo_dir.mkdir()
    baseline_path = checkpoints_dir / "student_epoch_000.pt"
    initial_val_nll = _evaluate_pseudo_nll(args, model, validation_loader, targets, device)
    _save_checkpoint(baseline_path, model, 0, initial_val_nll, args)
    initial_rows = _generate_stage2_labels(
        args, model, full_loader, pseudo_dir / "epoch_000.jsonl", baseline_path, device, 0
    )
    if initial_rows != len(raw):
        raise RuntimeError("baseline Stage 2 predictions do not cover the input data")
    best_nll = initial_val_nll
    best_epoch = 0
    stale_epochs = 0
    _save_checkpoint(args.run_dir / "best_validation_student.pt", model, 0, initial_val_nll, args)
    global_step = 0
    history: list[dict[str, Any]] = [{"epoch": 0, "validation_pseudo_nll": initial_val_nll, "optimizer_steps": 0}]
    started = time.time()
    early_stop_reason = "max_epochs_safety_ceiling"
    prediction_epochs = [0]
    prediction_files = [str(pseudo_dir / "epoch_000.jsonl")]

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for raw_batch in train_loader:
            batch = routed._move(routed.adaptation_view(raw_batch), device)
            rows = [int(value) for value in batch["source_row_indices"]]
            pseudo = _tokens_for_rows(model, targets, rows, device)
            weights = torch.ones(len(rows), dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with routed._autocast(args, device):
                loss, _nll = routed._direct_pseudo_sequence_loss(model, batch, pseudo, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite train pseudo NLL at epoch {epoch}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            optimizer.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            if global_step == 1 or global_step % args.log_every_steps == 0:
                print(json.dumps({
                    "epoch": epoch,
                    "step": global_step,
                    "train_pseudo_nll": epoch_losses[-1],
                    "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                    "elapsed_seconds": time.time() - started,
                }), flush=True)

        val_nll = _evaluate_pseudo_nll(args, model, validation_loader, targets, device)
        improved = val_nll < best_nll - args.min_delta
        if improved:
            best_nll = val_nll
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(args.run_dir / "best_validation_student.pt", model, epoch, val_nll, args)
        else:
            stale_epochs += 1
        epoch_record = {
            "epoch": epoch,
            "train_pseudo_nll": float(np.mean(epoch_losses)),
            "validation_pseudo_nll": val_nll,
            "best_validation_pseudo_nll": best_nll,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "optimizer_steps": global_step,
            "improved": improved,
            "elapsed_seconds": time.time() - started,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)

        should_snapshot = (
            epoch % args.checkpoint_interval_epochs == 0
            or (epoch >= args.min_epochs and stale_epochs >= args.patience)
            or epoch == args.max_epochs
        )
        if should_snapshot:
            checkpoint_path = checkpoints_dir / f"student_epoch_{epoch:03d}.pt"
            _save_checkpoint(checkpoint_path, model, epoch, val_nll, args)
            output_path = pseudo_dir / f"epoch_{epoch:03d}.jsonl"
            count = _generate_stage2_labels(args, model, full_loader, output_path, checkpoint_path, device, epoch)
            if count != len(raw):
                raise RuntimeError(f"Stage 2 predictions at epoch {epoch} do not cover the input data")
            prediction_epochs.append(epoch)
            prediction_files.append(str(output_path))

        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            early_stop_reason = "held_out_pseudo_nll_patience"
            break

    best_path = args.run_dir / "best_validation_student.pt"
    best_prediction_file = None
    if best_epoch not in prediction_epochs:
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_payload["student_state_dict"], strict=True)
        best_rows = _generate_stage2_labels(
            args,
            model,
            full_loader,
            pseudo_dir / f"best_validation_epoch_{best_epoch:03d}.jsonl",
            best_path,
            device,
            best_epoch,
        )
        if best_rows != len(raw):
            raise RuntimeError("best-validation Stage 2 predictions do not cover the input data")
        best_prediction_file = str(pseudo_dir / f"best_validation_epoch_{best_epoch:03d}.jsonl")
        prediction_files.append(best_prediction_file)
    final_manifest = {
        "complete": True,
        "early_stop_reason": early_stop_reason,
        "epochs_completed": epoch,
        "optimizer_steps": global_step,
        "best_epoch": best_epoch,
        "best_validation_pseudo_nll": best_nll,
        "initial_validation_pseudo_nll": initial_val_nll,
        "validation_rows": len(validation_rows),
        "training_rows": len(train_rows),
        "eligible_teacher_pseudo_rows": len(eligible),
        "checkpoint_prediction_epochs": prediction_epochs,
        "checkpoint_prediction_files": prediction_files,
        "best_validation_prediction_file": best_prediction_file,
        "training_ground_truth_loaded": False,
        "evaluation_metrics_used_for_early_stopping": False,
        "best_validation_checkpoint": str(best_path),
        "source_run": str(args.source_run),
        "seconds": time.time() - started,
    }
    (args.run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.run_dir / "complete.json").write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
    print(json.dumps(final_manifest, indent=2), flush=True)


if __name__ == "__main__":
    _run(_args())

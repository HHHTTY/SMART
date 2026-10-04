#!/usr/bin/env python
"""Stage 2 router-OPD with pretraining-style modality dropout.

This is an alternative to stable-pseudo teacher-forcing CE, not an additional
Stage 1 alignment pass.  A frozen Stage 1 model supplies the per-sample Router
Top-2 fused next-token distribution on the student's own greedy prefixes.  A
student initialized from the same checkpoint is updated under a randomly
dropped input view.  Stable-checkpoint consensus is used only as a label-free
sample gate; target structures are never loaded during adaptation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import run_batchwise_pseudoreward_tta as routed
import run_dual_branch_router_fusion as fusion
import run_on_policy_dual_branch_distillation as opd
import run_pmgfa_direct_casp as pmgfa
import run_stable_checkpoint_dropout_stage2 as stable_stage2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--prediction-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--full-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--full-view-policy",
        choices=("include_full", "exclude_full"),
        default="include_full",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3247)
    parser.add_argument("--save-epochs", default="1,2,3")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _autocast(args: argparse.Namespace, device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and args.precision == "bf16",
    )


def _decode_logits(
    model: Any,
    batch: Mapping[str, Any],
    sequences: torch.Tensor,
    keep: frozenset[str],
) -> torch.Tensor:
    """Decode without the detach used by the inference-only fusion helper."""
    state = fusion._prepare_state(model, batch, keep)
    decoder_input = sequences[:, :-1].contiguous()
    pad = int(model.target_tokenizer.pad_token_id)
    output = model.hf_model(
        encoder_outputs=state["encoder_outputs"],
        attention_mask=state["attention_mask"],
        decoder_input_ids=decoder_input,
        decoder_attention_mask=decoder_input.ne(pad).long(),
        labels=None,
        use_cache=False,
    )
    return output.logits


def _mix_log_probabilities(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if first_logits.shape != second_logits.shape:
        raise ValueError("Top-2 teacher branches have different shapes")
    if weights.shape != (first_logits.shape[0], 2):
        raise ValueError("Top-2 teacher weights do not match the batch")
    if bool((weights < 0).any()) or not torch.allclose(
        weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-5
    ):
        raise ValueError("Top-2 teacher weights must be non-negative and sum to one")
    first = F.log_softmax(first_logits.float() / temperature, dim=-1)
    second = F.log_softmax(second_logits.float() / temperature, dim=-1)
    log_weights = weights.float().clamp_min(1e-8).log()
    return torch.logsumexp(
        torch.stack(
            (
                first + log_weights[:, 0, None, None],
                second + log_weights[:, 1, None, None],
            ),
            dim=0,
        ),
        dim=0,
    )


def _router_top2_teacher_logp(
    teacher: Any,
    batch: Mapping[str, Any],
    sequences: torch.Tensor,
    routes: Sequence[Mapping[str, Any]],
    *,
    temperature: float,
) -> torch.Tensor:
    batch_size = len(routes)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, route in enumerate(routes):
        groups[(int(route["selected_action"]), int(route["top2_action"]))].append(index)
    result: torch.Tensor | None = None
    for (first_action, second_action), indices in sorted(groups.items()):
        current = routed._select_batch(batch, indices, batch_size)
        index = torch.tensor(indices, dtype=torch.long, device=sequences.device)
        current_sequences = sequences.index_select(0, index)
        first_keep = routed.retained_input_modalities(routed.ACTION_SUBSETS[first_action])
        second_keep = routed.retained_input_modalities(routed.ACTION_SUBSETS[second_action])
        first_logits = _decode_logits(teacher, current, current_sequences, first_keep)
        second_logits = _decode_logits(teacher, current, current_sequences, second_keep)
        weights = torch.tensor(
            [routes[value]["branch_weights"] for value in indices],
            dtype=torch.float32,
            device=sequences.device,
        )
        current_logp = _mix_log_probabilities(
            first_logits, second_logits, weights, temperature=temperature
        )
        if result is None:
            result = current_logp.new_empty((batch_size, *current_logp.shape[1:]))
        result.index_copy_(0, index, current_logp)
    teacher.excluded_input_modalities = frozenset()
    if result is None:
        raise RuntimeError("Router Top-2 teacher did not cover the batch")
    return result


def _per_sample_kl(
    teacher_logp: torch.Tensor,
    student_logits: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if teacher_logp.shape != student_logits.shape:
        raise ValueError("Teacher and student token distributions do not align")
    if token_mask.shape != student_logits.shape[:2]:
        raise ValueError("Token mask does not align with decoder positions")
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    token_kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    token_kl = token_kl * (temperature**2)
    mask = token_mask.to(token_kl.dtype)
    return (token_kl * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _save_checkpoint(
    path: Path,
    student: Any,
    optimizer: Any,
    args: argparse.Namespace,
    epoch: int,
    stable_summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> None:
    torch.save(
        {
            "method": "Stage 2 Router Top-2 on-policy distribution distillation",
            "stage": "stage2",
            "epoch": epoch,
            "student_state_dict": {
                name: value.detach().cpu() for name, value in student.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "args": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items()
            },
            "stable_pseudo_summary": dict(stable_summary),
            "history": list(history),
            "target_labels_loaded_during_adaptation": False,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("Invalid optimization settings")
    if args.temperature <= 0 or args.full_anchor_weight < 0:
        raise ValueError("Temperature must be positive and anchor weight non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    stable, stable_summary = stable_stage2._stable_rows(args.prediction_jsonl)
    if not stable:
        raise RuntimeError("Stable checkpoint gate accepted no rows")
    route_manifest, route_rows = pmgfa._load_route_records(args.route_manifest)
    if not set(stable).issubset(route_rows):
        raise ValueError("Router manifest does not cover all accepted stable rows")
    (args.run_dir / "stable_sample_gate.json").write_text(
        json.dumps(stable_summary, indent=2), encoding="utf-8"
    )

    data_config, preprocessors, raw, loader = routed._build_input_only_loader(args)
    stable = {index: row for index, row in stable.items() if index < len(raw)}
    if not stable:
        raise RuntimeError("Stable checkpoint gate accepted no rows in the loaded data range")
    stable_summary = {
        **stable_summary,
        "accepted_in_loaded_data_range": len(stable),
        "loaded_data_rows": len(raw),
    }
    (args.run_dir / "stable_sample_gate.json").write_text(
        json.dumps(stable_summary, indent=2), encoding="utf-8"
    )
    input_modalities = frozenset(
        name for name, metadata in data_config.items() if not metadata.get("target", False)
    )
    teacher = routed._build_model(args, data_config, preprocessors, device, trainable=False)
    student = routed._build_model(args, data_config, preprocessors, device, trainable=True)
    stable_stage2._load_student_state(teacher, args.checkpoint)
    stable_stage2._load_student_state(student, args.checkpoint)
    teacher.eval().requires_grad_(False)
    student.train()
    parameters = [value for value in student.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay, eps=1e-8
    )
    save_epochs = {int(value) for value in args.save_epochs.split(",") if value.strip()}
    history: list[dict[str, Any]] = []
    dropout_counts: Counter[str] = Counter()
    route_pair_counts: Counter[str] = Counter()
    started = time.time()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        rng = np.random.RandomState(args.seed + 1009 * epoch)
        epoch_records: list[dict[str, Any]] = []
        for batch_index, raw_batch in enumerate(loader, start=1):
            source_rows = [int(value) for value in raw_batch["source_row_indices"]]
            selected_indices = [
                local for local, source_row in enumerate(source_rows) if source_row in stable
            ]
            if not selected_indices:
                continue
            selected_raw = stable_stage2._select_batch(
                raw_batch, selected_indices, len(source_rows)
            )
            selected_rows = [source_rows[index] for index in selected_indices]
            routes = [route_rows[source_row] for source_row in selected_rows]
            batch = routed._move(routed.adaptation_view(selected_raw), device)
            excluded, retained = stable_stage2._sample_exclusions(
                rng, args.full_view_policy, input_modalities
            )
            dropout_counts["+".join(retained)] += 1
            route_pair_counts.update(
                f'{route["selected_action"]}+{route["top2_action"]}' for route in routes
            )

            student.excluded_input_modalities = excluded
            student.eval()
            with torch.no_grad(), _autocast(args, device):
                _decoded, rollout = routed._ordinary_generate_with_sequences(
                    args, student, batch, beams=1
                )
            student.train()
            mask = opd._token_mask(
                rollout,
                pad_token_id=int(student.target_tokenizer.pad_token_id),
                eos_token_id=int(student.target_tokenizer.eos_token_id),
            )
            with torch.no_grad(), _autocast(args, device):
                router_teacher_logp = _router_top2_teacher_logp(
                    teacher, batch, rollout, routes, temperature=args.temperature
                )
                full_teacher_logp = None
                if args.full_anchor_weight > 0:
                    full_teacher_logits = _decode_logits(
                        teacher, batch, rollout, fusion.ALL_MODALITIES
                    )
                    full_teacher_logp = F.log_softmax(
                        full_teacher_logits.float() / args.temperature, dim=-1
                    )

            optimizer.zero_grad(set_to_none=True)
            with _autocast(args, device):
                student_logits = _decode_logits(
                    student, batch, rollout, frozenset(input_modalities.difference(excluded))
                )
                router_kl = _per_sample_kl(
                    router_teacher_logp,
                    student_logits,
                    mask,
                    temperature=args.temperature,
                ).mean()
                full_anchor_kl = student_logits.sum() * 0.0
                if full_teacher_logp is not None:
                    full_anchor_kl = _per_sample_kl(
                        full_teacher_logp,
                        student_logits,
                        mask,
                        temperature=args.temperature,
                    ).mean()
                loss = router_kl + args.full_anchor_weight * full_anchor_kl
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Non-finite Stage 2 OPD loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
                raise FloatingPointError("Non-finite Stage 2 OPD gradient")
            optimizer.step()
            global_step += 1
            record = {
                "epoch": epoch,
                "batch": batch_index,
                "global_step": global_step,
                "accepted_samples": len(selected_rows),
                "dropped_modalities": sorted(excluded),
                "retained_spectral_modalities": list(retained),
                "router_kl": float(router_kl.detach().cpu()),
                "full_anchor_kl": float(full_anchor_kl.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            }
            history.append(record)
            epoch_records.append(record)
            if global_step == 1 or global_step % args.log_every == 0:
                print(json.dumps(record), flush=True)
        summary = {
            "epoch_summary": {
                "epoch": epoch,
                "steps": len(epoch_records),
                "accepted_examples": int(
                    sum(row["accepted_samples"] for row in epoch_records)
                ),
                "mean_router_kl": float(
                    np.mean([row["router_kl"] for row in epoch_records])
                ),
                "mean_full_anchor_kl": float(
                    np.mean([row["full_anchor_kl"] for row in epoch_records])
                ),
                "mean_loss": float(np.mean([row["loss"] for row in epoch_records])),
                "dropout_counts_cumulative": dict(dropout_counts),
                "elapsed_seconds": time.time() - started,
            }
        }
        print(json.dumps(summary), flush=True)
        if epoch in save_epochs:
            _save_checkpoint(
                args.run_dir / f"stage2_opd_epoch_{epoch}.pt",
                student,
                optimizer,
                args,
                epoch,
                stable_summary,
                history,
            )

    student.excluded_input_modalities = frozenset()
    complete = {
        "complete": True,
        "method": "Stage 2 Router Top-2 OPD replacing modality-dropout pseudo CE",
        "stage": "stage2",
        "init_checkpoint": str(args.checkpoint),
        "route_manifest": str(args.route_manifest),
        "route_teacher": route_manifest.get("teacher_view"),
        "objective": "on-policy KL(router Top-2 teacher || dropout student)",
        "full_anchor_weight": args.full_anchor_weight,
        "full_view_policy": args.full_view_policy,
        "stable_sample_count": len(stable),
        "dropout_counts": dict(dropout_counts),
        "route_pair_counts": dict(route_pair_counts),
        "epochs": args.epochs,
        "target_labels_loaded_during_adaptation": False,
        "elapsed_seconds": time.time() - started,
    }
    (args.run_dir / "adaptation.complete.json").write_text(
        json.dumps(complete, indent=2), encoding="utf-8"
    )
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    train(parse_args())

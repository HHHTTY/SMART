#!/usr/bin/env python
"""Train a new seven-action H/C/IR router for the official 34a model.

The official checkpoint exposes Formula, Multiplets (1H-NMR), Carbon (13C-NMR)
and IR.  This runner is intentionally independent from the old five-modality
MS router.  The router is trained on a 2k source subset, then frozen for an
unlabelled education route-only pass and a router-teacher/full-student TTT.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from torch import nn
from torch.utils.data import DataLoader

from analytical_fm.data.datamodules import MultiModalDataCollator
from analytical_fm.data.datasets import build_dataset_multimodal
from analytical_fm.modeling.wrapper import HFWrapper
from analytical_fm.utils import clean_sample


SPECTRAL = ("Multiplets", "Carbon", "IR")
INPUTS = ("Formula", *SPECTRAL)
ACTIONS = tuple(
    tuple(name for bit, name in enumerate(SPECTRAL) if mask & (1 << bit))
    for mask in range(1, 1 << len(SPECTRAL))
)


def canonical(value: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(clean_sample(str(value), canonicalise=False))
    except Exception:
        mol = None
    return Chem.MolToSmiles(mol) if mol is not None else None


def molecular_formula(value: str) -> str | None:
    can = canonical(value)
    if can is None:
        return None
    return rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(can))


def tanimoto(a: str, b: str) -> float:
    ma, mb = Chem.MolFromSmiles(str(a)), Chem.MolFromSmiles(str(b))
    if ma is None or mb is None:
        return 0.0
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    return value


def modality_length(value: Any) -> int:
    if isinstance(value, dict):
        value = value["tokenized_input"]
    return int(value.shape[0])


def select_batch(batch: dict[str, Any], indices: list[int], n: int) -> dict[str, Any]:
    idx = torch.tensor(indices, dtype=torch.long)

    def select(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim > 1 and value.shape[1] == n:
                return value.index_select(1, idx.to(value.device))
            if value.ndim and value.shape[0] == n:
                return value.index_select(0, idx.to(value.device))
            return value
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        if isinstance(value, list) and len(value) == n:
            return [value[i] for i in indices]
        return value

    return select(batch)


def view_batch(batch: dict[str, Any], keep: tuple[str, ...]) -> dict[str, Any]:
    """Keep Formula and selected spectra in the official concatenation order."""
    keep_set = {"Formula", *keep}
    original = batch["encoder_input"]
    spans: list[int] = []
    offset = 0
    for name in INPUTS:
        length = modality_length(original[name])
        if name in keep_set:
            spans.extend(range(offset, offset + length))
        offset += length
    row = dict(batch)
    row["encoder_input"] = {name: original[name] for name in INPUTS if name in keep_set}
    row["encoder_pad_mask"] = batch["encoder_pad_mask"].index_select(
        0, torch.tensor(spans, device=batch["encoder_pad_mask"].device)
    )
    return row


def ensure_dataset_dir(path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Make a loader directory; a single parquet is treated as all three splits."""
    if path.is_dir():
        return path, None
    if path.suffix != ".parquet":
        raise ValueError(f"Expected parquet file or directory: {path}")
    temp = tempfile.TemporaryDirectory(prefix="official34a_dataset_")
    root = Path(temp.name)
    for split in ("train", "validation", "test"):
        shutil.copy2(path, root / f"{split}.parquet")
    return root, temp


def load_artifacts(
    root: Path,
    preprocessor_path: Path,
    data_path: Path,
    *,
    split: str,
    num_cpu: int,
):
    with preprocessor_path.open("rb") as handle:
        data_config, preprocessors = pickle.load(handle)
    if OmegaConf.is_config(data_config):
        data_config = OmegaConf.to_container(data_config, resolve=True)
    data_config, datasets = build_dataset_multimodal(
        OmegaConf.create(copy.deepcopy(data_config)),
        data_path=str(data_path),
        cv_split=0,
        splitting=split,
        augment_config=None,
        num_cpu=max(1, num_cpu),
        mixture_config=None,
    )
    if OmegaConf.is_config(data_config):
        data_config = OmegaConf.to_container(data_config, resolve=True)
    return data_config, preprocessors, datasets


def loader(
    dataset: Dataset,
    data_config: dict[str, Any],
    preprocessors: dict[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    collator = MultiModalDataCollator(
        preprocessors=preprocessors,
        data_config=data_config,
        dataset=DatasetDict({"train": dataset}),
        model_type="CustomModel",
        extra_columns=["row_id", "Formula"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def make_model(
    data_config: dict[str, Any], preprocessors: dict[str, Any], root: Path, *, lr: float
) -> HFWrapper:
    config = OmegaConf.to_container(
        OmegaConf.load(root / "configs/model/custom_model.yaml"), resolve=True
    )
    config.update(
        {
            "model_name": str(root / "vendor/facebook_bart_base"),
            "batch_size": 32,
            "lr": lr,
            # These match the saved official 34a checkpoint.  Using the base
            # YAML defaults here would make the state dict incompatible.
            "positional_encoding_type": "learned",
            "gated_linear": True,
            "max_position_embeddings": 1024,
            "guided_generation": False,
            "n_beams": 10,
            "rejection_sampling": False,
        }
    )
    return HFWrapper(
        data_config=copy.deepcopy(data_config),
        target_tokenizer=preprocessors["Smiles"],
        modality_dropout=None,
        **config,
    )


def load_checkpoint(model: HFWrapper, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if any(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def feature_vector(batch: dict[str, Any]) -> torch.Tensor:
    pad = batch["encoder_pad_mask"]
    offset = 0
    values: list[torch.Tensor] = []
    formula_len = None
    for name in INPUTS:
        length = modality_length(batch["encoder_input"][name])
        observed = (~pad[offset : offset + length]).sum(0).float().log1p()
        if name == "Formula":
            formula_len = observed
        else:
            values.append(observed)
        offset += length
    assert formula_len is not None
    presence = torch.stack([value.gt(0).float() for value in values], dim=0)
    return torch.cat([formula_len[None], torch.stack(values), presence], dim=0).T


def autocast_context(precision: str):
    if not torch.cuda.is_available() or precision == "none":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )


def generate(
    model: HFWrapper,
    batch: dict[str, Any],
    keep: tuple[str, ...],
    beams: int,
    precision: str,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad(), autocast_context(precision):
        return model.generate(view_batch(batch, keep), n_beams=beams)


def decoded_beams(model: HFWrapper, tokens: torch.Tensor, beams: int) -> list[list[str]]:
    decoded = model.target_tokenizer.batch_decode(tokens, skip_special_tokens=True)
    return [decoded[i * beams : (i + 1) * beams] for i in range(tokens.shape[0] // beams)]


def source_router(
    root: Path,
    run: Path,
    checkpoint: Path,
    preprocessor: Path,
    source_data: Path,
    samples: int,
    router_epochs: int,
    batch_size: int,
    workers: int,
    beams: int,
    precision: str,
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_config, preprocessors, splits = load_artifacts(
        root, preprocessor, source_data, split="paper_random", num_cpu=workers
    )
    source_pool = splits["train"]
    order = np.random.default_rng(seed).permutation(len(source_pool))[: min(samples, len(source_pool))]
    source = source_pool.select(order.tolist()).add_column("row_id", list(range(len(order))))
    dl = loader(source, data_config, preprocessors, batch_size=batch_size, shuffle=False, workers=workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = make_model(data_config, preprocessors, root, lr=1e-4).to(device).eval()
    load_checkpoint(teacher, checkpoint)
    features, rewards = [], []
    with torch.inference_mode():
        for batch in dl:
            batch = move(batch, device)
            features.append(feature_vector(batch).cpu())
            targets = [str(value) for value in batch["target_smiles"]]
            action_scores = []
            for action in ACTIONS:
                pred_tokens = generate(teacher, batch, action, beams, precision)
                pred_beams = decoded_beams(teacher, pred_tokens.cpu(), beams)
                scores = []
                for candidates, target in zip(pred_beams, targets):
                    target_can = canonical(target)
                    scores.append(
                        max(
                            1.0 if canonical(candidate) == target_can else tanimoto(candidate, target)
                            for candidate in candidates
                        )
                    )
                action_scores.append(torch.tensor(scores, dtype=torch.float32))
            rewards.append(torch.stack(action_scores, dim=1).cpu())
    x = torch.cat(features)
    reward = torch.cat(rewards)
    labels = reward.argmax(1)
    split = max(1, int(0.8 * len(x)))
    mean, std = x[:split].mean(0), x[:split].std(0).clamp_min(1e-5)
    xn = (x - mean) / std
    net = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(), nn.Linear(64, len(ACTIONS)))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    counts = torch.bincount(labels[:split], minlength=len(ACTIONS)).float()
    weights = counts.clamp_min(1).reciprocal().sqrt()
    weights = weights / weights.mean().clamp_min(1e-6)
    history = []
    for epoch in range(1, router_epochs + 1):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(net(xn[:split]), labels[:split], weight=weights)
        loss.backward()
        opt.step()
        with torch.no_grad():
            train_pred = net(xn[:split]).argmax(1)
            val_pred = net(xn[split:]).argmax(1) if split < len(x) else torch.empty(0, dtype=torch.long)
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "train_action_accuracy": float((train_pred == labels[:split]).float().mean()),
                "validation_action_accuracy": float((val_pred == labels[split:]).float().mean()) if len(val_pred) else None,
            }
        )
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "actions": ACTIONS,
        "input_dim": x.shape[1],
        "samples": len(x),
        "label_counts": torch.bincount(labels, minlength=len(ACTIONS)).tolist(),
        "reward_mean_by_action": reward.mean(0).tolist(),
        "history": history,
    }
    torch.save(payload, run / "router.pt")
    (run / "router_summary.json").write_text(json.dumps({key: value for key, value in payload.items() if key not in {"state_dict", "feature_mean", "feature_std"}}, indent=2, default=list))
    np.savez(run / "router_training_arrays.npz", features=x.numpy(), rewards=reward.numpy(), labels=labels.numpy())


def load_router(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    input_dim = int(payload["input_dim"])
    if input_dim == 14 and "feature_schema" in payload:
        # The augmented router uses modality/formula compatibility, pair
        # agreement, and global availability features.  Keep the class
        # definition in the training module so route and train cannot drift.
        from official34a_router_augmented import Router as AugmentedRouter

        net = AugmentedRouter(dim=input_dim)
        feature_mode = "augmented"
    else:
        net = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, len(ACTIONS)))
        feature_mode = "legacy"
    net.load_state_dict(payload["state_dict"])
    return net.eval(), payload["feature_mean"], payload["feature_std"], feature_mode


def router_input(
    teacher: HFWrapper,
    batch: dict[str, Any],
    feature_mode: str,
) -> torch.Tensor:
    if feature_mode == "legacy":
        return feature_vector(batch)
    # Import lazily to avoid a module cycle when the augmented trainer imports
    # this runner as its base helper module.
    from official34a_router_augmented import extract_features, router_features

    features, masks = extract_features(teacher, batch)
    modality, pair, global_features, _ = router_features(features, masks)
    return torch.cat((modality.flatten(1), pair.flatten(1), global_features), dim=1)


def education_dataset(root, preprocessor, education_data, workers):
    data_dir, temp = ensure_dataset_dir(education_data)
    config, preprocessors, splits = load_artifacts(root, preprocessor, data_dir, split="given_splits", num_cpu=workers)
    return temp, config, preprocessors, splits["test"]


def route_education(
    root: Path,
    run: Path,
    checkpoint: Path,
    preprocessor: Path,
    education_data: Path,
    router_path: Path,
    batch_size: int,
    workers: int,
    precision: str,
) -> None:
    temp, data_config, preprocessors, edu = education_dataset(root, preprocessor, education_data, workers)
    try:
        edu = edu.add_column("row_id", list(range(len(edu))))
        dl = loader(edu, data_config, preprocessors, batch_size=batch_size, shuffle=False, workers=workers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher = make_model(data_config, preprocessors, root, lr=1e-4).to(device).eval()
        load_checkpoint(teacher, checkpoint)
        router, mean, std, feature_mode = load_router(router_path)
        router = router.to(device)
        rows = []
        with torch.inference_mode():
            for batch in dl:
                batch = move(batch, device)
                inputs = router_input(teacher, batch, feature_mode)
                actions = router((inputs - mean.to(device)) / std.to(device)).argmax(1).tolist()
                for local, action in enumerate(actions):
                    one = select_batch(batch, [local], len(actions))
                    tokens = generate(teacher, one, ACTIONS[action], 1, precision)[0].cpu().tolist()
                    rows.append({
                        "row_id": len(rows),
                        "action": int(action),
                        "modalities": list(ACTIONS[action]),
                        "token_ids": tokens,
                        "pseudo_smiles": teacher.target_tokenizer.decode(tokens, skip_special_tokens=True),
                    })
        run.mkdir(parents=True, exist_ok=True)
        (run / "route_manifest.json").write_text(json.dumps({"rows": rows, "actions": [list(a) for a in ACTIONS], "target_labels_loaded": False}, indent=2))
    finally:
        if temp is not None:
            temp.cleanup()


def pseudo_batch(batch: dict[str, Any], tokens: torch.Tensor, tokenizer) -> dict[str, Any]:
    row = dict(batch)
    row["decoder_input"] = {"Smiles": tokens[:, :-1].T.contiguous()}
    row["target"] = tokens[:, 1:].T.contiguous()
    row["decoder_pad_mask"] = row["target"].eq(int(tokenizer.pad_token_id))
    row["target_mask"] = row["decoder_pad_mask"]
    return row


def train_epoch(model, dl, token_map, optimizer, *, keep_by_batch: Iterable[tuple[str, ...] | None], device, tokenizer):
    model.train()
    total, steps = 0.0, 0
    for batch, keep in zip(dl, keep_by_batch):
        batch = move(batch, device)
        ids = [int(value) for value in batch["row_id"]]
        sequences = [torch.tensor(token_map[key], device=device, dtype=torch.long) for key in ids]
        tokens = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=int(tokenizer.pad_token_id)
        )
        if keep is not None:
            batch = view_batch(batch, keep)
        output = model(pseudo_batch(batch, tokens, tokenizer))
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
        optimizer.step()
        total += float(output.loss.detach())
        steps += 1
    return total / max(steps, 1)


def pseudo_validation_loss(
    model,
    dl,
    token_map,
    *,
    device,
    tokenizer,
    keep_by_batch: Iterable[tuple[str, ...] | None] | None = None,
) -> float:
    model.eval()
    total, steps = 0.0, 0
    keep_iter = iter(keep_by_batch) if keep_by_batch is not None else None
    with torch.inference_mode():
        for batch in dl:
            keep = next(keep_iter) if keep_iter is not None else None
            batch = move(batch, device)
            ids = [int(value) for value in batch["row_id"]]
            sequences = [torch.tensor(token_map[key], device=device, dtype=torch.long) for key in ids]
            tokens = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=int(tokenizer.pad_token_id)
            )
            if keep is not None:
                batch = view_batch(batch, keep)
            loss = model(pseudo_batch(batch, tokens, tokenizer)).loss
            total += float(loss)
            steps += 1
    return total / max(steps, 1)


def split_pseudolabel_rows(row_ids: list[int], seed: int, fraction: float = 0.15) -> tuple[list[int], list[int]]:
    if len(row_ids) < 4:
        raise ValueError(f"Need at least four pseudo-labelled rows for early stopping; got {len(row_ids)}")
    ordered = list(row_ids)
    random.Random(seed).shuffle(ordered)
    n_validation = min(len(ordered) - 1, max(1, round(len(ordered) * fraction)))
    validation_ids = sorted(ordered[:n_validation])
    validation_set = set(validation_ids)
    train_ids = sorted(row_id for row_id in ordered if row_id not in validation_set)
    return train_ids, validation_ids


def train_early_stopping(
    model,
    train_dl,
    validation_dl,
    token_map,
    optimizer,
    *,
    max_epochs: int,
    patience: int,
    lr_patience: int,
    min_lr: float,
    min_epochs: int,
    min_delta: float,
    device,
    tokenizer,
    checkpoint_dir: Path,
    phase: str,
    validation_keep_by_batch: list[tuple[str, ...] | None] | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], int, int]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state = None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_patience,
        threshold=min_delta,
        threshold_mode="abs",
        min_lr=min_lr,
    )
    for epoch in range(1, max_epochs + 1):
        rng = random.Random(seed + epoch)
        training_keep = None
        if phase == "stage2":
            training_keep = []
            for _ in range(len(train_dl)):
                dropped = set(rng.sample(list(SPECTRAL), rng.randint(1, 3)))
                training_keep.append(tuple(name for name in SPECTRAL if name not in dropped))
        train_loss = train_epoch(
            model,
            train_dl,
            token_map,
            optimizer,
            keep_by_batch=training_keep if training_keep is not None else [None] * len(train_dl),
            device=device,
            tokenizer=tokenizer,
        )
        val_loss = pseudo_validation_loss(
            model,
            validation_dl,
            token_map,
            device=device,
            tokenizer=tokenizer,
            keep_by_batch=validation_keep_by_batch,
        )
        scheduler.step(val_loss)
        improved = val_loss < best_loss - min_delta
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "train_pseudo_ce": train_loss,
            "validation_pseudo_ce": val_loss,
            "best_validation_pseudo_ce": best_loss,
            "improved": improved,
            "patience_used": stale_epochs,
            "learning_rate": learning_rate,
        }
        history.append(record)
        print(json.dumps({"phase": phase, **record}), flush=True)
        if epoch % 6 == 0:
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, checkpoint_dir / f"epoch_{epoch}.pt")
        if epoch >= min_epochs and stale_epochs >= patience:
            break
    stopped_epoch = history[-1]["epoch"]
    if best_state is None:
        raise RuntimeError(f"{phase} did not produce a valid best checkpoint")
    torch.save({"state_dict": model.state_dict(), "epoch": stopped_epoch}, checkpoint_dir / "stopped.pt")
    torch.save({"state_dict": best_state, "epoch": best_epoch}, checkpoint_dir / "best.pt")
    model.load_state_dict(best_state, strict=True)
    return history, best_epoch, stopped_epoch


def deterministic_dropout_schedule(n_batches: int, seed: int) -> list[tuple[str, ...]]:
    rng = random.Random(seed)
    schedule = []
    for _ in range(n_batches):
        dropped = set(rng.sample(list(SPECTRAL), rng.randint(1, 3)))
        schedule.append(tuple(name for name in SPECTRAL if name not in dropped))
    return schedule


def full_greedy_manifest(model, dl, device, precision) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with torch.inference_mode():
        for batch in dl:
            batch = move(batch, device)
            tokens = generate(model, batch, SPECTRAL, 1, precision).cpu()
            decoded = model.target_tokenizer.batch_decode(tokens, skip_special_tokens=True)
            for row_id, token_ids, raw in zip(batch["row_id"], tokens.tolist(), decoded):
                rows[int(row_id)] = {"token_ids": token_ids, "pseudo_smiles": raw}
    return rows


def stable_consensus(snapshot_rows: list[dict[int, dict[str, Any]]], edu_rows: dict[int, dict[str, Any]]) -> dict[int, list[int]]:
    if not snapshot_rows:
        return {}
    result = {}
    for row_id in sorted(set.intersection(*(set(snapshot) for snapshot in snapshot_rows))):
        canonical_predictions = [canonical(snapshot[row_id]["pseudo_smiles"]) for snapshot in snapshot_rows]
        if canonical_predictions[0] is None or len(set(canonical_predictions)) != 1:
            continue
        observed_formula = str(edu_rows[row_id].get("Formula", ""))
        if observed_formula and molecular_formula(canonical_predictions[0]) != observed_formula:
            continue
        result[row_id] = snapshot_rows[0][row_id]["token_ids"]
    return result


def evaluate_checkpoint(model, dl, device, precision, output: Path) -> dict[str, float]:
    total = top1 = top5 = top10 = valid = 0
    with torch.inference_mode():
        for batch in dl:
            batch = move(batch, device)
            tokens = generate(model, batch, SPECTRAL, 10, precision).cpu()
            candidates = decoded_beams(model, tokens, 10)
            for choices, target in zip(candidates, batch["target_smiles"]):
                target_can = canonical(target)
                candidate_can = [canonical(value) for value in choices]
                valid += int(candidate_can[0] is not None)
                total += 1
                top1 += int(target_can in candidate_can[:1])
                top5 += int(target_can in candidate_can[:5])
                top10 += int(target_can in candidate_can[:10])
    result = {"count": total, "top1": top1 / max(total, 1), "top5": top5 / max(total, 1), "top10": top10 / max(total, 1), "valid_top1": valid / max(total, 1)}
    output.write_text(json.dumps(result, indent=2))
    return result


def ttt(
    root: Path,
    run: Path,
    checkpoint: Path,
    preprocessor: Path,
    education_data: Path,
    router_path: Path,
    route_manifest_path: Path | None,
    epochs: int,
    batch_size: int,
    workers: int,
    precision: str,
    seed: int,
    ttt_lr: float,
    stage2_lr: float,
    early_stopping: bool,
    stage1_max_epochs: int,
    stage2_max_epochs: int,
    patience: int,
    lr_patience: int,
    min_lr: float,
    min_epochs: int,
    min_delta: float,
) -> None:
    temp, data_config, preprocessors, edu = education_dataset(root, preprocessor, education_data, workers)
    try:
        edu = edu.add_column("row_id", list(range(len(edu))))
        dl = loader(edu, data_config, preprocessors, batch_size=batch_size, shuffle=False, workers=workers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = make_model(data_config, preprocessors, root, lr=ttt_lr).to(device)
        load_checkpoint(base, checkpoint)
        manifest_path = route_manifest_path or (run / "route_manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"route manifest not found: {manifest_path}; run route first or pass --route-manifest"
            )
        route_manifest = json.loads(manifest_path.read_text())
        route_rows = {int(row["row_id"]): row for row in route_manifest["rows"]}
        token_map = {key: row["token_ids"] for key, row in route_rows.items()}
        student = copy.deepcopy(base).to(device)
        if not early_stopping:
            optimizer = torch.optim.AdamW(student.parameters(), lr=ttt_lr, weight_decay=1e-4)
        stage1 = run / "stage1"
        stage1.mkdir(parents=True, exist_ok=True)
        if early_stopping:
            train_ids, validation_ids = split_pseudolabel_rows(sorted(token_map), seed + 100)
            train_dl = loader(edu.select(train_ids), data_config, preprocessors, batch_size=batch_size, shuffle=False, workers=workers)
            validation_dl = loader(edu.select(validation_ids), data_config, preprocessors, batch_size=batch_size, shuffle=False, workers=workers)
            stage1_optimizer = torch.optim.AdamW(student.parameters(), lr=ttt_lr, weight_decay=1e-4)
            history, stage1_best_epoch, stage1_stopped_epoch = train_early_stopping(
                student, train_dl, validation_dl, token_map, stage1_optimizer,
                max_epochs=stage1_max_epochs, patience=patience, lr_patience=lr_patience,
                min_lr=min_lr, min_epochs=min_epochs,
                min_delta=min_delta, device=device, tokenizer=student.target_tokenizer,
                checkpoint_dir=stage1, phase="stage1", seed=seed + 1000,
            )
            (run / "stage1_split.json").write_text(json.dumps({"train_row_ids": train_ids, "validation_row_ids": validation_ids}, indent=2))
        else:
            history = []
            for epoch in range(1, epochs + 1):
                loss = train_epoch(student, dl, token_map, optimizer, keep_by_batch=[None] * len(dl), device=device, tokenizer=student.target_tokenizer)
                history.append({"epoch": epoch, "loss": loss})
                if epoch % 3 == 0 or epoch == epochs:
                    torch.save({"state_dict": student.state_dict(), "epoch": epoch, "history": history}, stage1 / f"epoch_{epoch}.pt")
            stage1_best_epoch = epochs
            stage1_stopped_epoch = epochs
        (stage1 / "history.json").write_text(json.dumps(history, indent=2))

        snapshots = []
        if early_stopping:
            available_epochs = list(range(6, stage1_stopped_epoch + 1, 6))
            if len(available_epochs) > 4:
                available_epochs = [available_epochs[round((i + 1) * len(available_epochs) / 4) - 1] for i in range(4)]
            snapshot_epochs = sorted(set(available_epochs + [stage1_best_epoch, stage1_stopped_epoch]))
        else:
            snapshot_epochs = [epoch for epoch in (3, 6, 9, 12) if epoch <= epochs]
        prediction_dir = run / "predictions"
        prediction_dir.mkdir(exist_ok=True)
        for epoch in snapshot_epochs:
            snapshot = copy.deepcopy(base).to(device)
            if epoch == stage1_best_epoch:
                checkpoint_path = stage1 / "best.pt"
            elif epoch == stage1_stopped_epoch:
                checkpoint_path = stage1 / "stopped.pt"
            else:
                checkpoint_path = stage1 / f"epoch_{epoch}.pt"
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            snapshot.load_state_dict(payload["state_dict"], strict=True)
            snapshot.eval()
            rows = full_greedy_manifest(snapshot, dl, device, precision)
            snapshots.append(rows)
            (prediction_dir / f"stage1_epoch_{epoch}.json").write_text(json.dumps(rows, indent=2))
        edu_rows = {int(row["row_id"]): row for row in edu}
        stable = stable_consensus(snapshots, edu_rows) if len(snapshots) >= 2 else token_map
        (run / "stable_consensus.json").write_text(json.dumps({str(key): value for key, value in stable.items()}, indent=2))

        stage2 = run / "stage2"
        stage2.mkdir(parents=True, exist_ok=True)
        stable_ids = sorted(stable)
        stable_ds = edu.select(stable_ids)
        if early_stopping:
            stage1_validation_set = set(validation_ids)
            stage2_validation_ids = sorted(set(stable_ids) & stage1_validation_set)
            stage2_train_ids = sorted(set(stable_ids) - stage1_validation_set)
            if len(stage2_validation_ids) < 4 or not stage2_train_ids:
                raise ValueError(
                    "Too few Stage 2 stable pseudo-labels remain in the held-out split: "
                    f"train={len(stage2_train_ids)}, validation={len(stage2_validation_ids)}"
                )
            stage2_train_dl = loader(edu.select(stage2_train_ids), data_config, preprocessors, batch_size=batch_size, shuffle=False, workers=workers)
            stage2_validation_dl = loader(edu.select(stage2_validation_ids), data_config, preprocessors, batch_size=batch_size, shuffle=False, workers=workers)
            stage2_validation_keep = deterministic_dropout_schedule(len(stage2_validation_dl), seed + 250)
            stage2_optimizer = torch.optim.AdamW(student.parameters(), lr=stage2_lr, weight_decay=1e-4)
            stage2_history, stage2_best_epoch, stage2_stopped_epoch = train_early_stopping(
                student, stage2_train_dl, stage2_validation_dl, stable, stage2_optimizer,
                max_epochs=stage2_max_epochs, patience=patience, lr_patience=lr_patience,
                min_lr=min_lr, min_epochs=min_epochs,
                min_delta=min_delta, device=device, tokenizer=student.target_tokenizer,
                checkpoint_dir=stage2, phase="stage2", validation_keep_by_batch=stage2_validation_keep,
                seed=seed + 5000,
            )
            (run / "stage2_split.json").write_text(json.dumps({"train_row_ids": stage2_train_ids, "validation_row_ids": stage2_validation_ids}, indent=2))
        else:
            stage2_history = []
            stable_dl = loader(stable_ds, data_config, preprocessors, batch_size=batch_size, shuffle=False, workers=workers)
            for epoch in range(1, 4):
                rng = random.Random(seed + 5000 + epoch)
                schedule = []
                for _ in range(len(stable_dl)):
                    dropped = set(rng.sample(list(SPECTRAL), rng.randint(1, 3)))
                    schedule.append(tuple(name for name in SPECTRAL if name not in dropped))
                loss = train_epoch(student, stable_dl, stable, optimizer, keep_by_batch=schedule, device=device, tokenizer=student.target_tokenizer)
                stage2_history.append({"epoch": epoch, "loss": loss, "stable_rows": len(stable)})
                torch.save({"state_dict": student.state_dict(), "epoch": epoch, "history": stage2_history}, stage2 / f"epoch_{epoch}.pt")
            stage2_best_epoch = 3
            stage2_stopped_epoch = 3
        (stage2 / "history.json").write_text(json.dumps(stage2_history, indent=2))
        evaluation_dl = loader(edu, data_config, preprocessors, batch_size=batch_size, shuffle=False, workers=workers)
        evaluate_checkpoint(base.eval(), evaluation_dl, device, precision, run / "baseline_metrics.json")
        if early_stopping:
            stage1_best = copy.deepcopy(base).to(device)
            stage1_payload = torch.load(stage1 / "best.pt", map_location="cpu", weights_only=False)
            stage1_best.load_state_dict(stage1_payload["state_dict"], strict=True)
            evaluate_checkpoint(stage1_best.eval(), evaluation_dl, device, precision, run / "stage1_metrics.json")
        evaluate_checkpoint(student.eval(), evaluation_dl, device, precision, run / "stage2_metrics.json")
        (run / "experiment.complete.json").write_text(json.dumps({
            "complete": True,
            "early_stopping": early_stopping,
            "stage1_lr": ttt_lr,
            "stage2_lr": stage2_lr if early_stopping else ttt_lr,
            "stage1_best_epoch": stage1_best_epoch,
            "stage1_stopped_epoch": stage1_stopped_epoch,
            "stage2_best_epoch": stage2_best_epoch,
            "stage2_stopped_epoch": stage2_stopped_epoch,
            "stage1_max_epochs": stage1_max_epochs,
            "stage2_max_epochs": stage2_max_epochs,
            "early_stopping_patience": patience,
            "lr_scheduler_patience": lr_patience,
            "min_validation_delta": min_delta,
            "route_rows": len(route_rows),
            "stable_rows": len(stable),
        }, indent=2))
    finally:
        if temp is not None:
            temp.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("router", "route", "ttt"), required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--preprocessor", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--source-data", type=Path)
    ap.add_argument("--education-data", type=Path)
    ap.add_argument("--router", type=Path)
    ap.add_argument(
        "--route-manifest",
        type=Path,
        help="Existing route_manifest.json; defaults to <run>/route_manifest.json in TTT mode.",
    )
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--router-epochs", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--precision", choices=("bf16", "fp16", "none"), default="bf16")
    ap.add_argument("--seed", type=int, default=3247)
    ap.add_argument("--ttt-lr", type=float, default=1e-5)
    ap.add_argument("--stage2-lr", type=float, default=5e-7)
    ap.add_argument("--early-stopping", action="store_true")
    ap.add_argument("--stage1-max-epochs", type=int, default=18)
    ap.add_argument("--stage2-max-epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr-patience", type=int, default=2)
    ap.add_argument("--min-lr", type=float, default=1e-8)
    ap.add_argument("--min-epochs", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    if args.mode == "ttt":
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if args.mode == "router":
        source_router(args.root, args.run, args.checkpoint, args.preprocessor, args.source_data, args.samples, args.router_epochs, args.batch_size, args.workers, args.beams, args.precision, args.seed)
    elif args.mode == "route":
        route_education(args.root, args.run, args.checkpoint, args.preprocessor, args.education_data, args.router, args.batch_size, args.workers, args.precision)
    else:
        ttt(args.root, args.run, args.checkpoint, args.preprocessor, args.education_data, args.router, args.route_manifest, args.epochs, args.batch_size, args.workers, args.precision, args.seed, args.ttt_lr, args.stage2_lr, args.early_stopping, args.stage1_max_epochs, args.stage2_max_epochs, args.patience, args.lr_patience, args.min_lr, args.min_epochs, args.min_delta)


if __name__ == "__main__":
    main()

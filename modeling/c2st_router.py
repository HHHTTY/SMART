"""C2ST-guided modality-gap routing utilities.

The representation extractor deliberately encodes each modality in isolation.
It never concatenates modalities before the domain score is computed, so a
shared encoder cannot hide a modality-specific source/target shift.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .modality_subset_router import ROUTED_MODALITIES
from .pmgfa_shift_features import masked_pool


C2ST_STAGES = ("embedding", "layer3", "layer6", "post_norm")


def _encoder(model: Any) -> Any:
    hf = model.hf_model
    return hf.model.encoder if hasattr(hf, "model") else hf.encoder


@torch.no_grad()
def extract_independent_c2st_features(
    model: Any,
    batch: Mapping[str, Any],
    *,
    modalities: Sequence[str] = ROUTED_MODALITIES,
    layer_indices: Sequence[int] = (3, 6),
) -> dict[str, torch.Tensor]:
    """Return ``[batch, 4, hidden]`` features for each isolated modality.

    ``layer_indices`` is one-based, matching the paper protocol. Missing
    modalities are omitted; callers must handle availability explicitly.
    """
    requested = tuple(str(name) for name in modalities)
    if tuple(layer_indices) != (3, 6):
        raise ValueError("C2ST extraction currently requires layers 3 and 6")
    outputs: dict[str, torch.Tensor] = {}
    original_input_ids = batch.get("encoder_input")
    if not isinstance(original_input_ids, Mapping):
        raise ValueError("C2ST extraction requires batch['encoder_input']")
    full_pad_mask = batch.get("encoder_pad_mask")
    modality_pad_masks = batch.get("encoder_modality_pad_masks")
    offset = 0
    for name, value in original_input_ids.items():
        name = str(name)
        length = int(model._input_sequence_length(value))
        if name not in requested:
            offset += length
            continue
        # Build a one-modality batch so fusion/context modules cannot leak
        # other modalities into the C2ST representation.
        single_batch = dict(batch)
        single_batch["encoder_input"] = {name: value}
        if isinstance(modality_pad_masks, Mapping) and name in modality_pad_masks:
            single_batch["encoder_modality_pad_masks"] = {
                name: modality_pad_masks[name]
            }
        elif isinstance(full_pad_mask, torch.Tensor):
            single_batch["encoder_pad_mask"] = full_pad_mask[offset : offset + length]
        input_ids, attention_mask = model.prepare_encoder_inputs(single_batch)
        if tuple(str(key) for key in input_ids) != (name,):
            raise RuntimeError(f"isolated C2ST batch changed modality ordering for {name}")
        hidden = model.multimodal_embedding(input_ids)
        encoder = _encoder(model)
        layers = getattr(encoder, "layers", None)
        if layers is None or len(layers) < 6:
            raise ValueError("C2ST extraction requires an encoder with at least six layers")
        mask = attention_mask.bool()
        stage_rows = [masked_pool(hidden, mask)]
        for index, layer in enumerate(layers, start=1):
            hidden = layer(hidden, src_key_padding_mask=~mask)
            if index == 3:
                stage_rows.append(masked_pool(hidden, mask))
            elif index == 6:
                stage_rows.append(masked_pool(hidden, mask))
                stage_rows.append(masked_pool(encoder.norm(hidden), mask))
                break
        if len(stage_rows) != len(C2ST_STAGES):
            raise RuntimeError(f"failed to extract all C2ST stages for {name}")
        outputs[name] = torch.stack(stage_rows, dim=1).float()
        offset += length
    return outputs


class DomainClassifier(nn.Module):
    """Small source-vs-target classifier for one modality/stage."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid domain classifier dimensions")
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError("domain classifier expects [batch, feature]")
        return self.net(value).squeeze(-1)


@dataclass(frozen=True)
class C2STCalibration:
    """Calibration for one modality/stage domain classifier.

    ``source_percentile`` keeps backward compatibility with online checkpoints.
    Offline C2ST uses balanced held-out Platt log-odds so target samples beyond
    the source score range remain distinguishable instead of all clipping to 1.
    """

    sorted_scores: torch.Tensor | None = None
    method: str = "source_percentile"
    slope: float = 1.0
    intercept: float = 0.0

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
        if self.method == "platt_log_odds":
            return scores.detach().float() * self.slope + self.intercept
        if self.method != "source_percentile" or self.sorted_scores is None:
            raise ValueError(f"unsupported C2ST calibration method: {self.method}")
        reference = self.sorted_scores.to(scores.device, scores.dtype)
        if reference.ndim != 1 or reference.numel() < 2:
            raise ValueError("calibration requires at least two source scores")
        positions = torch.searchsorted(
            reference, scores.detach().flatten().contiguous()
        ).clamp(
            0, reference.numel() - 1
        )
        return (positions.reshape(scores.shape).float() + 0.5) / reference.numel()

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"method": self.method}
        if self.method == "source_percentile":
            if self.sorted_scores is None:
                raise ValueError("source percentile calibration has no reference scores")
            payload["sorted_scores"] = self.sorted_scores.float().cpu()
        elif self.method == "platt_log_odds":
            payload.update(slope=float(self.slope), intercept=float(self.intercept))
        else:
            raise ValueError(f"unsupported C2ST calibration method: {self.method}")
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> "C2STCalibration":
        # Older checkpoints stored the source reference tensor directly.
        if not isinstance(payload, Mapping):
            return cls(torch.as_tensor(payload).float().cpu())
        method = str(payload.get("method", "source_percentile"))
        if method == "source_percentile":
            return cls(
                torch.as_tensor(payload["sorted_scores"]).float().cpu(),
                method=method,
            )
        if method == "platt_log_odds":
            return cls(
                method=method,
                slope=float(payload["slope"]),
                intercept=float(payload["intercept"]),
            )
        raise ValueError(f"unsupported C2ST calibration method: {method}")


def fit_percentile_calibration(scores: torch.Tensor) -> C2STCalibration:
    values = scores.detach().float().flatten()
    values = values[torch.isfinite(values)]
    if values.numel() < 2:
        raise ValueError("cannot calibrate C2ST scores from fewer than two values")
    return C2STCalibration(torch.sort(values).values.cpu())


def fit_platt_log_odds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    regularization: float = 1e-3,
) -> C2STCalibration:
    """Fit monotone Platt log-odds on a balanced held-out domain split."""
    values = logits.detach().float().flatten().cpu()
    truth = labels.detach().float().flatten().cpu()
    finite = torch.isfinite(values) & torch.isfinite(truth)
    values, truth = values[finite], truth[finite]
    if values.numel() < 4 or values.numel() != truth.numel():
        raise ValueError("Platt calibration requires at least four finite rows")
    if not truth.eq(0).any() or not truth.eq(1).any():
        raise ValueError("Platt calibration requires both domain labels")
    if regularization < 0:
        raise ValueError("Platt regularization must be non-negative")

    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(1e-6)
    standardized = (values - mean) / std
    raw_slope = torch.tensor(0.54132485, requires_grad=True)  # softplus^-1(1)
    intercept = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (raw_slope, intercept),
        lr=0.5,
        max_iter=100,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        slope = torch.nn.functional.softplus(raw_slope) + 1e-6
        calibrated = standardized * slope + intercept
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            calibrated, truth
        )
        loss = loss + regularization * (slope.square() + intercept.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    standardized_slope = float(
        (torch.nn.functional.softplus(raw_slope.detach()) + 1e-6).item()
    )
    fitted_intercept = float(intercept.detach())
    effective_slope = standardized_slope / float(std)
    effective_intercept = fitted_intercept - effective_slope * float(mean)
    return C2STCalibration(
        method="platt_log_odds",
        slope=effective_slope,
        intercept=effective_intercept,
    )


def trimmed_mean(values: torch.Tensor, trim_fraction: float = 0.10) -> torch.Tensor:
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("trimmed_mean expects a non-empty vector")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim fraction must lie in [0, .5)")
    ordered = values.float().sort().values
    trim = int(ordered.numel() * trim_fraction)
    kept = ordered[trim : ordered.numel() - trim if trim else ordered.numel()]
    return kept.mean()


def aggregate_modality_gaps(
    calibrated_scores: Mapping[str, torch.Tensor],
    *,
    trim_fraction: float = 0.10,
    lambda_std: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Aggregate four stage scores into one per-modality batch gap."""
    if lambda_std < 0:
        raise ValueError("lambda_std must be non-negative")
    result: dict[str, torch.Tensor] = {}
    for name, values in calibrated_scores.items():
        if values.ndim == 1:
            row_scores = values.float()
        elif values.ndim == 2 and values.shape[1] == len(C2ST_STAGES):
            row_scores = values.float().mean(dim=1)
        else:
            raise ValueError("calibrated scores must have shape [batch] or [batch, 4]")
        result[name] = trimmed_mean(row_scores, trim_fraction) + lambda_std * row_scores.std(
            unbiased=False
        )
    return result


def aggregate_gap_tensor(
    calibrated_scores: torch.Tensor,
    *,
    trim_fraction: float = 0.10,
    lambda_std: float = 0.25,
) -> torch.Tensor:
    """Return one batch gap for every routed modality.

    The input is ``[batch, modality, stage]``. Stages are averaged per sample
    before robust aggregation across the batch.
    """
    if calibrated_scores.ndim != 3:
        raise ValueError("calibrated scores must have shape [batch, modality, stage]")
    if calibrated_scores.shape[1] != len(ROUTED_MODALITIES):
        raise ValueError("calibrated scores have an unexpected modality dimension")
    if calibrated_scores.shape[2] != len(C2ST_STAGES):
        raise ValueError("calibrated scores have an unexpected stage dimension")
    if lambda_std < 0:
        raise ValueError("lambda_std must be non-negative")
    gaps = []
    for index in range(calibrated_scores.shape[1]):
        row_scores = calibrated_scores[:, index].float().mean(dim=1)
        gaps.append(
            trimmed_mean(row_scores, trim_fraction)
            + lambda_std * row_scores.std(unbiased=False)
        )
    return torch.stack(gaps)


def route_from_gaps(
    gaps: Mapping[str, torch.Tensor],
    availability: Mapping[str, torch.Tensor] | None = None,
) -> tuple[str, float, dict[str, float]]:
    """Select the lowest-gap available modality and return its margin."""
    valid = {}
    for name, value in gaps.items():
        if availability is not None and not bool(availability.get(name, torch.tensor(False)).all()):
            continue
        valid[name] = float(value.detach().float().cpu())
    if not valid:
        raise ValueError("no available modality has a finite C2ST gap")
    ordered = sorted(valid.items(), key=lambda item: (item[1], item[0]))
    margin = float(ordered[1][1] - ordered[0][1]) if len(ordered) > 1 else float("inf")
    return ordered[0][0], margin, valid


def binary_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute ROC AUC without a sklearn dependency."""
    scores = logits.detach().float().flatten()
    truth = labels.detach().long().flatten()
    if scores.numel() != truth.numel() or not truth.bool().any() or truth.bool().all():
        raise ValueError("AUC requires both binary classes")
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    positives = truth.eq(1)
    negatives = truth.eq(0)
    return float(
        (ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2)
        / (positives.sum() * negatives.sum())
    )


def balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.0) -> float:
    truth = labels.detach().bool().flatten()
    prediction = logits.detach().flatten().ge(threshold)
    positive = truth
    negative = ~truth
    if not positive.any() or not negative.any():
        raise ValueError("balanced accuracy requires both binary classes")
    tpr = prediction[positive].float().mean()
    tnr = (~prediction[negative]).float().mean()
    return float((tpr + tnr) / 2)


def cross_fitted_c2st_gaps(
    router: "C2STRouter",
    source_train: Mapping[str, torch.Tensor],
    source_validation: Mapping[str, torch.Tensor],
    current_target: Mapping[str, torch.Tensor],
    *,
    historical_target: Mapping[str, torch.Tensor] | None = None,
    folds: int = 4,
    steps: int = 20,
    train_batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 3247,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate batch gaps with target-sample-out cross-fitted domain heads.

    Each target row is scored only by a temporary head whose training fold
    excluded that row. The persistent router is never mutated. Stage-level
    gaps are ``2 * abs(AUC - .5)`` and therefore comparable without a
    source-percentile transform that can saturate under severe domain shift.
    """
    if folds < 2 or steps < 1 or train_batch_size < 1:
        raise ValueError("invalid cross-fitted C2ST configuration")
    if lr <= 0 or weight_decay < 0:
        raise ValueError("invalid cross-fitted C2ST optimizer configuration")
    OnlineC2STTrainer._validate_features(source_train)
    OnlineC2STTrainer._validate_features(source_validation)
    OnlineC2STTrainer._validate_features(current_target)
    if historical_target is not None:
        OnlineC2STTrainer._validate_features(historical_target)

    current_rows = int(next(iter(current_target.values())).shape[0])
    if current_rows < 2:
        raise ValueError("cross-fitted C2ST requires at least two target rows")
    fold_count = min(int(folds), current_rows)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(current_rows, generator=generator)
    heldout_folds = [value for value in torch.tensor_split(permutation, fold_count) if len(value)]
    device = next(router.parameters()).device
    oof_logits = torch.empty(
        current_rows,
        len(ROUTED_MODALITIES),
        len(C2ST_STAGES),
        dtype=torch.float32,
    )
    source_logits: dict[str, dict[int, list[torch.Tensor]]] = {
        name: {stage_index: [] for stage_index in range(len(C2ST_STAGES))}
        for name in ROUTED_MODALITIES
    }

    all_current = torch.arange(current_rows)
    source_rows = int(next(iter(source_train.values())).shape[0])
    for heldout in heldout_folds:
        train_mask = torch.ones(current_rows, dtype=torch.bool)
        train_mask[heldout] = False
        current_train = all_current[train_mask]
        fold_target = {
            name: current_target[name].index_select(0, current_train)
            if historical_target is None
            else torch.cat(
                (
                    historical_target[name],
                    current_target[name].index_select(0, current_train),
                ),
                dim=0,
            )
            for name in ROUTED_MODALITIES
        }
        target_rows = int(next(iter(fold_target.values())).shape[0])
        sample_rows = min(int(train_batch_size), source_rows, target_rows)
        if sample_rows < 1:
            raise ValueError("cross-fitted C2ST training fold is empty")

        classifiers = copy.deepcopy(router.classifiers).to(device)
        optimizers = {
            f"{name}/{stage}": torch.optim.AdamW(
                classifiers[name][stage_index].parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            for name in ROUTED_MODALITIES
            for stage_index, stage in enumerate(C2ST_STAGES)
        }
        labels = torch.cat(
            (
                torch.zeros(sample_rows, device=device),
                torch.ones(sample_rows, device=device),
            )
        )
        for _ in range(steps):
            source_index = torch.randint(
                source_rows, (sample_rows,), generator=generator
            )
            target_index = torch.randint(
                target_rows, (sample_rows,), generator=generator
            )
            for name in ROUTED_MODALITIES:
                for stage_index, stage in enumerate(C2ST_STAGES):
                    classifier = classifiers[name][stage_index]
                    classifier.train()
                    source_batch = source_train[name].index_select(
                        0, source_index
                    )[:, stage_index].to(device).float()
                    target_batch = fold_target[name].index_select(
                        0, target_index
                    )[:, stage_index].to(device).float()
                    logits = classifier(
                        torch.cat((source_batch, target_batch), dim=0)
                    )
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels
                    )
                    optimizer = optimizers[f"{name}/{stage}"]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

        with torch.no_grad():
            for modality_index, name in enumerate(ROUTED_MODALITIES):
                for stage_index, classifier in enumerate(classifiers[name]):
                    classifier.eval()
                    target_logits = classifier(
                        current_target[name].index_select(0, heldout)[:, stage_index]
                        .to(device)
                        .float()
                    ).cpu()
                    oof_logits[heldout, modality_index, stage_index] = target_logits
                    source_logits[name][stage_index].append(
                        classifier(
                            source_validation[name][:, stage_index].to(device).float()
                        ).cpu()
                    )

    stage_gaps = torch.empty(len(ROUTED_MODALITIES), len(C2ST_STAGES))
    for modality_index, name in enumerate(ROUTED_MODALITIES):
        for stage_index in range(len(C2ST_STAGES)):
            source_scores = torch.cat(source_logits[name][stage_index])
            target_scores = oof_logits[:, modality_index, stage_index]
            labels = torch.cat(
                (
                    torch.zeros(len(source_scores)),
                    torch.ones(len(target_scores)),
                )
            )
            auc = binary_auc(torch.cat((source_scores, target_scores)), labels)
            stage_gaps[modality_index, stage_index] = 2.0 * abs(auc - 0.5)
    gaps = stage_gaps.mean(dim=1) + router.lambda_std * stage_gaps.std(
        dim=1, unbiased=False
    )
    return oof_logits, stage_gaps, gaps


class C2STRouter(nn.Module):
    """Four independent modality domain classifiers plus score calibration."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 256,
        lambda_std: float = 0.25,
        trim_fraction: float = 0.10,
    ) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.lambda_std = float(lambda_std)
        self.trim_fraction = float(trim_fraction)
        self.classifiers = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [DomainClassifier(feature_dim, hidden_dim=hidden_dim) for _ in C2ST_STAGES]
                )
                for name in ROUTED_MODALITIES
            }
        )
        self.calibrations: dict[str, C2STCalibration] = {}

    def logits(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        rows = []
        for name in ROUTED_MODALITIES:
            value = features[name]
            if value.ndim != 3 or value.shape[1] != len(C2ST_STAGES) or value.shape[2] != self.feature_dim:
                raise ValueError(f"C2ST features for {name} must have shape [batch, 4, {self.feature_dim}]")
            rows.append(
                torch.stack(
                    [classifier(value[:, stage]) for stage, classifier in enumerate(self.classifiers[name])],
                    dim=1,
                )
            )
        return torch.stack(rows, dim=1)

    @torch.no_grad()
    def calibrated_scores(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        logits = self.logits(features)
        expected = {f"{name}/{stage}" for name in ROUTED_MODALITIES for stage in C2ST_STAGES}
        if set(self.calibrations) != expected:
            raise ValueError("C2ST score calibration is missing")
        return torch.stack(
            [
                torch.stack(
                    [
                        self.calibrations[f"{name}/{stage}"].transform(
                            logits[:, index, stage_index]
                        )
                        for stage_index, stage in enumerate(C2ST_STAGES)
                    ],
                    dim=1,
                )
                for index, name in enumerate(ROUTED_MODALITIES)
            ],
            dim=1,
        )

    @torch.no_grad()
    def gaps(self, features: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.calibrated_scores(features)
        gaps = aggregate_gap_tensor(
            scores,
            trim_fraction=self.trim_fraction,
            lambda_std=self.lambda_std,
        )
        return scores, gaps

    @torch.no_grad()
    def select(self, features: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, gaps = self.gaps(features)
        batch_size = scores.shape[0]
        selected = torch.argmin(gaps).expand(batch_size)
        ordered = torch.sort(gaps).values
        margin = (ordered[1] - ordered[0]).expand(batch_size) if len(gaps) > 1 else torch.full_like(selected.float(), float("inf"))
        return selected, scores, margin

    def checkpoint_payload(self) -> dict[str, object]:
        expected = {f"{name}/{stage}" for name in ROUTED_MODALITIES for stage in C2ST_STAGES}
        if set(self.calibrations) != expected:
            raise ValueError("cannot save an uncalibrated C2ST router")
        return {
            "router_type": "c2st_independent_v1",
            "modalities": ROUTED_MODALITIES,
            "stages": C2ST_STAGES,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "lambda_std": self.lambda_std,
            "trim_fraction": self.trim_fraction,
            "state_dict": self.state_dict(),
            "calibrations": {
                name: calibration.as_payload()
                for name, calibration in self.calibrations.items()
            },
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, object]) -> "C2STRouter":
        if payload.get("router_type") != "c2st_independent_v1" or tuple(payload.get("modalities", ())) != ROUTED_MODALITIES:
            raise ValueError("incompatible C2ST router checkpoint")
        if tuple(payload.get("stages", ())) != C2ST_STAGES:
            raise ValueError("incompatible C2ST stage schema")
        router = cls(
            int(payload["feature_dim"]),
            hidden_dim=int(payload.get("hidden_dim", 256)),
            lambda_std=float(payload.get("lambda_std", 0.25)),
            trim_fraction=float(payload.get("trim_fraction", 0.10)),
        )
        router.load_state_dict(payload["state_dict"])
        router.calibrations = {
            name: C2STCalibration.from_payload(values)
            for name, values in dict(payload["calibrations"]).items()
        }
        return router


class OnlineC2STTrainer:
    """Prequential domain-head updates with bounded target feature replay.

    Feature extraction remains outside this class so callers can enforce a
    frozen representation model. ``update`` must be called only after the
    current target batch has been routed.
    """

    def __init__(
        self,
        router: C2STRouter,
        *,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        replay_limit: int = 512,
        train_batch_size: int = 64,
        seed: int = 3247,
    ) -> None:
        if lr <= 0 or weight_decay < 0:
            raise ValueError("invalid online C2ST optimizer configuration")
        if replay_limit < 1 or train_batch_size < 1:
            raise ValueError("online replay and batch sizes must be positive")
        self.router = router
        self.replay_limit = int(replay_limit)
        self.train_batch_size = int(train_batch_size)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.target_replay: dict[str, torch.Tensor] = {}
        self.optimizers = {
            f"{name}/{stage}": torch.optim.AdamW(
                router.classifiers[name][stage_index].parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            for name in ROUTED_MODALITIES
            for stage_index, stage in enumerate(C2ST_STAGES)
        }

    @property
    def target_rows(self) -> int:
        if not self.target_replay:
            return 0
        return int(next(iter(self.target_replay.values())).shape[0])

    @staticmethod
    def _validate_features(features: Mapping[str, torch.Tensor]) -> None:
        if set(features) != set(ROUTED_MODALITIES):
            raise ValueError("online C2ST features must contain every routed modality")
        row_counts = set()
        for name in ROUTED_MODALITIES:
            value = features[name]
            if value.ndim != 3 or value.shape[1] != len(C2ST_STAGES):
                raise ValueError(f"online C2ST features for {name} have an invalid shape")
            row_counts.add(int(value.shape[0]))
        if len(row_counts) != 1 or not row_counts or next(iter(row_counts)) < 1:
            raise ValueError("online C2ST modalities must share a non-empty row count")

    def calibrate(self, source_features: Mapping[str, torch.Tensor]) -> None:
        """Refresh source-percentile calibration after domain-head updates."""
        self._validate_features(source_features)
        device = next(self.router.parameters()).device
        for name in ROUTED_MODALITIES:
            values = source_features[name]
            for stage_index, stage in enumerate(C2ST_STAGES):
                classifier = self.router.classifiers[name][stage_index]
                classifier.eval()
                with torch.no_grad():
                    logits = classifier(values[:, stage_index].to(device).float())
                self.router.calibrations[f"{name}/{stage}"] = fit_percentile_calibration(
                    logits.cpu()
                )

    def update(
        self,
        source_features: Mapping[str, torch.Tensor],
        target_features: Mapping[str, torch.Tensor],
        *,
        steps: int = 5,
    ) -> float:
        """Update all domain heads after routing the current target batch."""
        if steps < 1:
            raise ValueError("online C2ST steps must be positive")
        self._validate_features(source_features)
        self._validate_features(target_features)
        for name in ROUTED_MODALITIES:
            incoming = target_features[name].detach().float().cpu()
            replay = incoming if name not in self.target_replay else torch.cat(
                (self.target_replay[name], incoming), dim=0
            )
            self.target_replay[name] = replay[-self.replay_limit :]

        device = next(self.router.parameters()).device
        source_rows = int(next(iter(source_features.values())).shape[0])
        replay_rows = self.target_rows
        sample_rows = min(self.train_batch_size, source_rows, replay_rows)
        losses = []
        for _ in range(steps):
            source_index = torch.randint(
                source_rows, (sample_rows,), generator=self.generator
            )
            target_index = torch.randint(
                replay_rows, (sample_rows,), generator=self.generator
            )
            labels = torch.cat(
                (
                    torch.zeros(sample_rows, device=device),
                    torch.ones(sample_rows, device=device),
                )
            )
            for name in ROUTED_MODALITIES:
                for stage_index, stage in enumerate(C2ST_STAGES):
                    classifier = self.router.classifiers[name][stage_index]
                    optimizer = self.optimizers[f"{name}/{stage}"]
                    classifier.train()
                    source_batch = source_features[name].index_select(
                        0, source_index
                    )[:, stage_index].to(device).float()
                    target_batch = self.target_replay[name].index_select(
                        0, target_index
                    )[:, stage_index].to(device).float()
                    logits = classifier(torch.cat((source_batch, target_batch), dim=0))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    classifier.eval()
        return sum(losses) / len(losses)

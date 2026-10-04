"""Source-only calibration for the learned modality subset router.

The calibrator consumes source action utilities only.  It never needs target
SMILES or target-derived correctness and it does not introduce a hand-written
CNMR offset.  Calibration is represented as a temperature, a learned
15-action bias, a source-selected subset-size penalty, and an optional abstain
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from .modality_subset_router import ACTION_SUBSETS


ABSTAIN_ACTION = -1


@dataclass(frozen=True)
class RouterCalibration:
    temperature: float = 1.0
    action_bias: tuple[float, ...] = ()
    sparsity_lambda: float = 0.0
    abstain_threshold: float = 0.0
    abstain_utility: float = 0.0

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("calibration temperature must be positive")
        if self.action_bias and len(self.action_bias) != len(ACTION_SUBSETS):
            raise ValueError("action bias must contain one value per router action")
        if not 0 <= self.abstain_threshold <= 1:
            raise ValueError("abstain threshold must lie in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "temperature": float(self.temperature),
            "action_bias": [float(value) for value in self.action_bias],
            "sparsity_lambda": float(self.sparsity_lambda),
            "abstain_threshold": float(self.abstain_threshold),
            "abstain_utility": float(self.abstain_utility),
            "action_subsets": ACTION_SUBSETS,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RouterCalibration":
        observed = tuple(tuple(value) for value in payload.get("action_subsets", ()))
        if observed and observed != ACTION_SUBSETS:
            raise ValueError("calibration action order is incompatible")
        result = cls(
            temperature=float(payload.get("temperature", 1.0)),
            action_bias=tuple(float(value) for value in payload.get("action_bias", ())),
            sparsity_lambda=float(payload.get("sparsity_lambda", 0.0)),
            abstain_threshold=float(payload.get("abstain_threshold", 0.0)),
            abstain_utility=float(payload.get("abstain_utility", 0.0)),
        )
        result.validate()
        return result


def action_sizes(device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor([len(subset) for subset in ACTION_SUBSETS], dtype=torch.float32, device=device)


def valid_action_mask(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 2 or rewards.shape[1] != len(ACTION_SUBSETS):
        raise ValueError("rewards must have shape [rows, 15]")
    return torch.isfinite(rewards) & rewards.gt(-1e3)


def soft_oracle_targets(rewards: torch.Tensor, temperature: float = 0.20) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("oracle temperature must be positive")
    valid = valid_action_mask(rewards)
    if not valid.any(dim=1).all():
        raise ValueError("every source row needs at least one valid action")
    return torch.softmax((rewards / temperature).masked_fill(~valid, float("-inf")), dim=1)


def apply_calibration(
    logits: torch.Tensor,
    calibration: RouterCalibration | None = None,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != len(ACTION_SUBSETS):
        raise ValueError("router logits must have shape [rows, 15]")
    calibration = calibration or RouterCalibration()
    calibration.validate()
    bias = logits.new_zeros((len(ACTION_SUBSETS),))
    if calibration.action_bias:
        bias = logits.new_tensor(calibration.action_bias)
    return (
        logits / calibration.temperature
        + bias[None]
        - calibration.sparsity_lambda * action_sizes(logits.device)[None]
    )


def select_calibrated(
    logits: torch.Tensor,
    calibration: RouterCalibration | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    calibrated = apply_calibration(logits, calibration)
    probabilities = torch.softmax(calibrated, dim=1)
    confidence, selected = probabilities.max(dim=1)
    threshold = float((calibration or RouterCalibration()).abstain_threshold)
    if threshold > 0:
        selected = selected.masked_fill(confidence < threshold, ABSTAIN_ACTION)
    return selected, probabilities, confidence


def _masked_soft_cross_entropy(logits: torch.Tensor, rewards: torch.Tensor, temperature: float) -> torch.Tensor:
    targets = soft_oracle_targets(rewards, temperature)
    log_prob = torch.log_softmax(logits, dim=1)
    positive = targets.gt(0)
    terms = torch.zeros_like(targets)
    terms[positive] = targets[positive] * log_prob[positive]
    return -terms.sum(dim=1).mean()


def fit_temperature(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    indices: torch.Tensor,
    candidates: Iterable[float] | None = None,
    oracle_temperature: float = 0.20,
) -> float:
    """Fit scalar temperature against source utility-derived soft targets."""
    if candidates is None:
        candidates = torch.logspace(-1, 1, 81).tolist()
    candidates = list(float(value) for value in candidates)
    current_logits = logits.index_select(0, indices)
    current_rewards = rewards.index_select(0, indices)
    values = []
    for temperature in candidates:
        values.append(
            float(
                _masked_soft_cross_entropy(
                        current_logits / float(temperature), current_rewards, oracle_temperature
                )
            )
        )
    return candidates[int(torch.tensor(values).argmin())]


def fit_action_bias(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    indices: torch.Tensor,
    *,
    temperature: float = 1.0,
    oracle_temperature: float = 0.20,
    steps: int = 400,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> torch.Tensor:
    """Learn an action bias from source utility, initialized at zero."""
    if steps < 1 or lr <= 0 or l2 < 0:
        raise ValueError("invalid action-bias optimizer configuration")
    current_logits = logits.index_select(0, indices).float() / float(temperature)
    current_rewards = rewards.index_select(0, indices).float()
    bias = torch.zeros(len(ACTION_SUBSETS), requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=lr)
    for _ in range(steps):
        loss = _masked_soft_cross_entropy(
            current_logits + bias[None], current_rewards, oracle_temperature
        ) + l2 * bias.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return bias.detach()


def select_sparsity_lambda(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    indices: torch.Tensor,
    *,
    temperature: float = 1.0,
    bias: torch.Tensor | None = None,
    candidates: Iterable[float] | None = None,
) -> float:
    """Select lambda by source utility; no target correctness is consulted."""
    if candidates is None:
        candidates = torch.linspace(0, 2, 81).tolist()
    candidates = list(float(value) for value in candidates)
    base = logits.index_select(0, indices) / float(temperature)
    if bias is not None:
        base = base + bias.to(base.device)[None]
    current_rewards = rewards.index_select(0, indices)
    sizes = action_sizes(base.device)
    scores = []
    for value in candidates:
        selected = (base - float(value) * sizes[None]).argmax(dim=1)
        selected_reward = current_rewards.gather(1, selected[:, None]).squeeze(1)
        valid = selected_reward.gt(-1e3) & torch.isfinite(selected_reward)
        scores.append(float(selected_reward[valid].mean()) if valid.any() else -1e4)
    best = max(range(len(candidates)), key=lambda index: (scores[index], -candidates[index]))
    return candidates[best]


def select_abstain_threshold(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    indices: torch.Tensor,
    *,
    calibration: RouterCalibration,
    candidates: Iterable[float] | None = None,
) -> float:
    """Choose an abstention threshold using source utility and coverage."""
    if candidates is None:
        candidates = torch.linspace(0, 0.99, 100).tolist()
    candidates = list(float(value) for value in candidates)
    base = apply_calibration(logits.index_select(0, indices), calibration)
    probabilities = torch.softmax(base, dim=1)
    confidence, selected = probabilities.max(dim=1)
    current_rewards = rewards.index_select(0, indices)
    values = []
    for threshold in candidates:
        active = confidence >= threshold
        selected_reward = current_rewards.gather(1, selected[:, None]).squeeze(1)
        utility = torch.where(active, selected_reward, selected_reward.new_tensor(calibration.abstain_utility))
        values.append(float(utility.mean()))
    best = max(range(len(candidates)), key=lambda index: (values[index], -candidates[index]))
    return candidates[best]


def calibration_metrics(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    action_nll: torch.Tensor | None,
    indices: torch.Tensor,
    calibration: RouterCalibration | None = None,
    bins: int = 10,
) -> dict[str, object]:
    selected, probabilities, confidence = select_calibrated(
        logits.index_select(0, indices), calibration
    )
    current_rewards = rewards.index_select(0, indices)
    oracle = current_rewards.argmax(dim=1)
    active = selected.ne(ABSTAIN_ACTION)
    correct = torch.zeros_like(active)
    correct[active] = selected[active].eq(oracle[active])
    selected_reward = torch.zeros_like(confidence)
    selected_reward[active] = current_rewards[active].gather(1, selected[active, None]).squeeze(1)
    valid = active & selected_reward.gt(-1e3) & torch.isfinite(selected_reward)
    entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=1)
    calibration_error = torch.zeros((), dtype=torch.float32)
    for left in torch.linspace(0, 1, bins + 1)[:-1]:
        right = left + 1.0 / bins
        mask = confidence.ge(left) & (confidence.lt(right) if right < 1 else confidence.le(right))
        if mask.any():
            calibration_error += mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
    result: dict[str, object] = {
        "rows": int(len(indices)),
        "active_rows": int(active.sum()),
        "abstain_rate": float((~active).float().mean()),
        "action_accuracy": float(correct[active].float().mean()) if active.any() else 0.0,
        "oracle_route_agreement": float(correct[active].float().mean()) if active.any() else 0.0,
        "route_calibration_error": float(calibration_error),
        "selected_utility": float(selected_reward[valid].mean()) if valid.any() else -1e4,
        "pseudo_validity": float(valid.float().mean()),
        "action_entropy": float(entropy.mean()),
        "route_sparsity": float(
            torch.tensor([len(ACTION_SUBSETS[int(index)]) for index in selected[active].tolist()]).float().mean()
        ) if active.any() else 0.0,
        "selection_frequency": {
            "+".join(ACTION_SUBSETS[int(index)]): int((selected == index).sum())
            for index in range(len(ACTION_SUBSETS))
            if bool((selected == index).any())
        },
        "selected_probability": float(confidence[active].mean()) if active.any() else 0.0,
        "pseudo_nll": None,
        "formula_match": None,
    }
    if action_nll is not None and active.any():
        current_nll = action_nll.index_select(0, indices)
        result["pseudo_nll"] = float(current_nll[active].gather(1, selected[active, None]).mean())
    return result

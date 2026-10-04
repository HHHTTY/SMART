"""Molecular pseudo-reward and group-relative policy-gradient utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors


RDLogger.DisableLog("rdApp.*")
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*)")


def formula_signature(formula: str) -> tuple[tuple[str, int], ...]:
    text = str(formula).replace(" ", "")
    values: dict[str, int] = {}
    position = 0
    for match in _FORMULA_TOKEN.finditer(text):
        if match.start() != position:
            return ()
        name, count = match.groups()
        values[name] = values.get(name, 0) + (int(count) if count else 1)
        position = match.end()
    return tuple(sorted(values.items())) if position == len(text) and values else ()


def _molecule(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(str(smiles).replace(" ", ""))


def canonical_smiles(smiles: str) -> str | None:
    molecule = _molecule(smiles)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None


def molecular_formula(smiles: str) -> str | None:
    molecule = _molecule(smiles)
    return rdMolDescriptors.CalcMolFormula(molecule) if molecule is not None else None


@dataclass(frozen=True)
class MolecularRewardConfig:
    exact_weight: float = 1.0
    tanimoto_weight: float = 0.25
    formula_weight: float = 0.10
    invalid_penalty: float = 0.20


@dataclass(frozen=True)
class MolecularRewardBatch:
    total: torch.Tensor
    exact: torch.Tensor
    tanimoto: torch.Tensor
    formula_match: torch.Tensor
    invalid: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "exact": self.exact,
            "tanimoto": self.tanimoto,
            "formula_match": self.formula_match,
            "invalid": self.invalid,
        }


@dataclass(frozen=True)
class FingerprintDistanceTarget:
    probabilities: torch.Tensor
    distance: torch.Tensor
    tanimoto: torch.Tensor
    formula_match: torch.Tensor
    invalid: torch.Tensor


def molecular_pseudo_reward(
    candidates: Sequence[Sequence[str]],
    pseudo_smiles: Sequence[str],
    observed_formulas: Sequence[str],
    *,
    config: MolecularRewardConfig = MolecularRewardConfig(),
    device: torch.device | str | None = None,
) -> MolecularRewardBatch:
    """Score candidates against pseudo structures and observable Formula only."""
    if not (len(candidates) == len(pseudo_smiles) == len(observed_formulas)):
        raise ValueError("candidate, pseudo and Formula batches must have equal length")
    if not candidates:
        raise ValueError("reward requires a non-empty batch")
    width = len(candidates[0])
    if width < 1 or any(len(row) != width for row in candidates):
        raise ValueError("all candidate groups must have the same positive size")
    exact_rows: list[list[float]] = []
    tanimoto_rows: list[list[float]] = []
    formula_rows: list[list[float]] = []
    invalid_rows: list[list[float]] = []
    for rows, pseudo, observed in zip(candidates, pseudo_smiles, observed_formulas):
        pseudo_molecule = _molecule(pseudo)
        pseudo_canonical = (
            Chem.MolToSmiles(pseudo_molecule, canonical=True)
            if pseudo_molecule is not None
            else None
        )
        pseudo_fp = _MORGAN.GetFingerprint(pseudo_molecule) if pseudo_molecule is not None else None
        observed_signature = formula_signature(observed)
        exact_row = []
        tanimoto_row = []
        formula_row = []
        invalid_row = []
        for candidate in rows:
            molecule = _molecule(candidate)
            invalid = molecule is None
            invalid_row.append(float(invalid))
            if invalid:
                exact_row.append(0.0)
                tanimoto_row.append(0.0)
                formula_row.append(0.0)
                continue
            canonical = Chem.MolToSmiles(molecule, canonical=True)
            exact_row.append(float(pseudo_canonical is not None and canonical == pseudo_canonical))
            fingerprint = _MORGAN.GetFingerprint(molecule)
            tanimoto_row.append(
                float(DataStructs.TanimotoSimilarity(fingerprint, pseudo_fp))
                if pseudo_fp is not None
                else 0.0
            )
            predicted_formula = rdMolDescriptors.CalcMolFormula(molecule)
            formula_row.append(
                float(bool(observed_signature) and formula_signature(predicted_formula) == observed_signature)
            )
        exact_rows.append(exact_row)
        tanimoto_rows.append(tanimoto_row)
        formula_rows.append(formula_row)
        invalid_rows.append(invalid_row)
    exact = torch.tensor(exact_rows, dtype=torch.float32, device=device)
    tanimoto = torch.tensor(tanimoto_rows, dtype=torch.float32, device=device).clamp_(0.0, 1.0)
    formula = torch.tensor(formula_rows, dtype=torch.float32, device=device)
    invalid = torch.tensor(invalid_rows, dtype=torch.float32, device=device)
    total = (
        config.exact_weight * exact
        + config.tanimoto_weight * tanimoto
        + config.formula_weight * formula
        - config.invalid_penalty * invalid
    )
    return MolecularRewardBatch(total, exact, tanimoto, formula, invalid)


def fingerprint_distance_target(
    candidates: Sequence[Sequence[str]],
    pseudo_smiles: Sequence[str],
    observed_formulas: Sequence[str],
    *,
    temperature: float = 0.25,
    formula_penalty: float = 0.5,
    invalid_penalty: float = 1.0,
    device: torch.device | str | None = None,
) -> FingerprintDistanceTarget:
    """Build a detached chemical-distance distribution over fixed candidates."""
    if temperature <= 0:
        raise ValueError("fingerprint target temperature must be positive")
    if formula_penalty < 0 or invalid_penalty < 0:
        raise ValueError("chemical distance penalties must be non-negative")
    reward = molecular_pseudo_reward(
        candidates, pseudo_smiles, observed_formulas, device=device
    )
    distance = (
        1.0
        - reward.tanimoto
        + float(formula_penalty) * (1.0 - reward.formula_match)
        + float(invalid_penalty) * reward.invalid
    )
    probabilities = torch.softmax(-distance / float(temperature), dim=-1)
    return FingerprintDistanceTarget(
        probabilities=probabilities.detach(),
        distance=distance.detach(),
        tanimoto=reward.tanimoto.detach(),
        formula_match=reward.formula_match.detach(),
        invalid=reward.invalid.detach(),
    )


def fingerprint_listwise_loss(
    sequence_log_probs: torch.Tensor,
    target_probabilities: torch.Tensor,
    *,
    policy_temperature: float = 1.0,
) -> torch.Tensor:
    """Match candidate sequence probabilities to a chemical-distance target."""
    if sequence_log_probs.shape != target_probabilities.shape:
        raise ValueError("candidate log-probabilities and targets must have equal shape")
    if sequence_log_probs.ndim != 2:
        raise ValueError("candidate tensors must have shape [batch, candidates]")
    if policy_temperature <= 0:
        raise ValueError("policy temperature must be positive")
    log_policy = torch.log_softmax(
        sequence_log_probs.float() / float(policy_temperature), dim=-1
    )
    return -(target_probabilities.detach() * log_policy).sum(dim=-1).mean()


def group_relative_advantages(rewards: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    if rewards.ndim != 2:
        raise ValueError("group rewards must have shape [batch, candidates]")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    active = std > epsilon
    return torch.where(active, centered / std.clamp_min(epsilon), torch.zeros_like(centered))


def length_normalized_sequence_log_probs(
    logits: torch.Tensor,
    next_tokens: torch.Tensor,
    *,
    pad_token_id: int,
    eos_token_id: int | None,
) -> torch.Tensor:
    """Teacher-forced log probability, including the first EOS and no later token."""
    if logits.shape[:-1] != next_tokens.shape:
        raise ValueError("logits and next-token shapes do not match")
    token_mask = next_tokens.ne(int(pad_token_id))
    if eos_token_id is not None:
        eos = next_tokens.eq(int(eos_token_id))
        token_mask &= (eos.cumsum(dim=-1) - eos.long()).eq(0)
    token_log_prob = F.log_softmax(logits.float(), dim=-1).gather(
        -1, next_tokens.unsqueeze(-1)
    ).squeeze(-1)
    weights = token_mask.to(token_log_prob.dtype)
    return (token_log_prob * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def group_relative_policy_gradient_loss(
    sequence_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if sequence_log_probs.shape != advantages.shape or sequence_log_probs.ndim != 2:
        raise ValueError("log probabilities and advantages must have shape [batch, candidates]")
    if sample_weights.shape != sequence_log_probs.shape[:1]:
        raise ValueError("sample weights must have shape [batch]")
    denominator = sample_weights.sum().clamp_min(1e-8)
    per_sample = -(advantages.detach() * sequence_log_probs).mean(dim=1)
    return (sample_weights.detach() * per_sample).sum() / denominator

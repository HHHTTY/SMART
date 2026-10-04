"""Candidate-level multimodal compatibility scoring.

This module intentionally contains no model or chemistry dependencies.  It
combines already-computed candidate compatibility scores, handles unavailable
or uninformative endpoints, and provides paired rank diagnostics used by the
candidate-energy evaluation runner.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


TASK_NAMES = ("MSMS", "NMR", "IR")


def informative_standardize(
    values: Sequence[float], eps: float = 1e-8
) -> tuple[np.ndarray, bool]:
    """Z-score finite values and report whether the endpoint is informative."""
    array = np.asarray(values, dtype=np.float64)
    output = np.full(array.shape, -np.inf, dtype=np.float64)
    finite = np.isfinite(array)
    if int(finite.sum()) < 2:
        output[finite] = 0.0
        return output, False
    selected = array[finite]
    deviation = float(selected.std())
    if deviation <= eps:
        output[finite] = 0.0
        return output, False
    output[finite] = (selected - float(selected.mean())) / deviation
    return output, True


def _weighted_average(
    arrays: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    if len(arrays) != len(weights) or not arrays:
        raise ValueError("arrays and weights must have the same non-zero length.")
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = np.clip(normalized, 0.0, None)
    if not np.isfinite(normalized).all() or float(normalized.sum()) <= 0.0:
        normalized = np.ones(len(arrays), dtype=np.float64)
    normalized = normalized / float(normalized.sum())
    stacked = np.stack(arrays, axis=0)
    finite = np.isfinite(stacked)
    effective_weights = normalized[:, None] * finite
    denominator = effective_weights.sum(axis=0)
    output = np.full(stacked.shape[1], -np.inf, dtype=np.float64)
    valid = denominator > 0.0
    safe_values = np.where(finite, stacked, 0.0)
    output[valid] = (
        (effective_weights[:, valid] * safe_values[:, valid]).sum(axis=0)
        / denominator[valid]
    )
    return output


def structural_compatibility(
    components: Mapping[str, Sequence[float]],
    availability: Mapping[str, bool],
    gate_weights: Mapping[str, float],
    *,
    include_fingerprint: bool = True,
    task_names: Sequence[str] = TASK_NAMES,
    auxiliary_weighting: str = "gate",
) -> tuple[np.ndarray, bool, tuple[str, ...]]:
    """Combine fingerprint and auxiliary task compatibility scores.

    Every endpoint is standardized within a query's candidate set before it is
    combined.  Missing and constant endpoints are omitted.  The returned score
    is standardized once more so that its scale is comparable across modality
    availability patterns.
    """
    if auxiliary_weighting not in {"gate", "equal"}:
        raise ValueError("auxiliary_weighting must be 'gate' or 'equal'.")
    if not components:
        raise ValueError("At least one component array is required.")
    size = len(next(iter(components.values())))
    active_names: list[str] = []
    structure_parts: list[np.ndarray] = []

    if include_fingerprint and "fingerprint" in components:
        fingerprint, informative = informative_standardize(components["fingerprint"])
        if informative:
            structure_parts.append(fingerprint)
            active_names.append("fingerprint")

    auxiliary_parts: list[np.ndarray] = []
    auxiliary_weights: list[float] = []
    auxiliary_names: list[str] = []
    for name in task_names:
        if not availability.get(name, False) or name not in components:
            continue
        standardized, informative = informative_standardize(components[name])
        if not informative:
            continue
        auxiliary_parts.append(standardized)
        auxiliary_names.append(name)
        auxiliary_weights.append(
            float(gate_weights.get(name, 0.0))
            if auxiliary_weighting == "gate"
            else 1.0
        )
    if auxiliary_parts:
        structure_parts.append(_weighted_average(auxiliary_parts, auxiliary_weights))
        active_names.extend(auxiliary_names)

    if not structure_parts:
        return np.zeros(size, dtype=np.float64), False, ()
    structure = (
        structure_parts[0]
        if len(structure_parts) == 1
        else _weighted_average(structure_parts, [1.0] * len(structure_parts))
    )
    standardized, informative = informative_standardize(structure)
    if not informative:
        return np.zeros(size, dtype=np.float64), False, ()
    return standardized, True, tuple(active_names)


def candidate_compatibility_scores(
    sequence_scores: Sequence[float],
    components: Mapping[str, Sequence[float]],
    availability: Mapping[str, bool],
    gate_weights: Mapping[str, float],
    *,
    alpha: float,
    include_fingerprint: bool = True,
    task_names: Sequence[str] = TASK_NAMES,
    auxiliary_weighting: str = "gate",
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a higher-is-better sequence/structure compatibility score.

    If an energy convention is needed downstream, define energy as the negative
    of this score.  Keeping one higher-is-better convention here avoids mixing a
    log-probability with a distance under inconsistent signs.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    sequence, sequence_informative = informative_standardize(sequence_scores)
    structure, structure_informative, active = structural_compatibility(
        components,
        availability,
        gate_weights,
        include_fingerprint=include_fingerprint,
        task_names=task_names,
        auxiliary_weighting=auxiliary_weighting,
    )
    if sequence.shape != structure.shape:
        raise ValueError("Sequence and structural scores must have equal length.")

    if not structure_informative or alpha >= 1.0:
        total = sequence
        effective_alpha = 1.0
    elif not sequence_informative or alpha <= 0.0:
        total = structure
        effective_alpha = 0.0
    else:
        total = np.full(sequence.shape, -np.inf, dtype=np.float64)
        finite = np.isfinite(sequence) & np.isfinite(structure)
        total[finite] = alpha * sequence[finite] + (1.0 - alpha) * structure[finite]
        effective_alpha = alpha
    return total, {
        "requested_alpha": alpha,
        "effective_alpha": effective_alpha,
        "sequence_informative": sequence_informative,
        "structure_informative": structure_informative,
        "active_structure_endpoints": active,
    }


def stable_descending_order(scores: Sequence[float]) -> np.ndarray:
    """Sort high-to-low while preserving the original beam order on ties."""
    values = np.asarray(scores, dtype=np.float64)
    return np.argsort(-values, kind="stable")


def rank_metrics(ranks: Sequence[int | None]) -> dict[str, float | int]:
    """Aggregate exact rank metrics; None means target absent from the pool."""
    n = len(ranks)
    reciprocal = [0.0 if rank is None else 1.0 / rank for rank in ranks]
    result: dict[str, float | int] = {
        "n": n,
        "oracle_in_pool_count": sum(rank is not None for rank in ranks),
        "oracle_in_pool": sum(rank is not None for rank in ranks) / max(n, 1),
        "mrr": float(np.mean(reciprocal)) if reciprocal else 0.0,
    }
    for cutoff in (1, 5, 10):
        hits = sum(rank is not None and rank <= cutoff for rank in ranks)
        result[f"top{cutoff}_count"] = hits
        result[f"top{cutoff}"] = hits / max(n, 1)
    return result


def paired_rank_metrics(
    baseline_ranks: Sequence[int | None], candidate_ranks: Sequence[int | None]
) -> dict[str, object]:
    """Compare two rankings over an identical candidate pool."""
    if len(baseline_ranks) != len(candidate_ranks):
        raise ValueError("Paired rank arrays must have equal length.")
    cutoffs: dict[str, dict[str, int]] = {}
    for cutoff in (1, 5, 10):
        wins = losses = stable_hit = stable_miss = 0
        for before, after in zip(baseline_ranks, candidate_ranks):
            before_hit = before is not None and before <= cutoff
            after_hit = after is not None and after <= cutoff
            if after_hit and not before_hit:
                wins += 1
            elif before_hit and not after_hit:
                losses += 1
            elif before_hit:
                stable_hit += 1
            else:
                stable_miss += 1
        cutoffs[f"top{cutoff}"] = {
            "wins": wins,
            "losses": losses,
            "net": wins - losses,
            "stable_hit": stable_hit,
            "stable_miss": stable_miss,
        }

    improved = worsened = tied = 0
    reciprocal_deltas = []
    oracle_reciprocal_deltas = []
    for before, after in zip(baseline_ranks, candidate_ranks):
        before_rr = 0.0 if before is None else 1.0 / before
        after_rr = 0.0 if after is None else 1.0 / after
        reciprocal_deltas.append(after_rr - before_rr)
        if before is None or after is None:
            continue
        oracle_reciprocal_deltas.append(after_rr - before_rr)
        if after < before:
            improved += 1
        elif after > before:
            worsened += 1
        else:
            tied += 1
    return {
        "cutoffs": cutoffs,
        "oracle_rank_direction": {
            "improved": improved,
            "worsened": worsened,
            "tied": tied,
        },
        "mean_reciprocal_rank_delta": float(np.mean(reciprocal_deltas))
        if reciprocal_deltas
        else 0.0,
        "oracle_mean_reciprocal_rank_delta": float(
            np.mean(oracle_reciprocal_deltas)
        )
        if oracle_reciprocal_deltas
        else 0.0,
    }


def paired_bootstrap_intervals(
    baseline_ranks: Sequence[int | None],
    candidate_ranks: Sequence[int | None],
    *,
    seed: int = 3247,
    n_resamples: int = 10_000,
) -> dict[str, object]:
    """Percentile intervals for paired metric deltas over query resamples."""
    if len(baseline_ranks) != len(candidate_ranks):
        raise ValueError("Paired rank arrays must have equal length.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    n = len(baseline_ranks)
    if n == 0:
        return {"seed": seed, "n_resamples": n_resamples, "deltas": {}}

    before = np.asarray(
        [0.0 if rank is None else 1.0 / rank for rank in baseline_ranks],
        dtype=np.float64,
    )
    after = np.asarray(
        [0.0 if rank is None else 1.0 / rank for rank in candidate_ranks],
        dtype=np.float64,
    )
    per_query: dict[str, np.ndarray] = {"mrr": after - before}
    for cutoff in (1, 5, 10):
        before_hits = np.asarray(
            [rank is not None and rank <= cutoff for rank in baseline_ranks],
            dtype=np.float64,
        )
        after_hits = np.asarray(
            [rank is not None and rank <= cutoff for rank in candidate_ranks],
            dtype=np.float64,
        )
        per_query[f"top{cutoff}"] = after_hits - before_hits

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, n, size=(n_resamples, n))
    deltas: dict[str, dict[str, float]] = {}
    for name, values in per_query.items():
        bootstrap = values[sampled_indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        deltas[name] = {
            "point": float(values.mean()),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
        }
    return {"seed": seed, "n_resamples": n_resamples, "deltas": deltas}


def select_conservative_alpha(
    ranks_by_alpha: Mapping[float, Sequence[int | None]],
) -> tuple[float, dict[str, object]]:
    """Select a validation-only blend or fall back to sequence-only alpha=1."""
    if 1.0 not in ranks_by_alpha:
        raise ValueError("The alpha grid must include the sequence-only alpha=1.0.")
    baseline_ranks = ranks_by_alpha[1.0]
    baseline = rank_metrics(baseline_ranks)
    grid: dict[str, object] = {}
    admissible: list[float] = []
    for alpha in sorted(ranks_by_alpha):
        ranks = ranks_by_alpha[alpha]
        metrics = rank_metrics(ranks)
        paired = paired_rank_metrics(baseline_ranks, ranks)
        grid[str(alpha)] = {"metrics": metrics, "paired_vs_sequence": paired}
        cutoff_values = paired["cutoffs"]
        if all(
            cutoff_values[f"top{cutoff}"]["wins"]
            >= cutoff_values[f"top{cutoff}"]["losses"]
            for cutoff in (1, 5, 10)
        ):
            admissible.append(alpha)

    improving = [
        alpha
        for alpha in admissible
        if rank_metrics(ranks_by_alpha[alpha])["mrr"] > baseline["mrr"] + 1e-12
    ]
    selected = 1.0
    if improving:
        selected = max(
            improving,
            key=lambda alpha: (
                rank_metrics(ranks_by_alpha[alpha])["mrr"],
                rank_metrics(ranks_by_alpha[alpha])["top1_count"],
                rank_metrics(ranks_by_alpha[alpha])["top5_count"],
                rank_metrics(ranks_by_alpha[alpha])["top10_count"],
                alpha,
            ),
        )
    return selected, {
        "selected_alpha": selected,
        "fallback_to_sequence": selected == 1.0,
        "selection_rule": (
            "validation only; require paired wins>=losses at Top-1/5/10 and "
            "strict MRR improvement; then maximize MRR, Top-1/5/10, alpha"
        ),
        "grid": grid,
    }

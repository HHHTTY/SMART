"""Formula-conditioned source sampling for sim-to-real domain alignment."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np


_FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)([0-9]*)")
_ATOMIC_WEIGHTS = {
    "H": 1.00794,
    "B": 10.811,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984,
    "Si": 28.0855,
    "P": 30.9738,
    "S": 32.065,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.9045,
}
_HALOGENS = frozenset({"F", "Cl", "Br", "I"})
FORMULA_CONDITION_ELEMENTS = (
    "C",
    "H",
    "B",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
)
FORMULA_CONDITION_DIM = 2 * len(FORMULA_CONDITION_ELEMENTS) + 4


@dataclass(frozen=True)
class FormulaDescriptor:
    canonical: str
    molecular_weight: float
    dbe: float
    element_profile: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ChemistryMatch:
    source_index: int
    target_formula: str
    source_formula: str
    route: str
    molecular_weight_delta: Optional[float]
    dbe_delta: Optional[float]
    element_profile_match: bool


def formula_descriptor(formula: object) -> Optional[FormulaDescriptor]:
    """Parse a neutral molecular formula into matching attributes.

    The fallback profile records the presence of O, N, S, P, halogens, and
    uncommon elements. Exact matching still uses complete element counts.
    """

    text = str(formula).strip()
    matches = list(_FORMULA_PATTERN.finditer(text))
    if not matches or "".join(match.group(0) for match in matches) != text:
        return None
    composition: dict[str, int] = defaultdict(int)
    for match in matches:
        count = int(match.group(2) or 1)
        if count <= 0:
            return None
        composition[match.group(1)] += count
    if any(element not in _ATOMIC_WEIGHTS for element in composition):
        return None

    ordered = []
    if "C" in composition:
        ordered.append("C")
    if "H" in composition:
        ordered.append("H")
    ordered.extend(sorted(set(composition).difference(ordered)))
    canonical = "".join(
        element + (str(composition[element]) if composition[element] != 1 else "")
        for element in ordered
    )
    molecular_weight = sum(
        _ATOMIC_WEIGHTS[element] * count for element, count in composition.items()
    )
    carbon = composition.get("C", 0)
    hydrogen = composition.get("H", 0)
    nitrogen_like = composition.get("N", 0) + composition.get("P", 0)
    halogens = sum(composition.get(element, 0) for element in _HALOGENS)
    dbe = 1.0 + carbon + 0.5 * (nitrogen_like - hydrogen - halogens)
    known_organic = {"C", "H", "O", "N", "S", "P", *_HALOGENS}
    profile = (
        int(composition.get("O", 0) > 0),
        int(composition.get("N", 0) > 0),
        int(composition.get("S", 0) > 0),
        int(composition.get("P", 0) > 0),
        int(any(composition.get(element, 0) > 0 for element in _HALOGENS)),
        int(bool(set(composition).difference(known_organic))),
    )
    return FormulaDescriptor(canonical, molecular_weight, dbe, profile)


def formula_condition_vector(formula: object) -> tuple[float, ...]:
    """Encode observable Formula chemistry for a conditional domain boundary.

    The vector contains log-scaled element counts, atom fractions, molecular
    weight, DBE, total atom count, and a validity flag.  It never uses a
    structure label or target SMILES.
    """

    text = str(formula).strip()
    matches = list(_FORMULA_PATTERN.finditer(text))
    invalid = not matches or "".join(match.group(0) for match in matches) != text
    composition: dict[str, int] = defaultdict(int)
    if not invalid:
        for match in matches:
            count = int(match.group(2) or 1)
            if count <= 0 or match.group(1) not in _ATOMIC_WEIGHTS:
                invalid = True
                break
            composition[match.group(1)] += count
    if invalid:
        return (0.0,) * FORMULA_CONDITION_DIM

    total_atoms = sum(composition.values())
    counts = [float(composition.get(element, 0)) for element in FORMULA_CONDITION_ELEMENTS]
    log_counts = [math.log1p(count) / math.log(65.0) for count in counts]
    fractions = [count / max(1.0, float(total_atoms)) for count in counts]
    molecular_weight = sum(
        _ATOMIC_WEIGHTS[element] * count for element, count in composition.items()
    )
    carbon = composition.get("C", 0)
    hydrogen = composition.get("H", 0)
    nitrogen_like = composition.get("N", 0) + composition.get("P", 0)
    halogens = sum(composition.get(element, 0) for element in _HALOGENS)
    dbe = 1.0 + carbon + 0.5 * (nitrogen_like - hydrogen - halogens)
    global_features = (
        math.log1p(molecular_weight) / math.log(1001.0),
        math.tanh(dbe / 20.0),
        math.log1p(total_atoms) / math.log(129.0),
        1.0,
    )
    return tuple((*log_counts, *fractions, *global_features))


def _stratum(
    descriptor: FormulaDescriptor, mw_bin_width: float, dbe_bin_width: float
) -> tuple[tuple[int, ...], int, int]:
    return (
        descriptor.element_profile,
        int(math.floor(descriptor.molecular_weight / mw_bin_width)),
        int(math.floor(descriptor.dbe / dbe_bin_width)),
    )


def _stratum_distance(
    left: tuple[tuple[int, ...], int, int],
    right: tuple[tuple[int, ...], int, int],
) -> float:
    profile_distance = sum(a != b for a, b in zip(left[0], right[0]))
    return 8.0 * profile_distance + abs(left[1] - right[1]) + 2.0 * abs(left[2] - right[2])


def chemistry_matched_order(
    source_formulas: Sequence[object],
    target_formulas: Sequence[object],
    count: int,
    *,
    seed: int,
    mw_bin_width: float = 25.0,
    dbe_bin_width: float = 2.0,
) -> tuple[list[ChemistryMatch], dict[str, object]]:
    """Return a deterministic, without-replacement chemistry-matched order."""

    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(source_formulas):
        raise ValueError("count exceeds the source population")
    if len(target_formulas) == 0:
        raise ValueError("target_formulas cannot be empty")
    if mw_bin_width <= 0 or dbe_bin_width <= 0:
        raise ValueError("chemistry bin widths must be positive")

    rng = np.random.default_rng(seed)
    descriptor_cache: dict[str, Optional[FormulaDescriptor]] = {}

    def describe(value: object) -> Optional[FormulaDescriptor]:
        key = str(value).strip()
        if key not in descriptor_cache:
            descriptor_cache[key] = formula_descriptor(key)
        return descriptor_cache[key]

    source_descriptors = [describe(value) for value in source_formulas]
    target_descriptors = [describe(value) for value in target_formulas]
    exact_buckets: dict[str, list[int]] = defaultdict(list)
    stratum_buckets: dict[tuple[tuple[int, ...], int, int], list[int]] = defaultdict(list)
    for index, descriptor in enumerate(source_descriptors):
        if descriptor is None:
            continue
        exact_buckets[descriptor.canonical].append(index)
        stratum_buckets[_stratum(descriptor, mw_bin_width, dbe_bin_width)].append(index)
    initial_exact_formulas = frozenset(exact_buckets)
    for bucket in (*exact_buckets.values(), *stratum_buckets.values()):
        rng.shuffle(bucket)
    global_order = rng.permutation(len(source_formulas)).tolist()
    available_strata = tuple(stratum_buckets)
    nearest_cache: dict[
        tuple[tuple[int, ...], int, int], tuple[tuple[tuple[int, ...], int, int], ...]
    ] = {}
    used: set[int] = set()

    def pop_unused(bucket: list[int]) -> Optional[int]:
        while bucket:
            candidate = int(bucket.pop())
            if candidate not in used:
                return candidate
        return None

    def pop_global() -> int:
        while global_order:
            candidate = int(global_order.pop())
            if candidate not in used:
                return candidate
        raise RuntimeError("source population was exhausted")

    target_schedule: list[int] = []
    while len(target_schedule) < count:
        target_schedule.extend(rng.permutation(len(target_formulas)).tolist())
    target_schedule = target_schedule[:count]

    matches: list[ChemistryMatch] = []
    route_counts: Counter[str] = Counter()
    for target_index in target_schedule:
        target_formula = str(target_formulas[target_index]).strip()
        target_descriptor = target_descriptors[target_index]
        source_index: Optional[int] = None
        route = "global"
        if target_descriptor is not None:
            source_index = pop_unused(exact_buckets[target_descriptor.canonical])
            if source_index is not None:
                route = "exact_formula"
            else:
                target_stratum = _stratum(
                    target_descriptor, mw_bin_width, dbe_bin_width
                )
                source_index = pop_unused(stratum_buckets[target_stratum])
                if source_index is not None:
                    route = "same_stratum"
                else:
                    if target_stratum not in nearest_cache:
                        nearest_cache[target_stratum] = tuple(
                            sorted(
                                available_strata,
                                key=lambda item: _stratum_distance(target_stratum, item),
                            )
                        )
                    for candidate_stratum in nearest_cache[target_stratum]:
                        source_index = pop_unused(stratum_buckets[candidate_stratum])
                        if source_index is not None:
                            route = "nearest_stratum"
                            break
        if source_index is None:
            source_index = pop_global()
        used.add(source_index)
        source_formula = str(source_formulas[source_index]).strip()
        source_descriptor = source_descriptors[source_index]
        if target_descriptor is None or source_descriptor is None:
            mw_delta = None
            dbe_delta = None
            profile_match = False
        else:
            mw_delta = abs(
                source_descriptor.molecular_weight - target_descriptor.molecular_weight
            )
            dbe_delta = abs(source_descriptor.dbe - target_descriptor.dbe)
            profile_match = (
                source_descriptor.element_profile == target_descriptor.element_profile
            )
        matches.append(
            ChemistryMatch(
                source_index=source_index,
                target_formula=target_formula,
                source_formula=source_formula,
                route=route,
                molecular_weight_delta=mw_delta,
                dbe_delta=dbe_delta,
                element_profile_match=profile_match,
            )
        )
        route_counts[route] += 1

    valid_target = [item for item in target_descriptors if item is not None]
    exact_covered = sum(
        item is not None and item.canonical in initial_exact_formulas
        for item in target_descriptors
    )
    mw_deltas = [item.molecular_weight_delta for item in matches if item.molecular_weight_delta is not None]
    dbe_deltas = [item.dbe_delta for item in matches if item.dbe_delta is not None]
    audit: dict[str, object] = {
        "seed": seed,
        "requested": count,
        "source_rows": len(source_formulas),
        "target_rows": len(target_formulas),
        "valid_source_formula_rows": sum(item is not None for item in source_descriptors),
        "valid_target_formula_rows": len(valid_target),
        "target_rows_with_initial_exact_source": exact_covered,
        "target_exact_source_coverage": exact_covered / max(1, len(target_formulas)),
        "mw_bin_width": mw_bin_width,
        "dbe_bin_width": dbe_bin_width,
        "selection_routes": dict(route_counts),
        "mean_absolute_molecular_weight_delta": float(np.mean(mw_deltas)) if mw_deltas else None,
        "mean_absolute_dbe_delta": float(np.mean(dbe_deltas)) if dbe_deltas else None,
        "element_profile_match_fraction": sum(item.element_profile_match for item in matches)
        / max(1, len(matches)),
        "first_matches": [asdict(item) for item in matches[:10]],
    }
    return matches, audit

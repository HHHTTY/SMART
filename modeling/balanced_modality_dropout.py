"""Independent block-balanced modality-subset scheduling for continual TTT."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping


ROUTED_MODALITIES = ("HNMR", "CNMR", "MSMS", "IR")
PROPER_NONEMPTY_SUBSETS = (
    ("HNMR",),
    ("CNMR",),
    ("MSMS",),
    ("IR",),
    ("HNMR", "CNMR"),
    ("HNMR", "MSMS"),
    ("HNMR", "IR"),
    ("CNMR", "MSMS"),
    ("CNMR", "IR"),
    ("MSMS", "IR"),
    ("HNMR", "CNMR", "MSMS"),
    ("HNMR", "CNMR", "IR"),
    ("HNMR", "MSMS", "IR"),
    ("CNMR", "MSMS", "IR"),
)


@dataclass(frozen=True)
class ScheduledSubset:
    block_index: int
    index_in_block: int
    subset: tuple[str, ...]


class BlockBalancedModalityScheduler:
    """Shuffle every proper subset once per block and advance only on commit."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._random = random.Random(self.seed)
        self._consumed = 0
        self._block = self._new_block()

    def _new_block(self) -> list[tuple[str, ...]]:
        block = list(PROPER_NONEMPTY_SUBSETS)
        self._random.shuffle(block)
        return block

    @property
    def consumed(self) -> int:
        return self._consumed

    def peek(self) -> ScheduledSubset:
        index = self._consumed % len(PROPER_NONEMPTY_SUBSETS)
        return ScheduledSubset(
            block_index=self._consumed // len(PROPER_NONEMPTY_SUBSETS),
            index_in_block=index,
            subset=self._block[index],
        )

    def consume(self) -> ScheduledSubset:
        scheduled = self.peek()
        self._consumed += 1
        if self._consumed % len(PROPER_NONEMPTY_SUBSETS) == 0:
            self._block = self._new_block()
        return scheduled

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "consumed": self._consumed,
            "block": tuple(self._block),
            "random_state": self._random.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("scheduler seed differs from resume state")
        block = [tuple(str(name) for name in subset) for subset in state["block"]]
        if sorted(block) != sorted(PROPER_NONEMPTY_SUBSETS):
            raise ValueError("resume scheduler block is not a proper-subset permutation")
        self._consumed = int(state["consumed"])
        self._block = block
        self._random.setstate(state["random_state"])


def excluded_modalities_for_subset(
    subset: tuple[str, ...],
    *,
    input_modalities: tuple[str, ...] | list[str],
) -> frozenset[str]:
    """Drop complete modality spans while retaining Formula and the subset."""
    keep = frozenset(("Formula", *subset))
    unknown = set(subset).difference(ROUTED_MODALITIES)
    if unknown:
        raise ValueError(f"unknown dropout modalities: {sorted(unknown)}")
    return frozenset(set(input_modalities).difference(keep))

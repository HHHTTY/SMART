"""Utilities for the Anchor-Conditioned Adaptive Evidence Bottleneck (AEB).

The generator remains frozen.  This module only implements deterministic MS
evidence selection and the small source-trained compatibility scorer used by
the AEB experiment.  Selection operates on the collator's sequence-first
representation, before :class:`HFWrapper` applies the shared encoder.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def stable_sample_seed(sample_id: int, global_seed: int) -> int:
    """Return a reproducible 32-bit seed without Python hash randomisation."""

    payload = f"{int(global_seed)}:{int(sample_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _selection_positions(valid_length: int, budget: int, mode: str, sample_id: int, seed: int) -> list[int]:
    """Select interior MS evidence positions while protecting BOS/EOS.

    ``MSMSTextPreprocessor`` adds BOS and EOS around the spectrum.  The two
    boundary tokens are retained whenever a non-empty sample has them; the
    budget counts only interior evidence tokens.  Padding is never selected as
    evidence.  For malformed/empty rows this safely returns an empty list.
    """

    if valid_length <= 0 or budget <= 0:
        return []
    if valid_length <= 2:
        return []
    evidence = np.arange(1, valid_length - 1, dtype=np.int64)
    count = min(int(budget), int(evidence.size))
    if count == evidence.size:
        return evidence.tolist()
    mode = str(mode).lower()
    if mode == "prefix":
        chosen = evidence[:count]
    elif mode == "uniform":
        chosen = evidence[np.linspace(0, evidence.size - 1, count, dtype=np.int64)]
        # linspace can repeat for very small arrays; preserve the requested
        # count and deterministic order in that corner case.
        chosen = np.unique(chosen)
        if chosen.size < count:
            chosen = evidence[:count]
    elif mode == "random":
        rng = np.random.default_rng(stable_sample_seed(sample_id, seed))
        chosen = np.sort(rng.choice(evidence, size=count, replace=False))
    else:
        raise ValueError(f"Unknown AEB selection mode: {mode!r}")
    return [int(value) for value in chosen.tolist()]


def build_selection_indices(
    valid_mask: torch.Tensor,
    *,
    budgets: Sequence[int],
    mode: str,
    sample_ids: Sequence[int],
    global_seed: int = 3247,
    focus_scores: torch.Tensor | None = None,
    top_r: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a rectangular ``[max_budget, batch]`` gather index and mask.

    The returned mask follows this project's convention: ``True`` means pad / 
    ignored by attention.  Boundary BOS/EOS are always copied to the first and
    last retained positions.  If ``focus_scores`` is supplied, the highest
    scoring interior positions form the focus set and remaining capacity is
    filled by deterministic random background positions.
    """

    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be [sequence, batch]")
    seq_len, batch_size = valid_mask.shape
    if len(budgets) != batch_size or len(sample_ids) != batch_size:
        raise ValueError("budgets and sample_ids must match batch size")
    if focus_scores is not None and focus_scores.shape != (batch_size, seq_len):
        raise ValueError("focus_scores must be [batch, sequence]")
    if focus_scores is not None:
        # Selection indices are created on the collated mask's device (usually
        # CPU).  Accept scorer outputs produced on GPU and align them here so
        # token ranking never triggers a cross-device indexing error.
        focus_scores = focus_scores.to(device=valid_mask.device)
    max_budget = max([max(0, int(value)) for value in budgets] + [0])
    # +2 protects BOS/EOS; the actual valid rows are marked in output_mask.
    out_len = max_budget + 2
    indices = torch.zeros((out_len, batch_size), dtype=torch.long, device=valid_mask.device)
    output_mask = torch.ones((out_len, batch_size), dtype=torch.bool, device=valid_mask.device)
    valid_mask = valid_mask.to(dtype=torch.bool)
    for column in range(batch_size):
        valid_positions = torch.nonzero(valid_mask[:, column], as_tuple=False).flatten().tolist()
        if not valid_positions:
            continue
        # The text preprocessor uses BOS at the first valid position and EOS
        # at the last valid position.  If a malformed row has one token, keep
        # it as a boundary rather than deleting the entire modality.
        bos = int(valid_positions[0])
        eos = int(valid_positions[-1]) if len(valid_positions) > 1 else bos
        interior = valid_positions[1:-1] if len(valid_positions) > 2 else []
        budget = min(max(0, int(budgets[column])), len(interior))
        if focus_scores is None:
            chosen = _selection_positions(
                len(valid_positions), budget, mode, int(sample_ids[column]), global_seed
            )
            # _selection_positions uses local valid-token positions.  Convert
            # them to the original sequence positions for right-padded rows.
            chosen = [int(valid_positions[position]) for position in chosen]
        else:
            if budget <= 0:
                chosen = []
            else:
                score_row = focus_scores[column]
                interior_tensor = torch.tensor(interior, device=score_row.device, dtype=torch.long)
                ranked = interior_tensor[torch.argsort(score_row[interior_tensor], descending=True)]
                focus_count = min(int(top_r), budget, len(interior))
                focus = [int(value) for value in ranked[:focus_count].tolist()]
                remaining = [value for value in interior if value not in set(focus)]
                need = budget - len(focus)
                if need > 0 and remaining:
                    rng = np.random.default_rng(stable_sample_seed(int(sample_ids[column]), global_seed))
                    picked = rng.choice(np.asarray(remaining), size=min(need, len(remaining)), replace=False)
                    focus.extend(int(value) for value in np.sort(picked).tolist())
                chosen = sorted(focus)
        selected = [bos, *chosen, eos]
        length = len(selected)
        indices[:length, column] = torch.tensor(selected, dtype=torch.long, device=indices.device)
        output_mask[:length, column] = False
    return indices, output_mask


def _gather_sequence_first(value: Any, indices: torch.Tensor) -> Any:
    """Gather a collator value along its sequence-first dimension."""

    if isinstance(value, torch.Tensor):
        if value.ndim < 2:
            raise ValueError("Sequence-first modality tensors need [sequence, batch, ...]")
        if value.shape[1] != indices.shape[1]:
            raise ValueError("Modality batch dimension does not match selection")
        shape = [indices.shape[0], indices.shape[1]] + [1] * (value.ndim - 2)
        expanded = indices.reshape(shape).expand([indices.shape[0], indices.shape[1], *value.shape[2:]])
        return torch.gather(value, dim=0, index=expanded)
    if isinstance(value, Mapping):
        return {key: _gather_sequence_first(item, indices) for key, item in value.items()}
    raise TypeError(f"Unsupported tokenized modality value: {type(value).__name__}")


def clone_nested(value: Any) -> Any:
    """Clone tensors in a batch while preserving metadata containers."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(clone_nested(item) for item in value)
    return value


def apply_ms_selection(
    batch: Mapping[str, Any],
    *,
    budgets: Sequence[int],
    mode: str,
    sample_ids: Sequence[int],
    global_seed: int = 3247,
    focus_scores: torch.Tensor | None = None,
    top_r: int = 8,
) -> dict[str, Any]:
    """Return a cloned batch with MS evidence selected before model encoding."""

    result = clone_nested(batch)
    encoder_input = result.get("encoder_input")
    modality_masks = result.get("encoder_modality_pad_masks")
    if not isinstance(encoder_input, Mapping) or "MSMS" not in encoder_input:
        raise KeyError("AEB requires encoder_input['MSMS']")
    if not isinstance(modality_masks, Mapping) or "MSMS" not in modality_masks:
        raise KeyError("AEB requires encoder_modality_pad_masks['MSMS']")
    original_mask = modality_masks["MSMS"]
    if not isinstance(original_mask, torch.Tensor):
        raise TypeError("MSMS pad mask must be a tensor")
    indices, selected_mask = build_selection_indices(
        ~original_mask.bool(),
        budgets=budgets,
        mode=mode,
        sample_ids=sample_ids,
        global_seed=global_seed,
        focus_scores=focus_scores,
        top_r=top_r,
    )
    result["encoder_input"]["MSMS"] = _gather_sequence_first(encoder_input["MSMS"], indices)
    result["encoder_modality_pad_masks"]["MSMS"] = selected_mask
    parts = []
    for name, mask in result["encoder_modality_pad_masks"].items():
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"Pad mask for {name} is not a tensor")
        parts.append(mask)
    if parts:
        result["encoder_pad_mask"] = torch.cat(parts, dim=0)
    result.setdefault("aeb_selection", {})
    result["aeb_selection"].update(
        {
            "mode": mode,
            "budgets": [int(value) for value in budgets],
            "sample_ids": [int(value) for value in sample_ids],
            "selected_indices": indices.cpu().tolist(),
            "selected_pad_mask": selected_mask.cpu().tolist(),
        }
    )
    return result


def _valid_ms_positions(valid_mask: torch.Tensor, column: int) -> tuple[int, int, list[int]]:
    """Return BOS, EOS, and interior positions for one sequence-first row."""

    valid_positions = torch.nonzero(valid_mask[:, column].bool(), as_tuple=False).flatten().tolist()
    if not valid_positions:
        return 0, 0, []
    bos = int(valid_positions[0])
    eos = int(valid_positions[-1]) if len(valid_positions) > 1 else bos
    interior = [int(value) for value in valid_positions[1:-1]] if len(valid_positions) > 2 else []
    return bos, eos, interior


def _select_peak_positions(
    interior: list[int],
    budget: int,
    mode: str,
    sample_id: int,
    seed: int,
    *,
    focus_scores: torch.Tensor | None = None,
) -> list[int]:
    """Select complete ``(m/z, intensity)`` pairs from an MS sequence.

    The MS text tokenizer emits one numeric token per value.  Peak selection
    therefore always operates on adjacent pairs and returns both positions.
    ``noise_score`` is an explicit non-learned ranking control: independent
    random scores rank complete peaks, while ``random`` samples peak indices
    directly.  ``focus`` accepts learned per-token scores and ranks by the
    mean score of each pair.
    """

    pairs = [interior[index : index + 2] for index in range(0, len(interior) - 1, 2)]
    count = min(max(int(budget), 0), len(pairs))
    if count <= 0:
        return []
    mode = str(mode).lower()
    if count == len(pairs):
        chosen = list(range(len(pairs)))
    elif mode == "random":
        rng = np.random.default_rng(stable_sample_seed(sample_id, seed))
        chosen = sorted(int(value) for value in rng.choice(len(pairs), size=count, replace=False).tolist())
    elif mode == "noise_score":
        rng = np.random.default_rng(stable_sample_seed(sample_id, seed))
        noise = rng.standard_normal(len(pairs))
        chosen = sorted(int(value) for value in np.argsort(noise)[-count:].tolist())
    elif mode == "focus":
        if focus_scores is None:
            raise ValueError("focus peak selection requires focus_scores")
        pair_scores = []
        for pair in pairs:
            pair_scores.append(float(focus_scores[pair].mean().detach().cpu()))
        chosen = sorted(int(value) for value in np.argsort(np.asarray(pair_scores))[-count:].tolist())
    else:
        raise ValueError(f"Unknown peak selection mode: {mode!r}")
    selected: list[int] = []
    for pair_index in chosen:
        selected.extend(pairs[pair_index])
    return selected


def _select_numeric_positions(
    interior: list[int],
    budget: int,
    sample_id: int,
    seed: int,
) -> list[int]:
    """Select individual numeric tokens, intentionally allowing broken pairs."""

    count = min(max(int(budget), 0), len(interior))
    if count <= 0:
        return []
    rng = np.random.default_rng(stable_sample_seed(sample_id, seed))
    return sorted(int(value) for value in rng.choice(np.asarray(interior), size=count, replace=False).tolist())


def build_peak_selection_indices(
    valid_mask: torch.Tensor,
    *,
    budgets: Sequence[int],
    mode: str,
    sample_ids: Sequence[int],
    global_seed: int = 3247,
    focus_scores: torch.Tensor | None = None,
    preserve_positions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Build peak-level selection indices and masks for compressed/masked controls."""

    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be [sequence, batch]")
    seq_len, batch_size = valid_mask.shape
    if len(budgets) != batch_size or len(sample_ids) != batch_size:
        raise ValueError("budgets and sample_ids must match batch size")
    if focus_scores is not None and focus_scores.shape != (batch_size, seq_len):
        raise ValueError("focus_scores must be [batch, sequence]")
    focus_scores = focus_scores.to(valid_mask.device) if focus_scores is not None else None
    selected_rows: list[list[int]] = []
    for column in range(batch_size):
        _, _, interior = _valid_ms_positions(valid_mask, column)
        row_scores = focus_scores[column] if focus_scores is not None else None
        selected_rows.append(
            _select_peak_positions(
                interior,
                int(budgets[column]),
                mode,
                int(sample_ids[column]),
                global_seed,
                focus_scores=row_scores,
            )
        )

    if preserve_positions:
        indices = torch.arange(seq_len, dtype=torch.long, device=valid_mask.device).unsqueeze(1).expand(seq_len, batch_size).clone()
        output_mask = ~valid_mask.bool().clone()
        for column in range(batch_size):
            bos, eos, interior = _valid_ms_positions(valid_mask, column)
            allowed = {bos, eos, *selected_rows[column]}
            for position in interior:
                if position not in allowed:
                    output_mask[position, column] = True
        return indices, output_mask, selected_rows

    max_budget = max([2 * max(0, int(value)) for value in budgets] + [0])
    out_len = max_budget + 2
    indices = torch.zeros((out_len, batch_size), dtype=torch.long, device=valid_mask.device)
    output_mask = torch.ones((out_len, batch_size), dtype=torch.bool, device=valid_mask.device)
    for column in range(batch_size):
        bos, eos, _ = _valid_ms_positions(valid_mask, column)
        selected = [bos, *selected_rows[column], eos]
        length = len(selected)
        indices[:length, column] = torch.tensor(selected, dtype=torch.long, device=indices.device)
        output_mask[:length, column] = False
    return indices, output_mask, selected_rows


def apply_ms_peak_selection(
    batch: Mapping[str, Any],
    *,
    budgets: Sequence[int],
    mode: str = "random",
    sample_ids: Sequence[int],
    global_seed: int = 3247,
    focus_scores: torch.Tensor | None = None,
    preserve_positions: bool = False,
) -> dict[str, Any]:
    """Select complete MS peaks, either by compression or fixed-position masking."""

    result = clone_nested(batch)
    encoder_input = result.get("encoder_input")
    modality_masks = result.get("encoder_modality_pad_masks")
    if not isinstance(encoder_input, Mapping) or "MSMS" not in encoder_input:
        raise KeyError("peak selection requires encoder_input['MSMS']")
    if not isinstance(modality_masks, Mapping) or "MSMS" not in modality_masks:
        raise KeyError("peak selection requires encoder_modality_pad_masks['MSMS']")
    original_mask = modality_masks["MSMS"]
    if not isinstance(original_mask, torch.Tensor):
        raise TypeError("MSMS pad mask must be a tensor")
    indices, selected_mask, selected_rows = build_peak_selection_indices(
        ~original_mask.bool(),
        budgets=budgets,
        mode=mode,
        sample_ids=sample_ids,
        global_seed=global_seed,
        focus_scores=focus_scores,
        preserve_positions=preserve_positions,
    )
    if not preserve_positions:
        result["encoder_input"]["MSMS"] = _gather_sequence_first(encoder_input["MSMS"], indices)
        result["encoder_modality_pad_masks"]["MSMS"] = selected_mask
    else:
        result["encoder_modality_pad_masks"]["MSMS"] = selected_mask
    parts = [mask for mask in result["encoder_modality_pad_masks"].values()]
    result["encoder_pad_mask"] = torch.cat(parts, dim=0)
    result.setdefault("aeb_selection", {})
    result["aeb_selection"].update(
        {
            "unit": "peak",
            "mode": mode,
            "preserve_positions": bool(preserve_positions),
            "budgets_peaks": [int(value) for value in budgets],
            "selected_original_positions": selected_rows,
            "selected_pad_mask": selected_mask.cpu().tolist(),
        }
    )
    return result


def apply_ms_numeric_selection(
    batch: Mapping[str, Any],
    *,
    budgets: Sequence[int],
    sample_ids: Sequence[int],
    global_seed: int = 3247,
) -> dict[str, Any]:
    """Randomly retain individual numeric tokens, deliberately breaking peaks."""

    result = clone_nested(batch)
    encoder_input = result.get("encoder_input")
    modality_masks = result.get("encoder_modality_pad_masks")
    if not isinstance(encoder_input, Mapping) or "MSMS" not in encoder_input:
        raise KeyError("numeric selection requires encoder_input['MSMS']")
    original_mask = modality_masks["MSMS"]
    valid_mask = ~original_mask.bool()
    seq_len, batch_size = valid_mask.shape
    if len(budgets) != batch_size or len(sample_ids) != batch_size:
        raise ValueError("budgets and sample_ids must match batch size")
    rows: list[list[int]] = []
    for column in range(batch_size):
        _, _, interior = _valid_ms_positions(valid_mask, column)
        rows.append(_select_numeric_positions(interior, int(budgets[column]), int(sample_ids[column]), global_seed))
    out_len = max([max(0, int(value)) for value in budgets] + [0]) + 2
    indices = torch.zeros((out_len, batch_size), dtype=torch.long, device=valid_mask.device)
    selected_mask = torch.ones((out_len, batch_size), dtype=torch.bool, device=valid_mask.device)
    for column in range(batch_size):
        bos, eos, _ = _valid_ms_positions(valid_mask, column)
        selected = [bos, *rows[column], eos]
        indices[: len(selected), column] = torch.tensor(selected, dtype=torch.long, device=indices.device)
        selected_mask[: len(selected), column] = False
    result["encoder_input"]["MSMS"] = _gather_sequence_first(encoder_input["MSMS"], indices)
    result["encoder_modality_pad_masks"]["MSMS"] = selected_mask
    result["encoder_pad_mask"] = torch.cat(list(result["encoder_modality_pad_masks"].values()), dim=0)
    result.setdefault("aeb_selection", {})
    result["aeb_selection"].update(
        {
            "unit": "numeric_token",
            "mode": "random",
            "budgets_numeric_tokens": [int(value) for value in budgets],
            "selected_original_positions": rows,
            "selected_pad_mask": selected_mask.cpu().tolist(),
        }
    )
    return result


class AEBCompatibility(nn.Module):
    """Dual tower trained only on frozen source representations."""

    def __init__(
        self,
        anchor_dim: int = 1024,
        ms_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.anchor_tower = nn.Sequential(
            nn.Linear(anchor_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.ms_tower = nn.Sequential(
            nn.Linear(ms_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_anchor(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.anchor_tower(value), dim=-1)

    def encode_ms(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.ms_tower(value), dim=-1)

    def score_embeddings(self, anchor: torch.Tensor, ms: torch.Tensor) -> torch.Tensor:
        return self.encode_anchor(anchor) @ self.encode_ms(ms).transpose(0, 1) / self.temperature

    def score_pairs(self, anchor: torch.Tensor, ms: torch.Tensor) -> torch.Tensor:
        return (self.encode_anchor(anchor) * self.encode_ms(ms)).sum(dim=-1) / self.temperature

    def score_token_pairs(
        self,
        anchor: torch.Tensor,
        ms_tokens: torch.Tensor,
        ms_valid_mask: torch.Tensor | None = None,
        top_r: int = 8,
    ) -> torch.Tensor:
        """Score aligned anchor/MS samples using top-r token compatibility."""
        if ms_tokens.ndim != 3:
            raise ValueError("ms_tokens must be [batch, sequence, feature]")
        query = self.encode_anchor(anchor)
        keys = self.encode_ms(ms_tokens)
        token_scores = torch.einsum("bd,bld->bl", query, keys) / self.temperature
        if ms_valid_mask is not None:
            if ms_valid_mask.shape != token_scores.shape:
                raise ValueError("ms_valid_mask must match [batch, sequence]")
            token_scores = token_scores.masked_fill(~ms_valid_mask.to(dtype=torch.bool), float("-inf"))
        count = min(max(int(top_r), 1), token_scores.shape[1])
        values = torch.topk(token_scores, k=count, dim=1).values
        finite = torch.isfinite(values)
        return values.masked_fill(~finite, 0.0).sum(dim=1) / finite.sum(dim=1).clamp_min(1)

    def score_token_matrix(
        self,
        anchor: torch.Tensor,
        ms_tokens: torch.Tensor,
        ms_valid_mask: torch.Tensor | None = None,
        top_r: int = 8,
    ) -> torch.Tensor:
        """Return all-pairs top-r token compatibility scores."""
        if ms_tokens.ndim != 3:
            raise ValueError("ms_tokens must be [batch, sequence, feature]")
        query = self.encode_anchor(anchor)
        keys = self.encode_ms(ms_tokens)
        scores = torch.einsum("ad,bld->abl", query, keys) / self.temperature
        if ms_valid_mask is not None:
            scores = scores.masked_fill(~ms_valid_mask.to(dtype=torch.bool).unsqueeze(0), float("-inf"))
        count = min(max(int(top_r), 1), scores.shape[-1])
        values = torch.topk(scores, k=count, dim=-1).values
        finite = torch.isfinite(values)
        return values.masked_fill(~finite, 0.0).sum(dim=-1) / finite.sum(dim=-1).clamp_min(1)

    def forward(self, anchor: torch.Tensor, ms: torch.Tensor) -> torch.Tensor:
        return self.score_embeddings(anchor, ms)

    @staticmethod
    def info_nce_loss(scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError("InfoNCE scores must be square")
        return F.cross_entropy(scores, torch.arange(scores.shape[0], device=scores.device))

    @staticmethod
    def margin_ranking_loss(positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
        if positive.shape != negative.shape:
            raise ValueError("positive and negative scores must have equal shape")
        return F.relu(float(margin) - positive + negative).mean()

#!/usr/bin/env python
"""ICLR OpenNMT router/teacher/full/stable-dropout pipeline.

This runner is deliberately independent of analytical_fm.  It consumes the
already prepared SDBS OpenNMT lines, applies deterministic checkpoint-aware
source token normalization, and loads the official
OpenNMT checkpoints, and exposes four resumable commands:

``route``
    Score every valid modality action with the reproduced checkpoint and write
    a frozen route manifest.  Real SDBS uses a label-free checkpoint-native
    pseudo-reward; simulated source data can use the original target-aware
    beam-rank reward against ``tgt.txt``.
``stage1``
    Train the same checkpoint initialized twice conceptually: a frozen
    router-view teacher creates pseudo token sequences and a Full-view student
    learns them with teacher-forcing CE.
``stage2``
    Train from the best Stage-1 checkpoint on strict stable pseudo labels while
    randomly dropping one to three supported spectral modalities.
``pipeline``
    Run all three steps for one checkpoint.

The implementation keeps the source rows as whitespace-tokenized OpenNMT
tokens.  Single-segment SDBS MS is normalized to ``E0Pos`` to match the
official SDBS zero-shot formatter; this is a lexical mapping, not a claim
that EI-MS is equivalent to simulated MS/MS.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


ACTION_SUBSETS: tuple[tuple[str, ...], ...] = (
    ("HNMR",), ("CNMR",), ("HNMR", "CNMR"), ("MSMS",),
    ("HNMR", "MSMS"), ("CNMR", "MSMS"), ("HNMR", "CNMR", "MSMS"),
    ("IR",), ("HNMR", "IR"), ("CNMR", "IR"), ("HNMR", "CNMR", "IR"),
    ("MSMS", "IR"), ("HNMR", "MSMS", "IR"), ("CNMR", "MSMS", "IR"),
    ("HNMR", "CNMR", "MSMS", "IR"),
)
MODALITIES = ("HNMR", "CNMR", "MSMS", "IR")
NUMERIC_TOKEN_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
FORMULA_PART_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
MODEL_SUPPORT = {
    "nmr": frozenset(("HNMR", "CNMR")),
    "ir": frozenset(("IR",)),
    "ms": frozenset(("MSMS",)),
    "ir_ms": frozenset(("IR", "MSMS")),
    "nmr_ir": frozenset(("HNMR", "CNMR", "IR")),
    "nmr_ms": frozenset(("HNMR", "CNMR", "MSMS")),
    "all": frozenset(MODALITIES),
}


@dataclass
class Row:
    source: str
    target: str
    index: int
    formula: str
    parts: dict[str, list[str]]


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("route", "router", "stage1", "stage2", "pipeline"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument(
        "--router-src", type=Path,
        help=(
            "Optional augmented source used only to train the native router and "
            "compute its checkpoint-native rewards. Stage 1/2 continue to use --src."
        ),
    )
    p.add_argument("--tgt", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--official-code", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=0,
                   help="0 means unbounded epochs; requires early-stop-patience")
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--early-stop-min-epochs", type=int, default=2)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    p.add_argument("--stage1-checkpoint", type=Path)
    p.add_argument("--route-manifest", type=Path)
    p.add_argument("--seed", type=int, default=3247)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument(
        "--ms-energy-mode", choices=("all", "e1"), default="all",
        help=(
            "MS input contract. e1 keeps one generic SDBS MS section and maps it "
            "to E0Pos; all keeps every explicit E0Pos/E1Pos/E2Pos section and "
            "maps only a generic single-MS row to E0Pos."
        ),
    )
    p.add_argument(
        "--numeric-token-policy", choices=("exact", "nearest"), default="exact",
        help=(
            "Source numeric handling: exact preserves official OpenNMT lookup "
            "(unseen values become <unk>); nearest is an explicit ablation."
        ),
    )
    p.add_argument(
        "--router-numeric-token-policy", choices=("exact", "nearest"),
        help=(
            "Token policy for --router-src. Defaults to --numeric-token-policy; "
            "use nearest for medium native augmentation."
        ),
    )
    p.add_argument(
        "--router-reward",
        choices=("self_nll", "target_nll", "target_beam_rank"),
        default="self_nll",
        help=(
            "Reward used to train the native router. self_nll is label-free and "
            "is safe for real SDBS; target_nll scores the provided target SMILES "
            "and target_beam_rank scores target-aware beam Top-k plus NLL; both "
            "target modes are only for simulated/source data diagnostics."
        ),
    )
    p.add_argument("--router-beams", type=int, default=5,
                   help="Beam count for target_beam_rank source reward.")
    p.add_argument("--router-top1-weight", type=float, default=1.0)
    p.add_argument("--router-top5-weight", type=float, default=0.25)
    p.add_argument("--router-top10-weight", type=float, default=0.0)
    p.add_argument("--router-nll-weight", type=float, default=0.05)
    p.add_argument("--router-invalid-penalty", type=float, default=0.10)
    p.add_argument("--router-oracle-temperature", type=float, default=0.20,
                   help="Soft-KL target temperature, matching the original router.")
    p.add_argument(
        "--router-target-mode", choices=("hard_oracle", "soft_reward"),
        default="soft_reward",
        help=(
            "Router supervision target. soft_reward is the v3 masked "
            "temperature-scaled KL target; hard_oracle is an explicit ablation."
        ),
    )
    p.add_argument(
        "--router-feature-mode", choices=("source_reference_free", "token_stats"),
        default="source_reference_free",
        help=(
            "Observable router features. source_reference_free uses Formula-" 
            "anchored encoder cosine and pair cosine features, matching the "
            "original pipeline; token_stats is the legacy length-only ablation."
        ),
    )
    return p.parse_args()


def _router_source(args: argparse.Namespace) -> Path:
    return getattr(args, "router_src", None) or args.src


def _router_token_policy(args: argparse.Namespace) -> str:
    return getattr(args, "router_numeric_token_policy", None) or args.numeric_token_policy


def _validate_router_reward_config(args: argparse.Namespace) -> None:
    """Reject target beam rewards that would make every action a zero tie."""
    if args.router_reward != "target_beam_rank":
        return
    weights = (
        args.router_top1_weight,
        args.router_top5_weight,
        args.router_top10_weight,
        args.router_nll_weight,
        args.router_invalid_penalty,
    )
    if not any(abs(float(value)) > 0.0 for value in weights):
        raise ValueError(
            "target_beam_rank requires at least one non-zero reward term; "
            "otherwise all actions tie at zero and argmax applies an index bias"
        )


def _epoch_number(path: Path) -> int:
    """Extract the numeric epoch from a stage checkpoint/prediction filename."""
    match = re.search(r"_epoch_(\d+)(?:\.pt|\.json)$", path.name)
    if match is None:
        raise ValueError(f"stage epoch filename has no numeric epoch: {path.name}")
    return int(match.group(1))


def _select_reward_action(scores: torch.Tensor, nlls: torch.Tensor) -> torch.Tensor:
    """Select max reward, resolving exact reward ties by lower target NLL.

    Top-k beam rewards are discrete, so a zero-reward tie is common when no
    candidate contains the target.  A plain tensor ``argmax`` then silently
    turns action ordering into a policy.  NLL is already computed for every
    valid target-aware action and is the least surprising secondary signal.
    """
    finite = torch.isfinite(scores)
    best_score = scores.max(dim=1, keepdim=True).values
    tied = finite & scores.eq(best_score)
    secondary = torch.where(tied, -nlls, torch.full_like(nlls, float("-inf")))
    selected = secondary.argmax(dim=1)
    no_nll = ~torch.isfinite(secondary).any(dim=1)
    if bool(no_nll.any()):
        selected = torch.where(no_nll, scores.argmax(dim=1), selected)
    return selected


def _assert_router_alignment(clean_rows: Sequence[Row], router_rows: Sequence[Row]) -> None:
    """Ensure augmented rows preserve the clean sample ordering and formula."""
    if len(clean_rows) != len(router_rows):
        raise ValueError(
            "--src and --router-src must contain the same number of rows "
            f"within --limit: {len(clean_rows)} vs {len(router_rows)}"
        )
    for clean, augmented in zip(clean_rows, router_rows):
        if clean.index != augmented.index or clean.formula != augmented.formula:
            raise ValueError(
                "--router-src is not aligned with --src at row "
                f"{clean.index}; preserve source ordering and formula tokens"
            )


def _load_onmt(args: argparse.Namespace):
    sys.path.insert(0, str(args.official_code))
    import argparse as _argparse
    import torch.serialization
    torch.serialization.add_safe_globals([_argparse.Namespace])
    from onmt import opts
    parser = _argparse.ArgumentParser()
    opts.config_opts(parser)
    opts.translate_opts(parser)
    opt = parser.parse_args([
        "-model", str(args.checkpoint), "-src", "/dev/null",
        "-output", "/dev/null", "-gpu", "0" if args.device.startswith("cuda") else "-1",
    ])
    from onmt.bin.translate import build_translator
    translator = build_translator(opt, report_score=False)
    model = translator.model
    # Reuse the official OpenNMT beam implementation for source-GT reward.
    # The runner's tensor batches already have the same `src`/`srclen` contract
    # consumed by Translator.translate_batch.
    model._iclr_translator = translator
    model.train()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    vocabs = translator.vocabs
    return model, vocabs, device


def _tok_id(vocab: Any, token: str) -> int:
    return int(vocab.lookup_token(token))


_NUMERIC_VOCAB_CACHE: dict[int, tuple[list[float], list[str]]] = {}


def _numeric_vocab(vocab: Any) -> tuple[list[float], list[str]]:
    """Return numeric source-vocabulary values for checkpoint-aware mapping.

    OpenNMT's released checkpoints have fixed source vocabularies.  Raw SDBS
    peak values are often not present verbatim, so passing them directly to
    ``lookup_token`` silently produces ``<unk>``.  The official training
    formatter emits discrete numeric strings; for spectral values we map an
    unseen value to the nearest numeric token from that checkpoint's source
    vocabulary.  The cache is per vocabulary object because every checkpoint
    has a different source vocabulary.
    """
    key = id(vocab)
    cached = _NUMERIC_VOCAB_CACHE.get(key)
    if cached is not None:
        return cached
    values: list[tuple[float, str]] = []
    for token in getattr(vocab, "ids_to_tokens", ()):
        text = str(token)
        if NUMERIC_TOKEN_RE.fullmatch(text):
            try:
                values.append((float(text), text))
            except ValueError:
                continue
    values.sort(key=lambda item: (item[0], item[1]))
    result = ([item[0] for item in values], [item[1] for item in values])
    _NUMERIC_VOCAB_CACHE[key] = result
    return result


def _vocab_has(vocab: Any, token: str) -> bool:
    """Check whether ``token`` is represented instead of mapped to ``<unk>``."""
    unk = _tok_id(vocab, "<unk>")
    return _tok_id(vocab, token) != unk or token == "<unk>"


def _nearest_numeric_token(token: str, vocab: Any, numeric_policy: str = "exact") -> tuple[str, bool]:
    """Map an unseen numeric token to the nearest legal checkpoint token."""
    if numeric_policy != "nearest" or _vocab_has(vocab, token) or not NUMERIC_TOKEN_RE.fullmatch(token):
        return token, False
    try:
        value = float(token)
    except ValueError:
        return token, False
    values, tokens = _numeric_vocab(vocab)
    if not values:
        return token, False
    position = min(range(len(values)), key=lambda index: (abs(values[index] - value), values[index]))
    return tokens[position], True


def _source_token_groups(row: Row, keep: Iterable[str]) -> list[tuple[str, list[str]]]:
    """Return formula and kept modality tokens with their modality labels."""
    keep_set = set(keep)
    groups: list[tuple[str, list[str]]] = [("FORMULA", row.formula.split())]
    for name in ("HNMR", "CNMR", "IR", "MSMS"):
        if name in keep_set and name in row.parts:
            groups.append((name, list(row.parts[name])))
    return groups


def _canonical_source_tokens(
    row: Row,
    keep: Iterable[str],
    vocab: Any,
    numeric_policy: str = "exact",
) -> tuple[list[str], dict[str, int]]:
    """Canonicalize raw SDBS tokens for one checkpoint's source vocabulary.

    Formula and spectral numeric tokens retain their exact spelling by default,
    matching the official OpenNMT ``mode=none`` Field (whitespace split plus
    source-vocabulary lookup).  ``numeric_policy='nearest'`` is available only
    as an explicit compatibility ablation.  Legacy aliases are normalized at
    the same boundary so route training and inference see identical IDs.
    """
    output: list[str] = []
    stats = {"tokens": 0, "unknown": 0, "numeric_mapped": 0}
    for modality, group in _source_token_groups(row, keep):
        for offset, original in enumerate(group):
            token = original
            if modality == "MSMS" and token == "MS":
                token = "E0Pos"
            elif token == "derived_shift":
                token = "m"
            if modality != "FORMULA":
                token, mapped = _nearest_numeric_token(token, vocab, numeric_policy)
                stats["numeric_mapped"] += int(mapped)
            stats["tokens"] += 1
            stats["unknown"] += int(not _vocab_has(vocab, token))
            output.append(token)
    return output, stats


def _parse_rows(
    src_path: Path,
    tgt_path: Path | None,
    limit: int,
    *,
    read_targets: bool = False,
    ms_energy_mode: str = "e1",
) -> list[Row]:
    src = src_path.read_text(encoding="utf-8").splitlines()
    tgt = (tgt_path.read_text(encoding="utf-8").splitlines()
           if read_targets and tgt_path is not None else [""] * len(src))
    if len(src) != len(tgt):
        raise ValueError(f"source/target line counts differ: {len(src)} vs {len(tgt)}")
    rows: list[Row] = []
    for index, (source, target) in enumerate(zip(src[:limit], tgt[:limit])):
        tokens = source.split()
        marker_kinds = {
            "1HNMR": "HNMR", "13CNMR": "CNMR", "IR": "IR",
            "MS": "MSMS", "E0Pos": "MSMS", "E1Pos": "MSMS", "E2Pos": "MSMS",
        }
        marker_positions = [
            (position, marker, marker_kinds[marker])
            for position, marker in enumerate(tokens) if marker in marker_kinds
        ]
        if not marker_positions:
            raise ValueError(f"row {index} has no spectral marker")
        formula = " ".join(tokens[: marker_positions[0][0]])
        parts: dict[str, list[str]] = {}
        for position, (start, marker, name) in enumerate(marker_positions):
            end = (
                marker_positions[position + 1][0]
                if position + 1 < len(marker_positions) else len(tokens)
            )
            if name == "MSMS":
                if ms_energy_mode == "e1" and marker not in {"MS", "E1Pos"}:
                    continue
                # Preserve official multi-energy markers in ``all`` mode. The
                # Preserve explicit energy markers.  A generic SDBS ``MS`` row
                # has no energy label; the official SDBS formatter maps it to
                # the checkpoint's first-energy E0Pos marker.
                canonical_marker = "E0Pos" if marker == "MS" else marker
                # The simulated export can carry an auxiliary ``MSMS null``
                # header after a generic MS marker. It is not a source token.
                body = list(tokens[start + 1:end])
                if body[:2] == ["MSMS", "null"]:
                    body = body[2:]
                segment = [canonical_marker, *body]
                if name in parts:
                    # Keep every explicit energy marker in the flattened
                    # OpenNMT MS group: E0Pos peaks, then E1Pos peaks, then
                    # E2Pos peaks. Dropping later markers silently changes
                    # the checkpoint's simulated MS contract.
                    parts[name].extend(segment)
                else:
                    parts[name] = segment
            else:
                parts[name] = tokens[start:end]
        if ms_energy_mode not in {"all", "e1"}:
            raise ValueError(f"unknown MS energy mode: {ms_energy_mode}")
        rows.append(Row(source, target, index, formula, parts))
    return rows


def _compose(row: Row, keep: Iterable[str], vocab: Any | None = None,
             numeric_policy: str = "exact") -> str:
    if vocab is None:
        return " ".join(token for _modality, group in _source_token_groups(row, keep) for token in group)
    tokens, _stats = _canonical_source_tokens(row, keep, vocab, numeric_policy)
    return " ".join(tokens)


def _batch_src(rows: Sequence[Row], vocabs: Mapping[str, Any], keep_rows: Sequence[Sequence[str]],
               device: torch.device, numeric_policy: str = "exact") -> tuple[torch.Tensor, torch.Tensor]:
    vocab = vocabs["src"]
    sequences = []
    for row, keep in zip(rows, keep_rows):
        tokens, _stats = _canonical_source_tokens(row, keep, vocab, numeric_policy)
        sequences.append([_tok_id(vocab, token) for token in tokens])
    pad = _tok_id(vocab, "<blank>")
    length = torch.tensor([len(value) for value in sequences], dtype=torch.long, device=device)
    width = int(length.max().item())
    source = torch.full((len(sequences), width, 1), pad, dtype=torch.long, device=device)
    for index, value in enumerate(sequences):
        source[index, :len(value), 0] = torch.tensor(value, dtype=torch.long, device=device)
    return source, length


def _tokenization_summary(rows: Sequence[Row], vocabs: Mapping[str, Any],
                          support: frozenset[str] | None = None,
                          numeric_policy: str = "exact") -> dict[str, Any]:
    """Summarize the effective source tokenization for audit manifests."""
    vocab = vocabs["src"]
    totals = {"tokens": 0, "unknown": 0, "numeric_mapped": 0}
    by_modality: dict[str, dict[str, int]] = {
        "FORMULA": {key: 0 for key in totals},
        **{name: {key: 0 for key in totals} for name in MODALITIES},
    }
    for row in rows:
        selected = tuple(name for name in MODALITIES if support is None or name in support)
        for modality, group in _source_token_groups(row, selected):
            # Normalize one group without changing the fixed ordering used by
            # _batch_src.  Formula is exact; spectral numbers follow the
            # selected official lookup or explicit nearest-value ablation.
            for offset, original in enumerate(group):
                token = original
                if modality == "MSMS" and token == "MS":
                    token = "E0Pos"
                elif token == "derived_shift":
                    token = "m"
                mapped = False
                if modality != "FORMULA":
                    token, mapped = _nearest_numeric_token(token, vocab, numeric_policy)
                values = by_modality[modality]
                values["tokens"] += 1
                values["unknown"] += int(not _vocab_has(vocab, token))
                values["numeric_mapped"] += int(mapped)
    for values in by_modality.values():
        for key in totals:
            totals[key] += values[key]
    return {"total": totals, "by_modality": by_modality}


def _target_ids(text: str, vocab: Any, max_length: int) -> list[int]:
    values = [_tok_id(vocab, "<s>")]
    values.extend(_tok_id(vocab, token) for token in text.split())
    values.append(_tok_id(vocab, "</s>"))
    return values[:max_length]


def _pad_targets(rows: Sequence[Sequence[int]], pad: int, device: torch.device) -> torch.Tensor:
    width = max(len(value) for value in rows)
    output = torch.full((len(rows), width, 1), pad, dtype=torch.long, device=device)
    for index, value in enumerate(rows):
        output[index, :len(value), 0] = torch.tensor(value, dtype=torch.long, device=device)
    return output


def _forward_nll(model: Any, source: torch.Tensor, lengths: torch.Tensor,
                 target: torch.Tensor, pad: int) -> torch.Tensor:
    decoder, _ = model(source, target, lengths)
    logits = model.generator(decoder).float()
    labels = target[:, 1:, 0]
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                             ignore_index=pad, reduction="none").view_as(labels)
    valid = labels.ne(pad)
    return (losses * valid).sum(1) / valid.sum(1).clamp_min(1)


@torch.no_grad()
def _greedy(model: Any, source: torch.Tensor, lengths: torch.Tensor,
            vocabs: Mapping[str, Any], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    vocab = vocabs["tgt"]
    bos, eos, pad = (_tok_id(vocab, "<s>"), _tok_id(vocab, "</s>"), _tok_id(vocab, "<blank>"))
    enc_out, enc_final, src_len = model.encoder(source, lengths)
    model.decoder.init_state(source, enc_out, enc_final)
    current = torch.full((source.shape[0], 1, 1), bos, dtype=torch.long, device=source.device)
    done = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
    scores = torch.zeros(source.shape[0], dtype=torch.float32, device=source.device)
    for step in range(max_length - 1):
        decoder, _ = model.decoder(current[:, -1:], enc_out, src_len=src_len, step=step)
        logp = F.log_softmax(model.generator(decoder[:, -1]).float(), dim=-1)
        next_id = logp.argmax(-1)
        scores += logp.gather(1, next_id[:, None]).squeeze(1).masked_fill(done, 0)
        next_id = next_id.masked_fill(done, pad)
        current = torch.cat((current, next_id[:, None, None]), dim=1)
        done |= next_id.eq(eos)
        if bool(done.all()):
            break
    return current, scores


def _canonical_smiles(text: str) -> str | None:
    molecule = Chem.MolFromSmiles(str(text).replace(" ", ""))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def _formula_signature(text: str) -> tuple[tuple[str, int], ...] | None:
    """Normalize a spaced or compact molecular formula into element counts."""
    compact = re.sub(r"\s+", "", str(text))
    if not compact:
        return None
    counts: dict[str, int] = {}
    cursor = 0
    for match in FORMULA_PART_RE.finditer(compact):
        if match.start() != cursor:
            return None
        element, count = match.groups()
        counts[element] = counts.get(element, 0) + int(count or "1")
        cursor = match.end()
    if cursor != len(compact):
        return None
    return tuple(sorted(counts.items()))


def _smiles_formula_signature(text: str) -> tuple[tuple[str, int], ...] | None:
    molecule = Chem.MolFromSmiles(str(text).replace(" ", ""))
    if molecule is None:
        return None
    return _formula_signature(rdMolDescriptors.CalcMolFormula(molecule))


@torch.no_grad()
def _beam_decode(model: Any, source: torch.Tensor, lengths: torch.Tensor,
                 vocabs: Mapping[str, Any], beams: int, max_length: int,
                 min_length: int = 0) -> list[list[str]]:
    """Return official OpenNMT beam strings grouped by source row."""
    if beams < 1:
        raise ValueError("router-beams must be positive")
    translator = getattr(model, "_iclr_translator", None)
    if translator is None:
        raise RuntimeError("official OpenNMT translator is unavailable for beam reward")
    translator.beam_size = int(beams)
    translator.n_best = int(beams)
    translator.max_length = int(max_length)
    translator.min_length = int(min_length)
    result = translator.translate_batch({"src": source, "srclen": lengths}, False)
    predictions = result["predictions"]
    grouped: list[list[str]] = []
    for row in predictions:
        grouped.append([
            _decode(sequence.detach().cpu().tolist(), vocabs["tgt"])
            for sequence in row[:beams]
        ])
    if len(grouped) != int(source.shape[0]):
        raise RuntimeError(
            f"beam decoder returned {len(grouped)} rows for batch of {source.shape[0]}"
        )
    return grouped


def _decode(sequence: Sequence[int], vocab: Any) -> str:
    eos = _tok_id(vocab, "</s>")
    bos = _tok_id(vocab, "<s>")
    pad = _tok_id(vocab, "<blank>")
    tokens = []
    for value in sequence:
        value = int(value)
        if value in (bos, pad):
            continue
        if value == eos:
            break
        tokens.append(str(vocab.ids_to_tokens[value]))
    return "".join(tokens)


def _valid_actions(support: frozenset[str], rows: Sequence[Row]) -> list[tuple[int, tuple[str, ...]]]:
    """Return actions supported by the checkpoint and present in at least one row.

    Row-level availability is applied in ``route``.  Keeping the union here
    avoids silently dropping an action merely because the first row happens to
    be missing a modality.
    """
    result = []
    for index, subset in enumerate(ACTION_SUBSETS):
        if set(subset).issubset(support) and any(
            set(subset).issubset(row.parts) for row in rows
        ):
            result.append((index, subset))
    if not result:
        raise ValueError(f"checkpoint support {sorted(support)} has no valid action")
    return result


def route(args: argparse.Namespace) -> Path:
    started = time.time()
    _validate_router_reward_config(args)
    router_src = _router_source(args)
    router_policy = _router_token_policy(args)
    use_targets = args.router_reward in {"target_nll", "target_beam_rank"}
    rows = _parse_rows(
        router_src,
        args.tgt if use_targets else None,
        args.limit,
        read_targets=use_targets,
        ms_energy_mode=args.ms_energy_mode,
    )
    model, vocabs, device = _load_onmt(args)
    model.eval()
    name = args.checkpoint.stem
    valid = _valid_actions(MODEL_SUPPORT[name], rows)
    tgt_vocab = vocabs["tgt"]
    tgt_pad = _tok_id(tgt_vocab, "<blank>")
    out_rows: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        scores = torch.full((len(batch), len(ACTION_SUBSETS)), float("-inf"), device=device)
        nlls = scores.clone()
        top1_hits = torch.zeros_like(scores)
        top5_hits = torch.zeros_like(scores)
        top10_hits = torch.zeros_like(scores)
        invalid_actions = torch.zeros_like(scores)
        for action, subset in valid:
            source, lengths = _batch_src(batch, vocabs, [subset] * len(batch), device, router_policy)
            row_valid = torch.tensor(
                [set(subset).issubset(row.parts) for row in batch],
                dtype=torch.bool,
                device=device,
            )
            if use_targets:
                target = _pad_targets(
                    [_target_ids(row.target, vocabs["tgt"], args.max_target_length) for row in batch],
                    tgt_pad,
                    device,
                )
                target_nll = _forward_nll(model, source, lengths, target, tgt_pad)
                nlls[:, action] = torch.where(
                    row_valid, target_nll, torch.full_like(target_nll, float("-inf"))
                )
                # Source/simulation supervision: the reproduced checkpoint's
                # likelihood of the known target structure.  This is the
                # target-aware analogue of the original source reward; it is
                # never enabled for unlabeled real SDBS routing.
                rewards = -target_nll
                if args.router_reward == "target_beam_rank":
                    grouped = _beam_decode(
                        model, source, lengths, vocabs,
                        args.router_beams, args.max_target_length,
                    )
                    target_canonical = [_canonical_smiles(row.target) for row in batch]
                    for local, candidates_raw in enumerate(grouped):
                        candidates = [_canonical_smiles(value) for value in candidates_raw]
                        target = target_canonical[local]
                        rank = (
                            candidates.index(target)
                            if target is not None and target in candidates
                            else len(candidates)
                        )
                        top1_hits[local, action] = float(rank == 0)
                        top5_hits[local, action] = float(
                            rank < min(5, args.router_beams)
                        )
                        top10_hits[local, action] = float(
                            rank < min(10, args.router_beams)
                        )
                        invalid = not candidates_raw or candidates[0] is None
                        invalid_actions[local, action] = float(invalid)
                    rewards = (
                        args.router_top1_weight * top1_hits[:, action]
                        + args.router_top5_weight * top5_hits[:, action]
                        + args.router_top10_weight * top10_hits[:, action]
                        - args.router_nll_weight * target_nll
                        - args.router_invalid_penalty * invalid_actions[:, action]
                    )
                    top1_hits[:, action] = top1_hits[:, action].masked_fill(~row_valid, float("-inf"))
                    top5_hits[:, action] = top5_hits[:, action].masked_fill(~row_valid, float("-inf"))
                    top10_hits[:, action] = top10_hits[:, action].masked_fill(~row_valid, float("-inf"))
                    invalid_actions[:, action] = invalid_actions[:, action].masked_fill(~row_valid, float("-inf"))
            else:
                pseudo, pseudo_score = _greedy(model, source, lengths, vocabs, args.max_target_length)
                pseudo_nll = _forward_nll(model, source, lengths, pseudo, tgt_pad)
                nlls[:, action] = torch.where(
                    row_valid, pseudo_nll, torch.full_like(pseudo_nll, float("-inf"))
                )
                # Label-free real-data fallback: score the checkpoint's own
                # greedy sequence under teacher forcing.
                rewards = -pseudo_nll + 0.05 * pseudo_score / pseudo[:, :, 0].ne(tgt_pad).sum(1).clamp_min(1)
            scores[:, action] = rewards.masked_fill(~row_valid, float("-inf"))
        selected = _select_reward_action(scores, nlls)
        for local, row in enumerate(batch):
            action = int(selected[local])
            pseudo_source, pseudo_len = _batch_src([row], vocabs, [ACTION_SUBSETS[action]], device, router_policy)
            pseudo, pseudo_score = _greedy(model, pseudo_source, pseudo_len, vocabs, args.max_target_length)
            out_rows.append({
                "source_row_index": row.index,
                "selected_action_index": action,
                "selected_modalities": list(ACTION_SUBSETS[action]),
                "action_rewards": [None if not math.isfinite(float(value)) else float(value) for value in scores[local].detach().cpu()],
                "action_nll": [None if not math.isfinite(float(value)) else float(value) for value in nlls[local].detach().cpu()],
                "action_top1": [None if not math.isfinite(float(value)) else float(value) for value in top1_hits[local].detach().cpu()],
                "action_top5": [None if not math.isfinite(float(value)) else float(value) for value in top5_hits[local].detach().cpu()],
                "action_top10": [None if not math.isfinite(float(value)) else float(value) for value in top10_hits[local].detach().cpu()],
                "action_invalid": [None if not math.isfinite(float(value)) else float(value) for value in invalid_actions[local].detach().cpu()],
                "teacher_pseudo_token_ids": [int(value) for value in pseudo[0, :, 0].cpu()],
                "teacher_pseudo_smiles": _decode(pseudo[0, :, 0].cpu().tolist(), vocabs["tgt"]),
                "teacher_pseudo_score": float(pseudo_score[0].cpu()),
            })
        print(json.dumps({"route_rows": len(out_rows)}), flush=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "complete": True, "method": "iclr_open_nmt_checkpoint_native_router",
        "checkpoint": str(args.checkpoint), "checkpoint_name": name,
        "router_source": str(router_src),
        "support_modalities": sorted(MODEL_SUPPORT[name]), "rows": len(out_rows),
        "router_reward": args.router_reward,
            "reward_tie_break": "lower_target_nll_then_action_index",
            "target_labels_used": use_targets,
            "router_reward_config": {
                "beams": int(args.router_beams),
                "top1_weight": float(args.router_top1_weight),
                "top5_weight": float(args.router_top5_weight),
                "top10_weight": float(args.router_top10_weight),
                "nll_weight": float(args.router_nll_weight),
                "invalid_penalty": float(args.router_invalid_penalty),
            },
        "max_target_length": int(args.max_target_length),
        "tokenization": {
            "source_lines": "official whitespace-tokenized SDBS formatter output",
            "checkpoint_aware": True,
            "derived_shift": "m",
            "ms_marker": "E0Pos",
            "spectral_numeric_tokens": (
                "nearest source-vocabulary value when absent"
                if router_policy == "nearest" else "exact source-vocabulary lookup"
            ),
            "formula_tokens": "exact spelling",
            "numeric_token_policy": router_policy,
            "audit": _tokenization_summary(rows, vocabs, MODEL_SUPPORT[name], router_policy),
        },
        "ms_marker": "E0Pos", "ms_energy_mode": args.ms_energy_mode,
        "elapsed_seconds": time.time() - started,
        "routes": out_rows,
    }
    path = args.run_dir / "route_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_routes(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["source_row_index"]): row for row in payload["routes"]}


def _route_manifest_compatible(path: Path, args: argparse.Namespace, limit: int) -> bool:
    """Reject stale route manifests made with a different input contract."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokenization = payload.get("tokenization")
        if not isinstance(tokenization, dict):
            return False
        if str(payload.get("checkpoint")) != str(args.checkpoint):
            return False
        if str(payload.get("router_source", args.src)) != str(_router_source(args)):
            return False
        if str(payload.get("router_reward", "self_nll")) != str(args.router_reward):
            return False
        reward_config = payload.get("router_reward_config")
        expected_reward_config = {
            "beams": int(args.router_beams),
            "top1_weight": float(args.router_top1_weight),
            "top5_weight": float(args.router_top5_weight),
            "top10_weight": float(args.router_top10_weight),
            "nll_weight": float(args.router_nll_weight),
            "invalid_penalty": float(args.router_invalid_penalty),
        }
        if not isinstance(reward_config, dict):
            return False
        for key, expected in expected_reward_config.items():
            observed = reward_config.get(key)
            if isinstance(expected, int):
                if int(observed) != expected:
                    return False
            elif not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12):
                return False
        if payload.get("reward_tie_break") != "lower_target_nll_then_action_index":
            return False
        if int(payload.get("max_target_length", -1)) != int(args.max_target_length):
            return False
        if str(payload.get("ms_energy_mode", "all")) != str(args.ms_energy_mode):
            return False
        if int(payload.get("rows", -1)) != int(limit):
            return False
        if tokenization.get("checkpoint_aware") is not True:
            return False
        if tokenization.get("numeric_token_policy", "exact") != _router_token_policy(args):
            return False
        route_rows = payload.get("routes", ())
        if len(route_rows) != int(limit):
            return False
        return all("teacher_pseudo_token_ids" in row for row in route_rows)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _router_token_features(row: Row, vocabs: Mapping[str, Any],
                           support: frozenset[str], numeric_policy: str = "exact") -> torch.Tensor:
    """Observable features after the exact checkpoint source tokenization.

    Router training must see the same token contract as route reward and
    student training.  In particular, nearest-vocabulary numeric projection
    and alias normalization change effective lengths and unknown rates; raw
    whitespace counts would train against a different input distribution.
    Target SMILES and analytical_fm caches never enter this tensor.
    """
    vocab = vocabs["src"]
    formula_tokens, formula_stats = _canonical_source_tokens(row, (), vocab, numeric_policy)
    values = [
        float(len(formula_tokens)),
        float(formula_stats["unknown"]),
    ]
    total = len(formula_tokens)
    for name in MODALITIES:
        if name not in support:
            values.extend((0.0, 0.0, 0.0, 0.0))
            continue
        tokens, stats = _canonical_source_tokens(row, (name,), vocab, numeric_policy)
        total += len(tokens) - len(formula_tokens)
        values.extend((
            float(len(tokens) - len(formula_tokens)),
            float(stats["unknown"] - formula_stats["unknown"]),
            float(stats["numeric_mapped"]),
            float(bool(row.parts.get(name))),
        ))
    values.append(float(total))
    return torch.tensor(values, dtype=torch.float32)


@torch.no_grad()
def _router_encoder_features(
    model: Any,
    rows: Sequence[Row],
    vocabs: Mapping[str, Any],
    support: frozenset[str],
    numeric_policy: str,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Build the original source-reference-free encoder feature contract.

    Formula remains the anchor.  Each modality is encoded with Formula plus
    that modality, then represented by Formula compatibility and token count;
    pairwise final-layer cosines and global availability/token features follow
    the source-reference-free feature schema used by the original router.
    No target text or reward enters this representation.
    """
    model.eval()
    batch_count = len(rows)
    modality_values: dict[str, torch.Tensor] = {}
    pooled: dict[str, torch.Tensor] = {}
    token_pooled: dict[str, torch.Tensor] = {}
    vocab = vocabs["src"]
    formula_counts = torch.tensor(
        [
            len(_canonical_source_tokens(row, (), vocab, numeric_policy)[0])
            for row in rows
        ],
        dtype=torch.float32,
        device=device,
    )

    def encode(
        views: Sequence[Sequence[str]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_rows: list[torch.Tensor] = []
        token_mean_rows: list[torch.Tensor] = []
        spectral_count_rows: list[torch.Tensor] = []
        for start in range(0, batch_count, batch_size):
            batch = rows[start:start + batch_size]
            keep_rows = [tuple(view) for view in views[start:start + batch_size]]
            source, lengths = _batch_src(batch, vocabs, keep_rows, device, numeric_policy)
            encoded, _final, _encoded_lengths = model.encoder(source, lengths)
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            mask = torch.arange(encoded.shape[1], device=device)[None] < lengths[:, None]
            pooled_batch = (encoded.float() * mask.unsqueeze(-1)).sum(1) / lengths[:, None].clamp_min(1)
            token_normalized = F.normalize(encoded.float(), dim=-1)
            token_mean_batch = (
                token_normalized * mask.unsqueeze(-1)
            ).sum(1) / lengths[:, None].clamp_min(1)
            hidden_rows.append(pooled_batch)
            token_mean_rows.append(token_mean_batch)
            spectral_count_rows.append(
                torch.tensor(
                    [
                        max(
                            0,
                            len(_canonical_source_tokens(row, keep, vocab, numeric_policy)[0])
                            - int(formula_counts[start + offset].item()),
                        )
                        for offset, (row, keep) in enumerate(zip(batch, keep_rows))
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            )
        return (
            torch.cat(hidden_rows, 0),
            torch.cat(token_mean_rows, 0),
            torch.cat(spectral_count_rows, 0),
        )

    formula_hidden, _formula_token_mean, _formula_encoded_len = encode([()] * batch_count)
    anchor = F.normalize(formula_hidden, dim=-1)
    counts_by_modality: dict[str, torch.Tensor] = {}
    availability_by_modality: dict[str, torch.Tensor] = {}
    for name in MODALITIES:
        available = torch.tensor(
            [name in support and name in row.parts for row in rows],
            dtype=torch.bool,
            device=device,
        )
        availability_by_modality[name] = available
        if name not in support:
            pooled[name] = torch.zeros_like(anchor)
            token_pooled[name] = torch.zeros_like(anchor)
            modality_values[name] = torch.zeros((batch_count, 2), device=device)
            counts_by_modality[name] = torch.zeros(batch_count, device=device)
            continue
        hidden, token_mean, spectral_counts = encode([(name,)] * batch_count)
        current = F.normalize(hidden, dim=-1)
        current_token = F.normalize(token_mean, dim=-1)
        current = current.masked_fill(~available[:, None], 0.0)
        current_token = current_token.masked_fill(~available[:, None], 0.0)
        pooled[name] = current
        token_pooled[name] = current_token
        counts_by_modality[name] = spectral_counts.masked_fill(~available, 0.0)
        modality_values[name] = torch.stack(
            ((current * anchor).sum(-1), torch.log1p(counts_by_modality[name])), dim=-1
        )
    modality_tensor = torch.stack(
        [modality_values[name] for name in MODALITIES], dim=1
    )
    pair_values: list[torch.Tensor] = []
    for left in range(len(MODALITIES)):
        for right in range(left + 1, len(MODALITIES)):
            pair_values.append(
                torch.stack(
                    ((pooled[MODALITIES[left]] * pooled[MODALITIES[right]]).sum(-1),
                     (token_pooled[MODALITIES[left]] * token_pooled[MODALITIES[right]]).sum(-1)),
                    dim=-1,
                )
            )
    pair_tensor = torch.stack(pair_values, dim=1)
    available = torch.stack(
        [availability_by_modality[name] for name in MODALITIES], dim=1
    ).float()
    total = torch.stack(list(counts_by_modality.values()), dim=1).sum(1) + formula_counts
    global_tensor = torch.stack(
        (available.sum(1) / 4.0, torch.log1p(total)), dim=-1
    )
    return torch.cat(
        (modality_tensor.reshape(batch_count, -1),
         pair_tensor.reshape(batch_count, -1), global_tensor), dim=1
    ).cpu()


def _train_native_router(args: argparse.Namespace, rows: Sequence[Row],
                         route_path: Path, routes: dict[int, dict[str, Any]],
                         vocabs: Mapping[str, Any], model: Any,
                         device: torch.device) -> Path:
    """Fit a compact Router from checkpoint-native action rewards.

    This is the ICLR replacement for the old analytical_fm router checkpoint.
    Rewards are recomputed by ``route`` before this function is called; rows
    with unavailable actions remain masked throughout training and selection.
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    router_policy = _router_token_policy(args)
    support = MODEL_SUPPORT[args.checkpoint.stem]
    if args.router_feature_mode == "source_reference_free":
        x = _router_encoder_features(
            model, rows, vocabs, support, router_policy, device
        )
    else:
        x = torch.stack([
            _router_token_features(row, vocabs, support, router_policy)
            for row in rows
        ])
    reward_rows = []
    nll_rows = []
    for row in rows:
        values = routes[row.index].get("action_rewards", [])
        reward_rows.append([float(value) if value is not None else -1e4 for value in values])
        values = routes[row.index].get("action_nll", [])
        nll_rows.append([float(value) if value is not None else float("inf") for value in values])
    if args.router_oracle_temperature <= 0:
        raise ValueError("router oracle temperature must be positive")
    rewards = torch.tensor(reward_rows, dtype=torch.float32)
    valid_mask = rewards.gt(-1e3) & torch.isfinite(rewards)
    # Match the original source router: action rewards are converted directly
    # to a soft oracle with a fixed temperature.  Per-row z-scoring is not
    # equivalent and can turn small reward gaps into a collapsed policy.
    target_logits = (rewards / float(args.router_oracle_temperature)).masked_fill(
        ~valid_mask, float("-inf")
    )
    targets = torch.softmax(target_logits, dim=1)
    nll_values = torch.tensor(nll_rows, dtype=torch.float32)
    oracle_actions = _select_reward_action(rewards.masked_fill(~valid_mask, float("-inf")), nll_values)
    model = torch.nn.Sequential(torch.nn.LayerNorm(x.shape[1]), torch.nn.Linear(x.shape[1], 64),
                                torch.nn.GELU(), torch.nn.Linear(64, len(ACTION_SUBSETS)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    # Keep a deterministic 80/20 holdout independent of source row order.
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    split = min(len(rows) - 1, max(1, int(0.8 * len(rows))))
    train_indices = torch.tensor(order[:split], dtype=torch.long)
    valid_indices = torch.tensor(order[split:], dtype=torch.long)
    if args.epochs == 0 and args.early_stop_patience < 1:
        raise ValueError("unbounded router training requires --early-stop-patience > 0")
    class_counts = torch.bincount(
        oracle_actions[train_indices], minlength=len(ACTION_SUBSETS)
    ).float()
    class_weights = torch.zeros_like(class_counts)
    present_classes = class_counts.gt(0)
    class_weights[present_classes] = class_counts[present_classes].rsqrt()
    if bool(present_classes.any()):
        class_weights[present_classes] *= (
            present_classes.sum() / class_weights[present_classes].sum()
        )
    best = float("inf"); stale = 0; history = []; best_state = None
    max_epochs = args.epochs if args.epochs > 0 else 10_000_000
    for epoch in range(1, max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        masked_logits = logits.masked_fill(~valid_mask, -1e9)
        if args.router_target_mode == "hard_oracle":
            train_loss = F.cross_entropy(
                masked_logits[train_indices], oracle_actions[train_indices],
                weight=class_weights,
            )
        else:
            train_loss = F.kl_div(
                F.log_softmax(masked_logits[train_indices], dim=1),
                targets[train_indices], reduction="batchmean"
            )
        train_loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_logits = model(x[valid_indices]).masked_fill(
                ~valid_mask[valid_indices], -1e9
            )
            valid_pred = valid_logits.argmax(dim=1)
            selected_reward = rewards[valid_indices].gather(
                1, valid_pred[:, None]
            ).squeeze(1)
            oracle_reward = rewards[valid_indices].gather(
                1, oracle_actions[valid_indices, None]
            ).squeeze(1)
            valid_regret = float((oracle_reward - selected_reward).mean())
            valid_agreement = float(
                (valid_pred == oracle_actions[valid_indices]).float().mean()
            )
            valid_loss = float(F.kl_div(
                F.log_softmax(valid_logits, dim=1),
                targets[valid_indices], reduction="batchmean"
            ))
            objective = (
                valid_regret
                if args.router_target_mode == "hard_oracle"
                else valid_loss
            )
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss.detach()),
            "validation_kl": valid_loss,
            "validation_mean_regret": valid_regret,
            "validation_oracle_action_agreement": valid_agreement,
            "validation_selection_frequency": torch.bincount(
                valid_pred, minlength=len(ACTION_SUBSETS)
            ).tolist(),
        })
        if objective < best - args.early_stop_min_delta:
            best, stale, best_state = objective, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(json.dumps({"stage": "router", "epoch": epoch,
                          "train_loss": float(train_loss.detach()),
                          "validation_kl": valid_loss,
                          "validation_mean_regret": valid_regret,
                          "validation_oracle_action_agreement": valid_agreement,
                          "stale": stale}),
              flush=True)
        if epoch >= args.early_stop_min_epochs and stale >= args.early_stop_patience:
            break
    if best_state is not None: model.load_state_dict(best_state)
    with torch.no_grad():
        logits = model(x)
    for index, row in enumerate(rows):
        valid = torch.tensor([value > -1e3 for value in reward_rows[index]], dtype=torch.bool)
        masked = logits[index].masked_fill(~valid, float("-inf"))
        # This is the learned, target-free decision.  Do not use source target
        # NLL as a secondary signal here; it is only legal while constructing
        # source oracle supervision above.
        action = int(masked.argmax())
        routes[row.index]["router_selected_action_index"] = action
        routes[row.index]["router_selected_modalities"] = list(ACTION_SUBSETS[action])
    payload = {"state_dict": model.state_dict(), "feature_dim": int(x.shape[1]),
               "action_subsets": ACTION_SUBSETS, "history": history,
               "epochs": len(history), "best_validation_objective": best,
               "best_validation_kl": min(
                   (item["validation_kl"] for item in history), default=float("inf")
               ),
               "early_stop_patience": int(args.early_stop_patience),
               "early_stop_min_epochs": int(args.early_stop_min_epochs),
               "early_stop_min_delta": float(args.early_stop_min_delta),
               "oracle_temperature": float(args.router_oracle_temperature),
               "train_rows": int(len(train_indices)),
               "validation_rows": int(len(valid_indices)),
               "validation_split_seed": int(args.seed),
               "router_target_mode": args.router_target_mode,
               "oracle_action_counts": torch.bincount(
                   oracle_actions, minlength=len(ACTION_SUBSETS)
               ).tolist(),
               "reward_source": (
                   "simulated target-aware OpenNMT beam-rank reward"
                   if args.router_reward == "target_beam_rank" else
                   "simulated target-aware OpenNMT teacher-forced NLL"
                   if args.router_reward == "target_nll" else
                   "ICLR OpenNMT checkpoint-native label-free route reward"
               ),
               "target_labels_used": args.router_reward in {"target_nll", "target_beam_rank"},
               "feature_source": (
                   "source-reference-free encoder cosine/pair features v2"
                   if args.router_feature_mode == "source_reference_free"
                   else "checkpoint-aware source token statistics"
               ),
               "feature_mode": args.router_feature_mode,
               "feature_contract_version": "source_reference_free_v2",
               "tokenization_policy": {
                   "derived_shift": "m",
                   "ms_marker": "E0Pos",
                   "spectral_numeric_tokens": (
                       "nearest source-vocabulary value"
                   if router_policy == "nearest" else "exact source-vocabulary lookup"
                   ),
                   "formula_tokens": "exact spelling",
                   "numeric_token_policy": router_policy,
                   "router_source": str(_router_source(args)),
               }}
    out = args.run_dir / "router_native.pt"
    torch.save(payload, out)
    route_payload = json.loads(route_path.read_text(encoding="utf-8"))
    for row in route_payload["routes"]:
        update = routes[int(row["source_row_index"])]
        row["router_selected_action_index"] = update["router_selected_action_index"]
        row["router_selected_modalities"] = update["router_selected_modalities"]
    route_payload["router_training"] = {"checkpoint": str(out), "history": history,
                                         "target_labels_used": args.router_reward in {"target_nll", "target_beam_rank"},
                                         "reward_source": payload["reward_source"],
                                         "feature_source": payload["feature_source"],
                                         "feature_mode": args.router_feature_mode,
                                         "feature_contract_version": "source_reference_free_v2",
                                         "router_target_mode": args.router_target_mode,
                                         "oracle_action_counts": torch.bincount(
                                             oracle_actions, minlength=len(ACTION_SUBSETS)
                                         ).tolist(),
                                         "oracle_temperature": float(args.router_oracle_temperature),
                                         "tokenization_policy": payload["tokenization_policy"]}
    route_path.write_text(json.dumps(route_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _refresh_teacher_pseudo_on_clean_source(
    args: argparse.Namespace,
    rows: Sequence[Row],
    routes: dict[int, dict[str, Any]],
    route_path: Path,
    model: Any,
    vocabs: Mapping[str, Any],
    device: torch.device,
) -> None:
    """Regenerate Stage-1 teacher pseudo labels on clean rows.

    Route rewards may come from an augmented ``--router-src``.  The selected
    action is retained, but the pseudo sequence used to train the student must
    be produced from the clean Stage-1 source contract.
    """
    if _router_source(args) == args.src:
        return
    model.eval()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        for row in batch:
            route_row = routes[row.index]
            selected = tuple(
                route_row.get("router_selected_modalities", route_row["selected_modalities"])
            )
            source, lengths = _batch_src(
                [row], vocabs, [selected], device, args.numeric_token_policy
            )
            pseudo, pseudo_score = _greedy(
                model, source, lengths, vocabs, args.max_target_length
            )
            route_row["teacher_pseudo_token_ids"] = [
                int(value) for value in pseudo[0, :, 0].cpu()
            ]
            route_row["teacher_pseudo_smiles"] = _decode(
                pseudo[0, :, 0].cpu().tolist(), vocabs["tgt"]
            )
            route_row["teacher_pseudo_score"] = float(pseudo_score[0].cpu())
    payload = json.loads(route_path.read_text(encoding="utf-8"))
    payload["teacher_pseudo_source"] = str(args.src)
    payload["teacher_pseudo_token_policy"] = args.numeric_token_policy
    for item in payload["routes"]:
        updated = routes[int(item["source_row_index"])]
        for key in (
            "teacher_pseudo_token_ids", "teacher_pseudo_smiles", "teacher_pseudo_score"
        ):
            item[key] = updated[key]
    route_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _save_model(model: Any, path: Path, epoch: int, objective: float, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                "objective": objective, "history": history}, path)


def _load_state(model: Any, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)
    return payload


def _train_stage1(args: argparse.Namespace, rows: list[Row], routes: Mapping[int, Mapping[str, Any]],
                  model: Any, vocabs: Mapping[str, Any], device: torch.device) -> Path:
    if args.epochs == 0 and args.early_stop_patience < 1:
        raise ValueError("unbounded Stage 1 requires --early-stop-patience > 0")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history: list[dict[str, Any]] = []
    best = float("inf"); stale = 0; best_path = args.run_dir / "stage1_best.pt"
    max_epochs = args.epochs if args.epochs > 0 else 10_000_000
    split = max(1, int(0.8 * len(rows)))
    train_rows, valid_rows = rows[:split], rows[split:]
    for epoch in range(1, max_epochs + 1):
        losses = []
        for start in range(0, len(train_rows), args.batch_size):
            batch = train_rows[start:start + args.batch_size]
            selected = [tuple(routes[row.index].get("router_selected_modalities", routes[row.index]["selected_modalities"])) for row in batch]
            source_t, lengths_t = _batch_src(batch, vocabs, selected, device, args.numeric_token_policy)
            full = tuple(name for name in MODALITIES if name in MODEL_SUPPORT[args.checkpoint.stem])
            source_s, lengths_s = _batch_src(batch, vocabs, [full] * len(batch), device, args.numeric_token_policy)
            # Route manifests already contain target-vocabulary IDs. Preserve
            # them directly instead of interpreting IDs as SMILES tokens.
            pseudo = [list(map(int, routes[row.index]["teacher_pseudo_token_ids"])) for row in batch]
            pseudo = [value[:args.max_target_length] for value in pseudo]
            target = _pad_targets(pseudo, _tok_id(vocabs["tgt"], "<blank>"), device)
            with torch.no_grad():
                _forward_nll(model, source_t, lengths_t, target, _tok_id(vocabs["tgt"], "<blank>"))
            optimizer.zero_grad(set_to_none=True)
            loss = _forward_nll(model, source_s, lengths_s, target, _tok_id(vocabs["tgt"], "<blank>")).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_objective = sum(losses) / max(1, len(losses))
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for start in range(0, len(valid_rows), args.batch_size):
                batch = valid_rows[start:start + args.batch_size]
                selected = [tuple(routes[row.index].get("router_selected_modalities", routes[row.index]["selected_modalities"])) for row in batch]
                source_s, lengths_s = _batch_src(batch, vocabs, [tuple(name for name in MODALITIES if name in MODEL_SUPPORT[args.checkpoint.stem])] * len(batch), device, args.numeric_token_policy)
                pseudo = [list(map(int, routes[row.index]["teacher_pseudo_token_ids"]))[:args.max_target_length] for row in batch]
                target = _pad_targets(pseudo, _tok_id(vocabs["tgt"], "<blank>"), device)
                valid_losses.append(float(_forward_nll(model, source_s, lengths_s, target, _tok_id(vocabs["tgt"], "<blank>")).mean().cpu()))
        valid_objective = sum(valid_losses) / max(1, len(valid_losses))
        history.append({"epoch": epoch, "pseudo_ce": train_objective, "validation_pseudo_ce": valid_objective})
        if valid_objective < best - args.early_stop_min_delta:
            best, stale = valid_objective, 0
            _save_model(model, best_path, epoch, valid_objective, history)
        else:
            stale += 1
        _save_model(model, args.run_dir / f"stage1_epoch_{epoch}.pt", epoch, valid_objective, history)
        print(json.dumps({"stage": "stage1", "epoch": epoch, "pseudo_ce": train_objective, "validation_pseudo_ce": valid_objective, "stale": stale}), flush=True)
        model.train()
        if epoch >= args.early_stop_min_epochs and stale >= args.early_stop_patience:
            break
    best_epoch = next(
        (int(item["epoch"]) for item in history
         if abs(float(item["validation_pseudo_ce"]) - best) <= 1e-12),
        None,
    )
    (args.run_dir / "stage1_manifest.json").write_text(
        json.dumps({"complete": True, "epochs": len(history),
                    "best_epoch": best_epoch,
                    "best_validation_pseudo_ce": best,
                    "history": history}, indent=2),
        encoding="utf-8",
    )
    return best_path


def _select_stability_paths(
    paths: Iterable[Path], *, interval: int = 3, count: int = 4
) -> list[Path]:
    """Select the latest interval checkpoints using numeric epoch ordering."""
    ordered = sorted(paths, key=_epoch_number)
    if interval < 1 or count < 2:
        raise ValueError("stable pseudo selection requires interval >= 1 and count >= 2")
    interval_paths = [path for path in ordered if _epoch_number(path) % interval == 0]
    # The original protocol uses epochs 3/6/9/12.  An early-stopped smoke run
    # may not reach two interval snapshots; only then fall back to its latest
    # checkpoints so the diagnostic can still terminate explicitly.
    selected = interval_paths[-count:] if len(interval_paths) >= 2 else ordered[-count:]
    if len(selected) < 2:
        raise ValueError("stable pseudo labels require at least two checkpoint predictions")
    return selected


def _stable_pseudo(
    rows: Sequence[Row],
    paths: Sequence[Path],
    target_vocab: Any,
    report_path: Path | None = None,
) -> dict[int, list[int]]:
    """Keep valid, Formula-matched canonical consensus across Full snapshots."""
    if len(paths) < 2:
        raise ValueError("stable pseudo labels require at least two prediction banks")
    banks = [_load_routes(path) for path in paths]
    stable: dict[int, list[int]] = {}
    counts: Counter[str] = Counter()
    chosen_bank = len(banks) // 2
    for row in rows:
        values = [
            bank.get(row.index, {}).get("teacher_pseudo_token_ids")
            for bank in banks
        ]
        if any(value is None for value in values):
            counts["missing"] += 1
            continue
        texts = [_decode(list(map(int, value)), target_vocab) for value in values]
        canonical = [_canonical_smiles(text) for text in texts]
        if any(value is None for value in canonical):
            counts["invalid"] += 1
            continue
        if len(set(canonical)) != 1:
            counts["checkpoint_disagreement"] += 1
            continue
        counts["canonical_consensus"] += 1
        if _smiles_formula_signature(canonical[0]) != _formula_signature(row.formula):
            counts["formula_mismatch"] += 1
            continue
        stable[row.index] = list(map(int, values[chosen_bank]))
        counts["accepted"] += 1
    if report_path is not None:
        report_path.write_text(
            json.dumps({
                "prediction_paths": [str(path) for path in paths],
                "selected_epochs": [_epoch_number(path) for path in paths],
                "rows": len(rows),
                **dict(counts),
                "selection": (
                    "Full-view canonical consensus AND RDKit valid AND "
                    "observed Formula match"
                ),
                "ground_truth_loaded": False,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return stable


def _train_stage2(args: argparse.Namespace, rows: list[Row], model: Any, vocabs: Mapping[str, Any],
                  device: torch.device, pseudo: Mapping[int, Sequence[int]]) -> Path:
    if not pseudo:
        raise ValueError("Stage 2 has no stable pseudo labels")
    if args.epochs == 0 and args.early_stop_patience < 1:
        raise ValueError("unbounded Stage 2 requires --early-stop-patience > 0")
    model.train(); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    support = MODEL_SUPPORT[args.checkpoint.stem]
    history = []; best = float("inf"); stale = 0; best_path = args.run_dir / "stage2_best.pt"
    max_epochs = args.epochs if args.epochs > 0 else 10_000_000
    eligible = [row for row in rows if row.index in pseudo]
    split = max(1, int(0.8 * len(eligible)))
    train_rows, valid_rows = eligible[:split], eligible[split:]
    for epoch in range(1, max_epochs + 1):
        losses = []
        shuffled = list(train_rows); random.Random(args.seed + epoch).shuffle(shuffled)
        for start in range(0, len(shuffled), args.batch_size):
            batch = [row for row in shuffled[start:start + args.batch_size] if row.index in pseudo]
            if not batch: continue
            dropped = []
            for row in batch:
                available = [name for name in MODALITIES if name in support and name in row.parts]
                keep = set(available)
                drop_count = min(len(available) - 1, random.Random(args.seed + epoch * 100003 + row.index).randint(1, 3))
                if drop_count > 0:
                    keep -= set(random.Random(args.seed + epoch * 100003 + row.index).sample(available, drop_count))
                dropped.append(tuple(sorted(keep, key=MODALITIES.index)))
            source, lengths = _batch_src(batch, vocabs, dropped, device, args.numeric_token_policy)
            target = _pad_targets([list(pseudo[row.index]) for row in batch], _tok_id(vocabs["tgt"], "<blank>"), device)
            optimizer.zero_grad(set_to_none=True)
            loss = _forward_nll(model, source, lengths, target, _tok_id(vocabs["tgt"], "<blank>")).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_objective = sum(losses) / max(1, len(losses))
        model.eval(); valid_losses = []
        with torch.no_grad():
            for start in range(0, len(valid_rows), args.batch_size):
                batch = valid_rows[start:start + args.batch_size]
                available = [name for name in MODALITIES if name in support and name in batch[0].parts]
                source, lengths = _batch_src(batch, vocabs, [tuple(available)] * len(batch), device, args.numeric_token_policy)
                target = _pad_targets([list(pseudo[row.index]) for row in batch], _tok_id(vocabs["tgt"], "<blank>"), device)
                valid_losses.append(float(_forward_nll(model, source, lengths, target, _tok_id(vocabs["tgt"], "<blank>")).mean().cpu()))
        valid_objective = sum(valid_losses) / max(1, len(valid_losses))
        history.append({"epoch": epoch, "pseudo_ce": train_objective, "validation_pseudo_ce": valid_objective})
        if valid_objective < best - args.early_stop_min_delta:
            best, stale = valid_objective, 0; _save_model(model, best_path, epoch, valid_objective, history)
        else: stale += 1
        _save_model(model, args.run_dir / f"stage2_epoch_{epoch}.pt", epoch, valid_objective, history)
        print(json.dumps({"stage": "stage2", "epoch": epoch, "pseudo_ce": train_objective, "validation_pseudo_ce": valid_objective, "stale": stale}), flush=True)
        model.train()
        if epoch >= args.early_stop_min_epochs and stale >= args.early_stop_patience: break
    best_epoch = next(
        (int(item["epoch"]) for item in history
         if abs(float(item["validation_pseudo_ce"]) - best) <= 1e-12),
        None,
    )
    (args.run_dir / "stage2_manifest.json").write_text(
        json.dumps({"complete": True, "epochs": len(history),
                    "stable_rows": len(pseudo), "best_epoch": best_epoch,
                    "best_validation_pseudo_ce": best,
                    "history": history}, indent=2),
        encoding="utf-8",
    )
    return best_path


def pipeline(args: argparse.Namespace) -> None:
    random.seed(args.seed); torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    route_path = args.route_manifest or args.run_dir / "route_manifest.json"
    if not route_path.exists() or not _route_manifest_compatible(route_path, args, args.limit):
        route_path = route(args)
    rows = _parse_rows(args.src, None, args.limit, ms_energy_mode=args.ms_energy_mode)
    router_src = _router_source(args)
    router_policy = _router_token_policy(args)
    router_rows = rows if router_src == args.src else _parse_rows(
        router_src, None, args.limit, ms_energy_mode=args.ms_energy_mode
    )
    _assert_router_alignment(rows, router_rows)
    routes = _load_routes(route_path)
    model, vocabs, device = _load_onmt(args)
    _train_native_router(args, router_rows, route_path, routes, vocabs, model, device)
    # Router rewards are generated on augmented views. For student training,
    # preserve the same selected modalities but regenerate the teacher's
    # pseudo sequence on the clean source rows.
    _refresh_teacher_pseudo_on_clean_source(
        args, rows, routes, route_path, model, vocabs, device
    )
    stage1 = _train_stage1(args, rows, routes, model, vocabs, device)
    _load_state(model, stage1)
    prediction_paths = []
    stability_checkpoints = _select_stability_paths(
        args.run_dir.glob("stage1_epoch_*.pt")
    )
    for checkpoint in stability_checkpoints:
        payload = _load_state(model, checkpoint)
        pred_rows = []
        model.eval()
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            full = tuple(name for name in MODALITIES if name in MODEL_SUPPORT[args.checkpoint.stem])
            source, lengths = _batch_src(batch, vocabs, [full] * len(batch), device, args.numeric_token_policy)
            seq, score = _greedy(model, source, lengths, vocabs, args.max_target_length)
            for local, row in enumerate(batch):
                pred_rows.append({"source_row_index": row.index, "teacher_pseudo_token_ids": [int(v) for v in seq[local, :, 0].cpu()]})
        path = args.run_dir / f"{checkpoint.stem}.json"
        path.write_text(json.dumps({"routes": pred_rows}, indent=2), encoding="utf-8"); prediction_paths.append(path)
    pseudo = _stable_pseudo(
        rows, prediction_paths, vocabs["tgt"],
        args.run_dir / "stable_pseudo_manifest.json",
    )
    _load_state(model, stage1)
    _train_stage2(args, rows, model, vocabs, device, pseudo)
    (args.run_dir / "pipeline.complete.json").write_text(json.dumps({
        "complete": True, "checkpoint": str(args.checkpoint),
        "stable_rows": len(pseudo),
        "stable_checkpoint_epochs": [
            _epoch_number(path) for path in stability_checkpoints
        ],
    }, indent=2), encoding="utf-8")


def main() -> None:
    args = _args()
    if args.limit < 1 or args.batch_size < 1: raise ValueError("limit and batch-size must be positive")
    if args.command == "route": route(args); return
    if args.command == "pipeline": pipeline(args); return
    rows = _parse_rows(args.src, None, args.limit, ms_energy_mode=args.ms_energy_mode)
    router_src = _router_source(args)
    router_rows = rows if router_src == args.src else _parse_rows(
        router_src, None, args.limit, ms_energy_mode=args.ms_energy_mode
    )
    _assert_router_alignment(rows, router_rows)
    if args.command == "router":
        path = args.route_manifest or args.run_dir / "route_manifest.json"
        if not path.exists() or not _route_manifest_compatible(path, args, args.limit): path = route(args)
        model, vocabs, device = _load_onmt(args)
        _train_native_router(args, router_rows, path, _load_routes(path), vocabs, model, device); return
    model, vocabs, device = _load_onmt(args)
    if args.command == "stage1":
        path = args.route_manifest or args.run_dir / "route_manifest.json"
        if not path.exists() or not _route_manifest_compatible(path, args, args.limit): path = route(args)
        routes = _load_routes(path)
        _refresh_teacher_pseudo_on_clean_source(args, rows, routes, path, model, vocabs, device)
        _train_stage1(args, rows, routes, model, vocabs, device); return
    if not args.stage1_checkpoint: raise ValueError("stage2 requires --stage1-checkpoint")
    _load_state(model, args.stage1_checkpoint)
    predictions = _select_stability_paths(
        args.run_dir.glob("stage1_epoch_*.json")
    )
    if not predictions: raise ValueError("stage2 requires stage1 prediction JSON files")
    pseudo = _stable_pseudo(
        rows, predictions, vocabs["tgt"],
        args.run_dir / "stable_pseudo_manifest.json",
    )
    _train_stage2(args, rows, model, vocabs, device, pseudo)


if __name__ == "__main__": main()

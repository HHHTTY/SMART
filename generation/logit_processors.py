"""Formula-aware logit processing for SMILES generation.

The processor deliberately constrains only atom counts.  SMILES grammar is
handled by the model itself; trying to parse every incomplete prefix as a
complete molecule makes ordinary branch/ring prefixes look like zero atoms
and can dead-end beam search.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor


class GuidedFormulaProcessor(LogitsProcessor):
    """Constrain SMILES beam search with an observed molecular formula.

    Formula constraints are applied incrementally using the lexical SMILES
    atoms in each token.  RDKit is used only to decide whether a prefix is a
    complete valid molecule and whether its *full* formula matches the target.
    This distinction is important because most useful beam prefixes are not
    valid standalone SMILES yet (for example ``c1cc(``).
    """

    # Full periodic-table symbols. The old implementation stopped applying
    # look-ahead constraints after its first nine entries, silently allowing
    # over-budget Si/Na/metal atoms. Most of these are represented in brackets
    # in SMILES, but they still need exact formula accounting.
    _ELEMENTS = (
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    )
    _ELEMENT_INDEX = {element: index for index, element in enumerate(_ELEMENTS)}

    # Bracket atoms must be matched before unbracketed atoms.  The remaining
    # alternatives are ordered longest-first so ``Cl`` is never read as C.
    _SMILES_ATOM_RE = re.compile(
        r"\[[^\]]*\]|Br|Cl|Si|Se|As|Na|Li|Al|Sn|Te|Zn|Pb|Hg|Ag|Ge|Ti|Sb|Mg|Bi|Ca|Fe|Cu|Mn|Co|Ni|"
        r"K|B|C|N|O|P|S|F|I|b|c|n|o|p|s"
    )
    _FORMULA_PART_RE = re.compile(r"([A-Z][a-z]?)([0-9]*)")
    _BRACKET_ELEMENT_RE = re.compile(
        r"^(?:\d+)?([A-Z][a-z]?|[bcnops])(?:H([0-9]*))?"
    )
    _UNBRACKETED_ELEMENTS = {
        "B",
        "C",
        "N",
        "O",
        "P",
        "S",
        "F",
        "Cl",
        "Br",
        "I",
        "b",
        "c",
        "n",
        "o",
        "p",
        "s",
    }

    def __init__(
        self, n_beams: int, chemical_formula: List[str], target_tokenizer: AutoTokenizer
    ):
        super().__init__()
        if n_beams < 1:
            raise ValueError("n_beams must be at least 1")
        if not chemical_formula:
            raise ValueError("chemical_formula must contain at least one formula")

        self.n_beams = int(n_beams)
        self.target_tokenizer = target_tokenizer
        self.eos_token_id = getattr(target_tokenizer, "eos_token_id", None)
        self.vocab_size = int(getattr(target_tokenizer, "vocab_size", 0))
        if self.vocab_size < 1:
            vocab = getattr(target_tokenizer, "vocab", {})
            self.vocab_size = max((int(value) for value in vocab.values()), default=-1) + 1
        if self.vocab_size < 1:
            raise ValueError("target_tokenizer must expose a non-empty vocabulary")

        # Formula targets are kept as dictionaries so unsupported elements are
        # still compared exactly instead of disappearing from a fixed vector.
        self._target_counts = [self._parse_formula(formula) for formula in chemical_formula]
        self._prefix_cache: Dict[str, tuple[bool, Dict[str, int], Dict[str, int]]] = {}
        self._token_atom_delta = self._build_token_atom_delta()

        # Retain this public attribute for compatibility with code that used
        # the original implementation for diagnostics.
        self.chemical_formula_beams = np.repeat(
            np.stack([self._encode_formula(counts) for counts in self._target_counts], axis=0),
            self.n_beams,
            axis=0,
        )

    @classmethod
    def _canonical_element(cls, element: str) -> str:
        if element in {"b", "c", "n", "o", "p", "s"}:
            return element.upper()
        return element

    @classmethod
    def _parse_formula(cls, formula: str) -> Dict[str, int]:
        text = "" if formula is None else str(formula).strip().replace(" ", "")
        if not text:
            raise ValueError("Molecular formula cannot be empty")
        # RDKit appends formal charge as ``+``, ``-``, ``2+``, etc.  Formula
        # guidance constrains elemental composition, so charge is ignored.
        text = re.sub(r"(?:[+-]\d*|\d+[+-])$", "", text)
        if not text:
            raise ValueError(f"Invalid molecular formula: {formula!r}")
        counts: Dict[str, int] = {}
        position = 0
        for match in cls._FORMULA_PART_RE.finditer(text):
            if match.start() != position:
                raise ValueError(f"Invalid molecular formula: {formula!r}")
            element, count_text = match.groups()
            count = int(count_text) if count_text else 1
            if count < 0:
                raise ValueError(f"Invalid atom count in molecular formula: {formula!r}")
            counts[element] = counts.get(element, 0) + count
            position = match.end()
        if position != len(text):
            raise ValueError(f"Invalid molecular formula: {formula!r}")
        return counts

    @classmethod
    def _encode_formula(cls, counts: Mapping[str, int]) -> np.ndarray:
        encoded = np.zeros(len(cls._ELEMENTS), dtype=np.int16)
        for element, count in counts.items():
            index = cls._ELEMENT_INDEX.get(element)
            if index is not None:
                encoded[index] = count
        return encoded

    @classmethod
    def _atom_counts_from_token(cls, token: object) -> Dict[str, int]:
        """Return the atoms contributed by one lexical SMILES token."""
        text = "" if token is None else str(token).strip()
        # Handle common subword markers used by BPE/SentencePiece tokenizers.
        while text.startswith(("##", "Ġ", "▁")):
            text = text[1:] if text.startswith(("Ġ", "▁")) else text[2:]
        if text in {"", "<pad>", "<unk>", "<bos>", "<eos>"}:
            return {}
        if text.startswith("[") and text.endswith("]"):
            body = text[1:-1]
            match = cls._BRACKET_ELEMENT_RE.match(body)
            if match is None:
                return {}
            element, hydrogen_count = match.groups()
            element = cls._canonical_element(element)
            counts = {element: 1}
            if hydrogen_count is not None:
                counts["H"] = int(hydrogen_count) if hydrogen_count else 1
            else:
                # Stereochemistry and isotope annotations can occur between
                # the atom symbol and an explicit H, e.g. ``[C@@H]``.
                trailing_h = re.search(r"H([0-9]*)", body[match.end() :])
                if trailing_h is not None:
                    counts["H"] = int(trailing_h.group(1)) if trailing_h.group(1) else 1
            return counts
        if text in cls._UNBRACKETED_ELEMENTS:
            return {cls._canonical_element(text): 1}
        return {}

    def _build_token_atom_delta(self) -> np.ndarray:
        delta = np.zeros((self.vocab_size, len(self._ELEMENTS)), dtype=np.int16)
        vocab = getattr(self.target_tokenizer, "vocab", {})
        for token, token_id in vocab.items():
            token_id = int(token_id)
            if token_id < 0 or token_id >= self.vocab_size:
                continue
            for element, count in self._atom_counts_from_token(token).items():
                index = self._ELEMENT_INDEX.get(element)
                if index is not None:
                    delta[token_id, index] += count
        return delta

    @classmethod
    def _count_smiles_atoms(cls, smiles: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        # Remove special tokens defensively.  ``skip_special_tokens=True``
        # normally does this, but unknown tokens can be tokenizer-dependent.
        text = re.sub(r"<(?:pad|unk|bos|eos)>", "", smiles).replace(" ", "")
        for match in cls._SMILES_ATOM_RE.finditer(text):
            for element, count in cls._atom_counts_from_token(match.group(0)).items():
                counts[element] = counts.get(element, 0) + count
        return counts

    @classmethod
    def _counts_equal(cls, left: Mapping[str, int], right: Mapping[str, int]) -> bool:
        return {key: value for key, value in left.items() if value} == {
            key: value for key, value in right.items() if value
        }

    @classmethod
    def _counts_exceed(cls, current: Mapping[str, int], target: Mapping[str, int]) -> bool:
        return any(value > target.get(element, 0) for element, value in current.items())

    def make_formula_encoding(self, formula: str) -> np.ndarray:
        """Return the legacy fixed-width encoding used by older callers."""
        return self._encode_formula(self._parse_formula(formula))

    def _prefix_state(self, smiles: str) -> tuple[bool, Dict[str, int], Dict[str, int]]:
        normalized = smiles.replace(" ", "")
        cached = self._prefix_cache.get(normalized)
        if cached is not None:
            return cached

        explicit_counts = self._count_smiles_atoms(normalized)
        molecule = Chem.MolFromSmiles(normalized) if normalized else None
        if molecule is None:
            state = (False, explicit_counts, {})
        else:
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            state = (True, explicit_counts, self._parse_formula(formula) if formula else {})
        self._prefix_cache[normalized] = state
        return state

    def _target_for_row(self, row: int, n_rows: int) -> Dict[str, int]:
        batch_size = len(self._target_counts)
        if n_rows % batch_size != 0:
            raise ValueError(
                "Formula-guided generation received a logits batch that is not "
                f"a multiple of the formula batch ({n_rows} vs {batch_size})."
            )
        beams = n_rows // batch_size
        return self._target_counts[row // beams]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply incremental atom-budget masks without dead-ending decoding."""
        if scores.ndim != 2 or input_ids.ndim != 2 or scores.shape[0] != input_ids.shape[0]:
            raise ValueError("input_ids and scores must be 2-D tensors with matching batch size")
        if scores.shape[1] > self._token_atom_delta.shape[0]:
            # This is unusual, but keeps the processor safe when a model has
            # resized its decoder vocabulary after the tokenizer was created.
            padded = np.zeros((scores.shape[1], len(self._ELEMENTS)), dtype=np.int16)
            padded[: self._token_atom_delta.shape[0]] = self._token_atom_delta
            token_delta = padded
        else:
            token_delta = self._token_atom_delta[: scores.shape[1]]

        RDLogger.DisableLog("rdApp.*")
        decoded = self.target_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        for row, raw_smiles in enumerate(decoded):
            smiles = raw_smiles.replace(" ", "")
            valid, explicit_counts, complete_counts = self._prefix_state(smiles)
            target = self._target_for_row(row, input_ids.shape[0])

            # A valid, exact-formula prefix is already a complete answer.  In
            # this one case EOS is forced so beam search cannot add a trailing
            # branch/ring token and spoil the valid molecule.
            if (
                valid
                and self._counts_equal(complete_counts, target)
                and self.eos_token_id is not None
            ):
                eos = int(self.eos_token_id)
                if eos < scores.shape[1]:
                    scores[row].fill_(-float("inf"))
                    scores[row, eos] = 0.0
                continue

            # An unfinished branch, ring, or charge expression is commonly
            # invalid to RDKit even though it is a perfectly recoverable
            # SMILES prefix.  Leave such rows untouched: applying formula
            # masks while the parser cannot see the pending structure changes
            # the model's grammar search and can make every continuation
            # invalid.
            if not valid:
                continue

            # Once the model has exceeded a target formula there is no legal
            # formula-consistent continuation.  Fall back to the model's
            # ordinary decoding instead of masking the row into a dead end.
            if self._counts_exceed(explicit_counts, target):
                continue

            # Mask only atom tokens that would exceed the target.  Operators,
            # ring labels, branches, and other syntax remain available, which
            # lets incomplete prefixes recover naturally.
            current = np.zeros(len(self._ELEMENTS), dtype=np.int16)
            for element, count in explicit_counts.items():
                index = self._ELEMENT_INDEX.get(element)
                if index is not None:
                    current[index] = count
            target_vector = np.zeros(len(self._ELEMENTS), dtype=np.int16)
            for element, count in target.items():
                index = self._ELEMENT_INDEX.get(element)
                if index is not None:
                    target_vector[index] = count
            too_large = np.any(current[None, :] + token_delta > target_vector[None, :], axis=1)
            atom_tokens = np.any(token_delta != 0, axis=1)
            masked_ids = np.flatnonzero(too_large & atom_tokens)
            if masked_ids.size:
                scores[row, torch.as_tensor(masked_ids, device=scores.device)] = -float("inf")

            # EOS is intentionally left at the model's score for mismatched or
            # incomplete prefixes.  The old blanket EOS mask pushed those
            # beams to max_length, where HF appended EOS to invalid strings.

        return scores

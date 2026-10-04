#!/usr/bin/env python3
"""Build a checkpoint-compatible numeric preprocessor for Chemotion.

The epoch-24 checkpoint owns the SDBS tokenizer vocabulary and embedding
sizes.  This wrapper keeps those weights unchanged and maps an unseen numeric
NMR token to the nearest numeric token already present in the SDBS vocabulary.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


_NUMERIC_TOKEN = re.compile(r"^-?\d+(?:\.\d+)?$")
_FORMULA_TOKEN = re.compile(r"^([A-Z][a-z]?)(\d*)$")


def _numeric_vocab(tokenizer: Any) -> tuple[set[str], list[tuple[float, str]]]:
    vocab = tokenizer.get_vocab()
    exact = set(vocab)
    numeric: list[tuple[float, str]] = []
    for token in vocab:
        if not _NUMERIC_TOKEN.fullmatch(token):
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value):
            numeric.append((value, token))
    numeric.sort(key=lambda item: item[0])
    return exact, numeric


def _nearest_numeric(token: str, numeric: list[tuple[float, str]]) -> str:
    if not numeric or not _NUMERIC_TOKEN.fullmatch(token):
        return token
    value = float(token)
    # Numeric vocabularies are short; this avoids changing token IDs or model
    # embedding dimensions while preserving the closest learned shift.
    return min(numeric, key=lambda item: (abs(item[0] - value), item[0]))[1]


def _map_text(text: str, exact: set[str], numeric: list[tuple[float, str]]) -> str:
    tokens = text.split()
    return " ".join(
        token if token in exact else _nearest_numeric(token, numeric)
        for token in tokens
    )


def _formula_vocab(tokenizer: Any) -> dict[str, list[tuple[int, str]]]:
    """Index learned Formula tokens by element for count-compatible mapping."""
    result: dict[str, list[tuple[int, str]]] = {}
    for token in tokenizer.get_vocab():
        match = _FORMULA_TOKEN.fullmatch(str(token))
        if not match:
            continue
        count = int(match.group(2) or 1)
        result.setdefault(match.group(1), []).append((count, str(token)))
    for values in result.values():
        values.sort()
    return result


def _map_formula(text: str, exact: set[str], by_element: dict[str, list[tuple[int, str]]]) -> tuple[str, list[dict[str, Any]]]:
    """Map unseen element-count tokens without changing tokenizer IDs."""
    mapped: list[str] = []
    audit: list[dict[str, Any]] = []
    for token in re.findall(r"[A-Z][a-z]?\d*", str(text)):
        if token in exact:
            mapped.append(token)
            continue
        match = _FORMULA_TOKEN.fullmatch(token)
        candidates = by_element.get(match.group(1), []) if match else []
        if candidates:
            count = int(match.group(2) or 1)
            replacement = min(candidates, key=lambda item: (abs(item[0] - count), item[0]))[1]
        else:
            replacement = token
        mapped.append(replacement)
        audit.append({"original": token, "mapped": replacement})
    # Formula tokenizers consume the molecular formula as one compact string;
    # inserting spaces would make every element/count fragment a separate
    # lexical item and can dramatically increase unknown tokens.
    return "".join(mapped), audit


class ChemotionFormulaPreprocessor:
    """Formula wrapper preserving the checkpoint Formula vocabulary."""

    def __init__(self, base: Any):
        self.base = base
        self.tokenizer = base
        self._exact = set(base.get_vocab())
        self._by_element = _formula_vocab(base)
        self.last_audit: list[list[dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        # During pickle restore ``base`` may not exist yet; avoid recursive
        # lookup through this proxy until the wrapped tokenizer is restored.
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __call__(self, formulas: Any = None, *, text: Any = None, **kwargs: Any) -> Any:
        # Text preprocessors are called with ``text=...`` for source-length
        # probing and positionally with a batch during collation.
        value = text if text is not None else formulas
        single = isinstance(value, str)
        values = [value] if single else list(value)
        mapped, audits = zip(*(_map_formula(item, self._exact, self._by_element) for item in values))
        self.last_audit = list(audits)
        return self.base(mapped[0] if single else list(mapped), **kwargs)


class ChemotionCarbonPreprocessor:
    """Carbon preprocessor retaining the original checkpoint vocabulary."""

    def __init__(self, base: Any):
        self.base = base
        self.tokenizer = base.tokenizer
        self.max_sequence_length = base.max_sequence_length
        self._exact, self._numeric = _numeric_vocab(self.tokenizer)

    def __getattr__(self, name: str) -> Any:
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __call__(self, carbon_nmrs: list[list[dict[str, Any]]]) -> Any:
        normalized = []
        for spectrum in carbon_nmrs:
            if spectrum is None:
                normalized.append(None)
                continue
            peaks = []
            for peak in spectrum:
                if not isinstance(peak, dict):
                    continue
                value = peak.get("delta (ppm)", peak.get("ppm", peak.get("shift")))
                if value is None:
                    continue
                peaks.append({"delta (ppm)": float(value), "intensity": peak.get("intensity", 1.0)})
            normalized.append(peaks)
        processed = self.base.process_carbon(normalized)
        mapped = [_map_text(value, self._exact, self._numeric) for value in processed]
        tokenized = self.tokenizer(
            mapped,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        no_data_mask = [value == "" for value in mapped]
        tokenized["attention_mask"][no_data_mask] = 0
        return tokenized


class ChemotionMultipletPreprocessor:
    """1H preprocessor retaining the original checkpoint vocabulary."""

    def __init__(self, base: Any):
        self.base = base
        self.tokenizer = base.tokenizer
        self.max_sequence_length = base.max_sequence_length
        self._exact, self._numeric = _numeric_vocab(self.tokenizer)

    def __getattr__(self, name: str) -> Any:
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __call__(self, multiplets: list[list[dict[str, Any]]]) -> Any:
        normalized = []
        for spectrum in multiplets:
            if spectrum is None:
                normalized.append(None)
                continue
            peaks = []
            for peak in spectrum:
                if not isinstance(peak, dict):
                    continue
                ppm = peak.get("ppm", peak.get("shift", peak.get("centroid")))
                if ppm is None:
                    continue
                ppm = float(ppm)
                start = float(peak.get("rangeMin", peak.get("start", ppm)))
                end = float(peak.get("rangeMax", peak.get("end", ppm)))
                peaks.append({
                    "rangeMax": max(start, end),
                    "rangeMin": min(start, end),
                    "centroid": ppm,
                    "category": str(peak.get("category") or peak.get("multiplicity") or "m"),
                    "nH": peak.get("nH", peak.get("integration", 1)),
                    "j_values": str(peak.get("j_values") or "None"),
                })
            normalized.append(peaks)
        processed, numerical = self.base.process_multiplets(
            normalized,
            self.base.encoding,
            self.base.j_values,
        )
        mapped = [_map_text(value, self._exact, self._numeric) for value in processed]
        tokenized = self.tokenizer(
            mapped,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        if self.base.encoding == "numerical_encoding":
            tokenized["numerical_values"] = self.base.add_padding_numerical_values(
                tokenized, numerical
            )
        no_data_mask = [value == "" for value in mapped]
        tokenized["attention_mask"][no_data_mask] = 0
        return tokenized


class ChemotionMSMSTextPreprocessor:
    """MS/MS text preprocessor with checkpoint-compatible numeric tokens."""

    def __init__(self, base: Any):
        self.base = base
        self.tokenizer = base.tokenizer
        self.max_sequence_length = base.max_sequence_length
        self._exact, self._numeric = _numeric_vocab(self.tokenizer)

    def __getattr__(self, name: str) -> Any:
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __call__(self, msms_spectra: list[Any]) -> Any:
        processed = self.base.process_msms(msms_spectra)
        peak_budget = max(1, (self.max_sequence_length - 2) // 2)
        capped = []
        for value in processed:
            fields = value.split()
            peaks = []
            for index in range(0, len(fields) - 1, 2):
                try:
                    mz, intensity = float(fields[index]), float(fields[index + 1])
                except ValueError:
                    continue
                if math.isfinite(mz) and math.isfinite(intensity):
                    peaks.append((mz, intensity))
            if len(peaks) > peak_budget:
                peaks = sorted(peaks, key=lambda peak: (-peak[1], peak[0]))[:peak_budget]
            peaks.sort(key=lambda peak: peak[0])
            capped.append(" ".join(f"{mz:g} {intensity:g}" for mz, intensity in peaks))
        mapped = [_map_text(value, self._exact, self._numeric) for value in capped]
        tokenized = self.tokenizer(
            mapped,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        no_data_mask = [not self.base._has_numeric_peak(value) for value in msms_spectra]
        tokenized["attention_mask"][no_data_mask] = 0
        return tokenized


def build(input_path: Path, output_path: Path) -> None:
    data_config, preprocessors = pd.read_pickle(input_path)
    preprocessors = dict(preprocessors)
    preprocessors["CNMR"] = ChemotionCarbonPreprocessor(preprocessors["CNMR"])
    preprocessors["HNMR"] = ChemotionMultipletPreprocessor(preprocessors["HNMR"])
    preprocessors["MSMS"] = ChemotionMSMSTextPreprocessor(preprocessors["MSMS"])
    preprocessors["Formula"] = ChemotionFormulaPreprocessor(preprocessors["Formula"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle((data_config, preprocessors), output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

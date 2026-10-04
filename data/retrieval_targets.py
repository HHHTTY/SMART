"""Deterministic labels for task-specialized multimodal retrieval heads."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

NMR_ENVIRONMENT_SMARTS = (
    "[cH]", "[cH0]", "[C;X4;H3]", "[C;X4;H2]", "[C;X4;H1]", "[C;X4;H0]",
    "[C;X3](=O)", "[C;X3](=N)", "[C]#N", "[C]#C", "[C;X4]-[O]", "[C;X4]-[N]",
    "[C;X4]-[S]", "[C;X4]-[F,Cl,Br,I]", "[c]-[O]", "[c]-[N]", "[c]-[F,Cl,Br,I]",
    "[O;H1]", "[N;H1,H2]", "[S;H1]", "[nH]", "[nH0]", "[o]", "[s]",
    "[C;r3]", "[C;r4]", "[C;r5]", "[C;r6]", "[c;r5]", "[c;r6]", "[C]=[C]", "[C]=[N]",
)

IR_FUNCTIONAL_GROUP_SMARTS = (
    "[O;H1]", "[N;H1,H2]", "[C;X3](=O)[#6]", "[C;X3](=O)[O;H1]",
    "[C;X3](=O)O[#6]", "[C;X3](=O)N", "[C;X3](=O)[F,Cl,Br,I]", "[C]#N",
    "[C]#C", "[N+](=O)[O-]", "S(=O)(=O)", "[S;X2]", "P(=O)", "[O;X2]([#6])[#6]",
    "[N;X3]([#6])([#6])[#6]", "[nH]", "[c]1[c][c][c][c][c]1", "[F,Cl,Br,I]",
    "[C]=[C]", "[C]=[N]", "[N]=[N]", "[O]-[O]", "[S]-[H]", "[Si]",
    "[B]", "[C;X4]-[O;X2]-[C;X4]", "[C;X3](=S)", "[N]-[O]", "[C]-[F]",
    "[C]-[Cl]", "[C]-[Br]", "[C]-[I]",
)


def _compile_smarts(patterns: Sequence[str]) -> tuple[Chem.Mol, ...]:
    compiled = tuple(Chem.MolFromSmarts(pattern) for pattern in patterns)
    if any(pattern is None for pattern in compiled):
        raise ValueError("Invalid SMARTS pattern in retrieval target definition.")
    return compiled  # type: ignore[return-value]


_NMR_PATTERNS = _compile_smarts(NMR_ENVIRONMENT_SMARTS)
_IR_PATTERNS = _compile_smarts(IR_FUNCTIONAL_GROUP_SMARTS)


def molecular_fingerprint(smiles: str, n_bits: int = 128, radius: int = 2) -> np.ndarray:
    """Return a binary Morgan fingerprint for a labeled source molecule."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint = generator.GetFingerprint(molecule)
    output = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, output)
    return output


def _smarts_presence(smiles: str, patterns: Sequence[Chem.Mol]) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return np.asarray(
        [float(molecule.HasSubstructMatch(pattern)) for pattern in patterns],
        dtype=np.float32,
    )


def nmr_atom_environment(smiles: str) -> np.ndarray:
    """Return 32 graph-derived environments observable in 1D 1H/13C NMR."""
    return _smarts_presence(smiles, _NMR_PATTERNS)


def ir_functional_groups(smiles: str) -> np.ndarray:
    """Return 32 IR-relevant functional-group labels."""
    return _smarts_presence(smiles, _IR_PATTERNS)


def _coerce_peaks(spectrum: Any) -> Iterable[tuple[float, float]]:
    if isinstance(spectrum, str):
        spectrum = json.loads(spectrum)
    if isinstance(spectrum, Mapping):
        spectrum = spectrum.get("peaks", spectrum.get("spectrum", []))
    if spectrum is None:
        return
    for peak in spectrum:
        if isinstance(peak, Mapping):
            position = peak.get("position", peak.get("mz", peak.get("m/z")))
            intensity = peak.get("intensity", peak.get("i", 1.0))
        else:
            position, intensity = peak[:2]
        if position is None or intensity is None:
            continue
        position = float(position)
        intensity = float(intensity)
        if math.isfinite(position) and math.isfinite(intensity) and position >= 0:
            yield position, max(intensity, 0.0)


def _stable_feature_bit(feature: str, n_bits: int) -> int:
    digest = hashlib.blake2b(
        feature.encode("utf-8"), digest_size=8, person=b"MSFragFPv1"
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % n_bits


def _atom_environment(atom: Chem.Atom) -> str:
    isotope = atom.GetIsotope()
    symbol = f"{isotope}{atom.GetSymbol()}" if isotope else atom.GetSymbol()
    return ":".join(
        (
            symbol,
            str(atom.GetHybridization()),
            f"d{atom.GetDegree()}",
            f"h{atom.GetTotalNumHs()}",
            f"q{atom.GetFormalCharge()}",
            f"a{int(atom.GetIsAromatic())}",
        )
    )


def _bond_environment(bond: Chem.Bond) -> str:
    endpoints = sorted(
        (_atom_environment(bond.GetBeginAtom()), _atom_environment(bond.GetEndAtom()))
    )
    return "|".join(
        (
            endpoints[0],
            str(bond.GetBondType()),
            endpoints[1],
            f"ring={int(bond.IsInRing())}",
            f"conj={int(bond.GetIsConjugated())}",
        )
    )


def _components_after_cuts(
    molecule: Chem.Mol, cut_bond_indices: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    cut_bonds = set(cut_bond_indices)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for bond in molecule.GetBonds():
        if bond.GetIdx() in cut_bonds:
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adjacency[begin].append(end)
        adjacency[end].append(begin)

    unseen = set(range(molecule.GetNumAtoms()))
    components: list[tuple[int, ...]] = []
    while unseen:
        stack = [unseen.pop()]
        component: list[int] = []
        while stack:
            atom_idx = stack.pop()
            component.append(atom_idx)
            neighbours = [idx for idx in adjacency[atom_idx] if idx in unseen]
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _component_formula(molecule: Chem.Mol, atom_indices: Sequence[int]) -> str:
    counts: dict[tuple[int, int, str], int] = defaultdict(int)
    hydrogen_count = 0
    for atom_idx in atom_indices:
        atom = molecule.GetAtomWithIdx(atom_idx)
        key = (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetSymbol())
        counts[key] += 1
        if atom.GetAtomicNum() != 1:
            hydrogen_count += int(atom.GetTotalNumHs())
    if hydrogen_count:
        counts[(1, 0, "H")] += hydrogen_count

    pieces = []
    for (_, isotope, symbol), count in sorted(counts.items()):
        element = f"{isotope}{symbol}" if isotope else symbol
        pieces.append(element if count == 1 else f"{element}{count}")
    return "".join(pieces)


def theoretical_fragmentation_fingerprint(
    smiles: str,
    n_bits: int = 256,
) -> np.ndarray:
    """Return a precursor- and ionization-independent fragmentation fingerprint.

    The binary target hashes local bond environments and the component formulas
    produced by graph cuts. Single-bond cuts cover bridge bonds; paired cuts
    within each perceived ring provide fragment features for cyclic structures.
    Because component formulas are derived from the molecular graph rather than
    observed peaks, the same target definition applies to CID-MS/MS and EI-MS.
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    output = np.zeros((n_bits,), dtype=np.float32)
    features: set[str] = set()
    base_component_count = len(_components_after_cuts(molecule, ()))

    for atom in molecule.GetAtoms():
        features.add(f"ATOM|{_atom_environment(atom)}")

    bond_environments = {
        bond.GetIdx(): _bond_environment(bond) for bond in molecule.GetBonds()
    }
    for bond_idx, environment in bond_environments.items():
        features.add(f"BOND|{environment}")
        components = _components_after_cuts(molecule, (bond_idx,))
        if len(components) <= base_component_count:
            continue
        formulas = sorted(_component_formula(molecule, component) for component in components)
        features.add(f"CUT1|{environment}|{'/'.join(formulas)}")
        features.update(f"FRAGMENT|{environment}|{formula}" for formula in formulas)

    seen_ring_pairs: set[tuple[int, int]] = set()
    for ring_bonds in molecule.GetRingInfo().BondRings():
        for first, second in combinations(sorted(ring_bonds), 2):
            pair = (first, second)
            if pair in seen_ring_pairs:
                continue
            seen_ring_pairs.add(pair)
            components = _components_after_cuts(molecule, pair)
            if len(components) <= base_component_count:
                continue
            formulas = sorted(
                _component_formula(molecule, component) for component in components
            )
            environments = sorted((bond_environments[first], bond_environments[second]))
            features.add(
                f"CUT2|{environments[0]}|{environments[1]}|{'/'.join(formulas)}"
            )

    for feature in features:
        output[_stable_feature_bit(feature, n_bits)] = 1.0
    return output


def tagged_msms_spectrum(
    spectra: Sequence[Any],
    tags: Sequence[str],
    min_intensity: float = 0.0,
    decimals: int = 1,
) -> str:
    """Serialize multiple MS/MS conditions as tagged m/z-intensity tokens."""
    if len(spectra) != len(tags):
        raise ValueError("Each MS/MS condition must have one tag.")
    if decimals < 0:
        raise ValueError("decimals must be non-negative.")
    tokens: list[str] = []
    for tag, spectrum in zip(tags, spectra):
        tokens.append(tag)
        peaks = [peak for peak in _coerce_peaks(spectrum) if peak[1] >= min_intensity]
        if not peaks:
            tokens.append("NoPeak")
            continue
        for position, intensity in peaks:
            tokens.extend(
                (f"{position:.{decimals}f}", f"{intensity:.{decimals}f}")
            )
    return " ".join(tokens)

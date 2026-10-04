"""Raw-spectrum perturbations for source-side spectral repair training."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np
from torch.utils.data import get_worker_info


_ENERGY_MARKER = re.compile(r"^E(?:10|20|40)(?:Pos|Neg)$")


def _format_peak_value(value: float) -> str:
    return f"{float(value):.1f}"


def perturb_msms_text(
    spectrum: str,
    rng: np.random.Generator,
    *,
    severity: float = 1.0,
    structural: bool = True,
) -> str:
    """Perturb valid MS peak records while preserving energy headers."""
    tokens = str(spectrum).split()
    sections: list[tuple[str, list[tuple[float, float]]]] = []
    index = 0
    while index < len(tokens):
        marker = tokens[index]
        if not _ENERGY_MARKER.match(marker):
            index += 1
            continue
        index += 1
        peaks: list[tuple[float, float]] = []
        while index < len(tokens) and not _ENERGY_MARKER.match(tokens[index]):
            if index + 1 >= len(tokens):
                break
            try:
                peaks.append((float(tokens[index]), float(tokens[index + 1])))
            except ValueError:
                index += 1
                continue
            index += 2
        sections.append((marker, peaks))
    if not sections:
        return str(spectrum)

    output: list[str] = []
    global_mass_shift = float(rng.normal(0.0, 0.03 * severity))
    global_scale = float(rng.lognormal(0.0, 0.20 * severity))
    for marker, peaks in sections:
        output.append(marker)
        transformed: list[tuple[float, float]] = []
        for mass, intensity in peaks:
            if structural and len(peaks) > 1 and rng.random() < 0.05 * severity:
                continue
            new_mass = mass + global_mass_shift + float(rng.normal(0.0, 0.01 * severity))
            new_intensity = max(
                0.0,
                intensity
                * global_scale
                * float(rng.lognormal(0.0, 0.10 * severity)),
            )
            if structural and new_intensity < float(rng.uniform(0.0, 2.0 * severity)):
                continue
            transformed.append((new_mass, new_intensity))
        if not transformed and peaks:
            transformed.append(max(peaks, key=lambda value: value[1]))
        if structural and transformed and rng.random() < 0.35 * severity:
            low = min(value[0] for value in transformed)
            high = max(value[0] for value in transformed)
            if high > low:
                transformed.append(
                    (
                        float(rng.uniform(low, high)),
                        float(rng.uniform(0.5, 5.0 * severity)),
                    )
                )
        transformed.sort(key=lambda value: value[0])
        for mass, intensity in transformed:
            output.extend((_format_peak_value(mass), _format_peak_value(intensity)))
    return " ".join(output)


def perturb_hnmr_peaks(
    peaks: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
    *,
    severity: float = 1.0,
    structural: bool = True,
) -> list[dict[str, Any]]:
    """Apply bounded global/local ppm shifts and optional peak dropout."""
    shifted: list[dict[str, Any]] = []
    global_shift = float(rng.normal(0.0, 0.015 * severity))
    integral_scale = float(rng.lognormal(0.0, 0.08 * severity))
    for peak in peaks:
        if structural and len(peaks) > 1 and rng.random() < 0.04 * severity:
            continue
        value = copy.deepcopy(dict(peak))
        local_shift = global_shift + float(rng.normal(0.0, 0.003 * severity))
        for key in ("centroid", "delta", "rangeMax", "rangeMin"):
            if key in value and value[key] is not None:
                value[key] = float(value[key]) + local_shift
        if "nH" in value and value["nH"] is not None:
            value["nH"] = max(1, int(round(float(value["nH"]) * integral_scale)))
        shifted.append(value)
    if not shifted and peaks:
        shifted.append(copy.deepcopy(dict(peaks[0])))
    return shifted


def perturb_ir_spectrum(
    spectrum: Sequence[float],
    rng: np.random.Generator,
    *,
    severity: float = 1.0,
    structural: bool = True,
) -> list[float]:
    """Apply bounded baseline, scale, broadening, shift and local corruption."""
    values = np.asarray(spectrum, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        return list(spectrum)
    x = np.linspace(-1.0, 1.0, values.size)
    scale = float(rng.lognormal(0.0, 0.08 * severity))
    slope = float(rng.normal(0.0, 0.02 * severity))
    offset = float(rng.normal(0.0, 0.01 * severity))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    baseline = offset + slope * x + 0.01 * severity * np.sin(np.pi * x + phase)
    output = values * scale + baseline

    sigma = max(0.0, float(rng.uniform(0.0, 1.2 * severity)))
    if sigma > 0.15:
        radius = max(1, int(np.ceil(3.0 * sigma)))
        grid = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (grid / sigma) ** 2)
        kernel /= kernel.sum()
        output = np.convolve(output, kernel, mode="same")

    shift = float(rng.normal(0.0, 1.5 * severity))
    positions = np.arange(values.size, dtype=np.float64)
    output = np.interp(positions - shift, positions, output, left=output[0], right=output[-1])

    if structural and rng.random() < 0.5 * severity:
        width = max(2, int(rng.uniform(0.005, 0.03) * values.size))
        start = int(rng.integers(0, max(1, values.size - width)))
        amplitude = float(rng.normal(0.0, 0.04 * severity))
        window = np.hanning(width * 2 + 1)[width:]
        output[start : start + width] += amplitude * window[: min(width, values.size - start)]
    return output.astype(np.float32).tolist()


def perturb_spectral_sample(
    sample: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    modality: Optional[str] = None,
    severity: float = 1.0,
    structural: bool = True,
    candidates: Sequence[str] = ("HNMR", "MSMS", "IR"),
) -> tuple[dict[str, Any], str]:
    """Copy and perturb exactly one available repair modality."""
    output = copy.deepcopy(dict(sample))
    available = [name for name in candidates if name in output and output[name] is not None]
    if not available:
        return output, "none"
    selected = str(modality or rng.choice(available))
    if selected not in available:
        raise ValueError(f"requested perturbation modality {selected!r} is unavailable")
    if selected == "MSMS":
        output[selected] = perturb_msms_text(
            output[selected], rng, severity=severity, structural=structural
        )
    elif selected == "HNMR":
        output[selected] = perturb_hnmr_peaks(
            output[selected], rng, severity=severity, structural=structural
        )
    elif selected == "IR":
        output[selected] = perturb_ir_spectrum(
            output[selected], rng, severity=severity, structural=structural
        )
    else:
        raise ValueError(f"unsupported repair modality {selected!r}")
    return output, selected


@dataclass
class SpectralRepairPairCollator:
    """Collate clean and raw-perturbed views with the original preprocessor."""

    base_collator: Any
    seed: int = 3247
    severity: float = 1.0
    structural: bool = True
    candidates: Sequence[str] = ("HNMR", "MSMS", "IR")
    _calls: int = field(default=0, init=False, repr=False)

    def __call__(self, samples: list[MutableMapping[str, Any]]) -> dict[str, Any]:
        worker = get_worker_info()
        worker_offset = 0 if worker is None else (worker.id + 1) * 1_000_000
        rng = np.random.default_rng(self.seed + worker_offset + self._calls)
        self._calls += 1
        corrupt_samples = []
        context_samples = []
        selected = []
        for sample in samples:
            corrupt, modality = perturb_spectral_sample(
                sample,
                rng,
                severity=self.severity,
                structural=self.structural,
                candidates=self.candidates,
            )
            context_corrupt, _ = perturb_spectral_sample(
                sample,
                rng,
                modality=modality,
                severity=self.severity,
                structural=False,
                candidates=self.candidates,
            )
            corrupt_samples.append(corrupt)
            context_samples.append(context_corrupt)
            selected.append(modality)
        return {
            "clean": self.base_collator(samples),
            "corrupt": self.base_collator(corrupt_samples),
            "context_corrupt": self.base_collator(context_samples),
            "perturbed_modalities": selected,
        }

#!/usr/bin/env python3
"""Run a project CLI after registering education-compatible tokenizer proxies."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from tools import build_chemotion_compatible_preprocessor as compat


for name in (
    "ChemotionFormulaPreprocessor",
    "ChemotionCarbonPreprocessor",
    "ChemotionMultipletPreprocessor",
    "ChemotionMSMSTextPreprocessor",
):
    setattr(sys.modules["__main__"], name, getattr(compat, name))

if len(sys.argv) < 2:
    raise SystemExit("usage: run_education_compat.py TARGET_SCRIPT [ARGS ...]")

target = Path(sys.argv[1]).resolve()
sys.argv = [str(target), *sys.argv[2:]]
runpy.run_path(str(target), run_name="__main__")

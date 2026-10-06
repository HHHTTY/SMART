# SMART: Spectral Multimodal Adaptation via Routed Test-Time Tuning for Zero-Shot Molecular Elucidation

This repository provides the official implementation for the paper
**"SMART: Spectral Multimodal Adaptation via Routed Test-Time Tuning for
Zero-Shot Molecular Elucidation"**, which is under double-anonymous review.


## Overview

A molecular structure generator pretrained on *simulated* spectra can become
less accurate when *all* real measured spectra are supplied at deployment:
complementary chemical evidence is still present, but the learned fusion rule
fails to exploit it. SMART adapts the generator to real measurements without
any target structural labels by separating the view that supplies supervision
from the complete observations used for inference:

1. **Structural-utility routing.** A source-trained router selects, per
   molecule, a frozen teacher view over the fifteen non-empty combinations of
   H-NMR, C-NMR, MS, and IR (formula is retained in all views).
2. **Stage 1 — hybrid alignment.** A Full-input student is aligned to the
   frozen teacher through encoder InfoNCE contrastive alignment, which
   preserves cross-view molecular distinctions, and shared-prefix decoder
   Jensen–Shannon alignment, which transfers preferences among structural
   continuations.
3. **Stable-consensus filtering.** Only formula-compatible predictions that
   persist across training states are kept as pseudo labels.
4. **Stage 2 — missing-modality refinement.** The student is refined on these
   stable pseudo labels under randomly dropped spectral subsets, recovering
   robustness to incomplete observations.

## Repository Structure

| Location | Responsibility |
| --- | --- |
| `data/` | Datasets, tokenization, spectrum preprocessing, and collation |
| `modeling/` | Shared generation backbone, Router, and research modules |
| `generation/` | Generation-time logit processing |
| `trainer/` | Training-loop components |
| `cli/` | Training, prediction, and earlier source-retrieval TTT entry points |
| `configs/` | Backbone, input, and adaptation configurations |
| `run_router_teacher_full_casp_stage1.py` | Stage 1 router-teacher / Full-student alignment (see below) |
| `run_stable_checkpoint_dropout_stage2.py` | Stage 2 stable pseudo-label + modality-dropout training |
| `run_routed_pseudolabel_ttt_earlystop.py` | Routed pseudo-label adaptation with label-free early stopping |
| `run_stage2_router_opd_dropout.py` | Alternative Stage 2 router-OPD objective |
| `eval_stage2_modality_combinations.py` | Evaluate source/Stage-1/Stage-2 checkpoints across matched modality views |
| `eval_routed_ttt_sdbs500.py` | Predictions by modality view with post-hoc scoring |
| `sdbs_code/` | SDBS dataset builders and experiment utilities |
| `chemotion/` | Chemotion preparation, reported-results table, and provenance notes |
| `root_tools/` | Education benchmark builders and zero-shot evaluation |
| `snapshot_tools/` | Education compatibility runner |
| `optional_spectrallm/` | Optional external-backbone (OpenNMT, SpectraLLM) controls |
| `examples/` | Small fixtures for interface testing (not benchmark data) |
| `tests/` | Contract tests |
| `docs/` | Data contract, implementation status, and codebase references |

## Requirements

Python 3.10 or newer, with PyTorch >= 2.4. Key dependencies include
`transformers` (4.32–4.48), `pytorch-lightning`, `datasets`, `rdkit`,
`hydra-core`, `scikit-learn`, and `pybaselines`; the complete list is in
`pyproject.toml`.

```bash
conda create -n smart python=3.10 -y
conda activate smart
pip install -e .
```

Dataset manifests, pretrained checkpoints, and fitted preprocessor artifacts
are required inputs and are **not** distributed with this repository.

## Data Preparation

Read `docs/DATA_CONTRACT.md` before preparing a benchmark.

**SDBS** (3,669-record evaluation cohort; 1,000/500-record development
subsets) — builders live in `sdbs_code/root_tools/`:

```bash
python sdbs_code/root_tools/build_sdbs_final.py --help
python sdbs_code/root_tools/build_sdbs_final_ir1800.py --help
```

**Chemotion** (1,995 records) — preparation code and the reported-results
table with provenance notes are in `chemotion/`.

**Education** — an independent three-spectrum backbone and protocol, built and
evaluated via `root_tools/build_education_zeroshot.py` and
`root_tools/evaluate_education_zeroshot.py`.

> **Formula provenance.** The public SDBS and Chemotion builders derive the
> input molecular formula from the known structure. This is a data-preparation
> step; the adaptation objectives themselves never load ground-truth
> structures, and only observable inputs and frozen-teacher pseudo labels are
> read during training. Formula provenance should be reported together with
> any evaluation.

## Running

All runners accept the shared arguments `--data-path`, `--preprocessor`,
`--checkpoint`, `--model-config`, and `--data-config`; run any script with
`--help` for the full interface.

**Stage 1 — teacher/student alignment** (supports encoder NT-Xent, CE, KL,
hidden-state, and fingerprint objectives; see *Implementation Status*):

```bash
python run_router_teacher_full_casp_stage1.py \
    --data-path <tokenized_dataset> \
    --checkpoint <source_checkpoint> \
    --route-manifest <router_route_directory> \
    --run-dir runs/stage1
```

**Stage 2 — stable pseudo-label refinement with modality dropout**:

```bash
python run_stable_checkpoint_dropout_stage2.py \
    --data-path <tokenized_dataset> \
    --checkpoint <stage1_checkpoint> \
    --run-dir runs/stage2
```

**Evaluation across modality views** (separates prediction from scoring;
ground-truth SMILES are consumed only by the post-hoc scoring step):

```bash
python eval_stage2_modality_combinations.py \
    --data-path <tokenized_dataset> \
    --base-checkpoint <source_checkpoint> \
    --stage1-checkpoint runs/stage1 \
    --stage2-checkpoint runs/stage2 \
    --save-predictions
```

The end-to-end SDBS pipeline used in development is scripted in
`sdbs_code/snapshot_tools/run_router_teacher_full_pseudo_ce_stage2_interval3.sh`
(this shell pipeline is the recorded historical hard-CE, fixed-checkpoint
variant; see the next section).

## Main Results

Exact-match accuracy (%) on 3,669 experimental SDBS molecules, comparing the
unadapted source checkpoint with the SMART pipeline (central estimates from
the recorded runs):

| View | Source Top-1 | Source Top-5 | Source Top-10 | SMART Top-1 | SMART Top-5 | SMART Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H | 4.03 | 8.72 | 11.31 | 22.16 | 45.13 | 55.38 |
| C | 50.70 | 71.49 | 76.04 | 52.25 | 72.88 | 77.65 |
| MS | 2.04 | 4.99 | 6.76 | 13.49 | 33.96 | 44.04 |
| IR | 1.88 | 4.96 | 7.28 | 7.22 | 18.56 | 26.60 |
| H + C | 13.76 | 25.81 | 31.92 | 55.16 | 75.31 | 79.75 |
| C + MS | 11.17 | 21.18 | 26.11 | 51.87 | 72.85 | 77.54 |
| C + IR | 30.44 | 50.23 | 56.72 | 50.18 | 70.92 | 75.74 |
| H + C + MS | 5.70 | 11.15 | 14.64 | 55.41 | 76.12 | 80.54 |
| H + C + IR | 10.90 | 20.90 | 27.56 | 53.18 | 73.81 | 78.82 |
| C + MS + IR | 8.86 | 17.99 | 22.21 | 50.70 | 72.09 | 76.94 |
| **Full** | **5.94** | **11.42** | **14.91** | **55.66** | **76.51** | **80.97** |

Stage endpoints on SDBS, and a separate alignment-objective comparison
(different result summaries; see the paper for provenance):

| Stage summary | Top-1 | Top-5 | Top-10 | Objective summary | Top-1 |
| --- | ---: | ---: | ---: | --- | ---: |
| Routed teacher | 48.87 | 70.40 | 75.47 | Encoder InfoNCE only | 14.90 |
| Stage 1 | 53.09 | 73.78 | 78.00 | Decoder JS only | 54.50 |
| Stage 2 | 55.66 | 76.51 | 80.97 | Decoder KL only | 54.10 |
| | | | | Hard sequence CE | 51.40 |
| | | | | InfoNCE + JS | 55.66 |

On Chemotion (1,995 records; selected views), SMART improves Full Top-1 from
0.100% to 4.511%, exposing the limits of the recovery under stronger
acquisition shift; C + MS reaches the highest Top-10 at 9.323%. Full per-view
tables and dispersion caveats are provided in the paper and its appendix.

## Implementation Status

The paper's reported SDBS endpoint is attributed by the recorded provenance to
the hybrid encoder InfoNCE + decoder JS objective. The public runners in this
repository additionally implement earlier research objectives (NT-Xent, CE,
KL, hidden-state alignment, fingerprint listwise), and the scripted SDBS shell
pipeline reproduces the historical hard-pseudo-label CE variant with fixed
checkpoints. The protocol matrix, the mapping between reported results and
entry points, and the outstanding items for a complete public integration of
the hybrid loss and adaptive checkpoint selection are documented in
`docs/IMPLEMENTATION_STATUS.md`. Reported results must always be associated
with their actual configuration and checkpoint rather than inferred from a
script name.

Dataset manifests, pretrained checkpoints, and preprocessor artifacts are not
included; the small files under `examples/` are interface fixtures only and do
not substitute for them.

## Tests and Checks

The following checks require no model checkpoints or datasets:

```bash
python scripts/audit_publication.py --strict-language
python -m unittest discover -s tests -v
```

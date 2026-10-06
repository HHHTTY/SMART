<div align="center">

# 🔬 SMART

**Spectral Multimodal Adaptation via Routed Test-Time Tuning<br>for Zero-Shot Molecular Structure Elucidation**

[![Status](https://img.shields.io/badge/Status-Under%20Anonymous%20Review-orange)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)]()
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.32–4.48-FFD21E)]()
[![RDKit](https://img.shields.io/badge/Cheminformatics-RDKit%202024.09%2B-8E44AD)]()

*Adapting a spectra-to-structure generator from simulated to real measurements —*
***without a single target structural label.***

📋 [Overview](#-overview) · 🧪 [Experiments](#-experiments) · 🛠️ [Installation](#%EF%B8%8F-installation) · 📦 [Data](#-data-preparation) · 🚀 [Usage](#-usage) · 📁 [Structure](#%EF%B8%8F-repository-structure) · 📜 [Citation](#-citation)

</div>

---

> **🕯️ Anonymous submission.** This repository accompanies a paper under
> double-anonymous review and is hosted anonymously for the review period.
> No author-identifying information is included.

## 📖 Overview

A molecular structure generator pretrained on **simulated** spectra often gets
**worse** when you hand it **all** real measured spectra at deployment: the
complementary chemical evidence is still there, but the learned fusion rule
fails to exploit it — a negative-transfer failure under heterogeneous
acquisition shift.

SMART fixes the *rule*, not the data: it separates **the view that supplies
supervision** from **the complete observations used for inference**.

```
        real spectra:  H-NMR · C-NMR · MS · IR  (+ molecular formula)
                                │
                                ▼
              ┌─────────────────────────────────┐
              │  🧭 Structural-utility Router    │  source-trained, 15 views;
              │     (frozen teacher view)        │  picks the reliable subset
              └────────────────┬────────────────┘
                               ▼
              ┌─────────────────────────────────┐
              │  ⚗️  Stage 1 · Hybrid alignment  │  encoder InfoNCE  +
              │     (no target labels)           │  decoder Jensen–Shannon
              └────────────────┬────────────────┘     on shared prefixes
                               ▼
              ┌─────────────────────────────────┐
              │  🧷 Stable-consensus filter      │  keep formula-compatible
              │                                  │  predictions that persist
              └────────────────┬────────────────┘  across training states
                               ▼
              ┌─────────────────────────────────┐
              │  🌀 Stage 2 · Missing-modality   │  modality-dropout training
              │     refinement                   │  on stable pseudo labels
              └─────────────────────────────────┘
```

**✨ Key ideas at a glance**

| | Idea | Why it matters |
| --- | --- | --- |
| 🧭 | **Routed supervision** — a source-trained structural-utility router freezes a per-molecule teacher view | Supervision comes only from where the source model is *already* trustworthy |
| ⚗️ | **Hybrid alignment** — encoder InfoNCE + shared-prefix decoder Jensen–Shannon | Preserves cross-view molecular distinctions *and* transfers structural-continuation preferences |
| 🧷 | **Stable-consensus pseudo labels** — predictions surviving across checkpoints | Agreement filters label noise without peeking at ground truth |
| 🌀 | **Missing-modality refinement** — training under randomly dropped spectral subsets | Recovers robustness when real instruments deliver incomplete observations |

## 🧪 Experiments

All numbers are **exact-match accuracy (%)** from the recorded runs reported in
the paper; beam 10, canonical structure comparison. Evaluation cohorts carry no
structural labels during adaptation.

### 📊 Main results — SDBS, 3,669 experimental molecules

Source checkpoint (simulated-only pretraining) vs. SMART, across all eleven
reported views:

| View | Source Top-1 | Source Top-5 | Source Top-10 | SMART Top-1 | SMART Top-5 | SMART Top-10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| H | 4.03 | 8.72 | 11.31 | 22.16 | 45.13 | 55.38 |
| **C** *(strongest single view)* | **50.70** | **71.49** | **76.04** | **52.25** | **72.88** | **77.65** |
| MS | 2.04 | 4.99 | 6.76 | 13.49 | 33.96 | 44.04 |
| IR | 1.88 | 4.96 | 7.28 | 7.22 | 18.56 | 26.60 |
| H + C | 13.76 | 25.81 | 31.92 | 55.16 | 75.31 | 79.75 |
| C + MS | 11.17 | 21.18 | 26.11 | 51.87 | 72.85 | 77.54 |
| C + IR | 30.44 | 50.23 | 56.72 | 50.18 | 70.92 | 75.74 |
| H + C + MS | 5.70 | 11.15 | 14.64 | 55.41 | 76.12 | 80.54 |
| H + C + IR | 10.90 | 20.90 | 27.56 | 53.18 | 73.81 | 78.82 |
| C + MS + IR | 8.86 | 17.99 | 22.21 | 50.70 | 72.09 | 76.94 |
| **🚀 Full** | **5.94** | **11.42** | **14.91** | **55.66** | **76.51** | **80.97** |

- 🔧 **Repairs a broken fusion rule:** Full Top-1 climbs **5.94 → 55.66**
  (**+49.72 pts**) — negative transfer is reversed, not merely softened.
- 📈 **Beats the strongest single view:** against source carbon (50.70), the
  gain is **+4.96 pts** Top-1 and **+4.93 pts** Top-10 — complete observations
  finally pay off *beyond* an already-strong partial input.
- 🧩 **Graceful degradation:** H+C+MS lands at 55.41 (−0.25 vs. Full): most of
  the recovered accuracy is accessible without infrared.

### 🪜 Where the gains come from — stage endpoints & objective comparison

Two independent result summaries (see the paper for provenance of each):

| Stage summary | Top-1 | Top-5 | Top-10 | | Objective summary | Top-1 |
| :--- | ---: | ---: | ---: | --- | :--- | ---: |
| Routed teacher | 48.87 | 70.40 | 75.47 | | Encoder InfoNCE only | 14.90 |
| Stage 1 | 53.09 | 73.78 | 78.00 | | Decoder JS only | 54.50 |
| Stage 2 | **55.66** | **76.51** | **80.97** | | Decoder KL only | 54.10 |
| | | | | | Hard sequence CE | 51.40 |
| | | | | | **InfoNCE + JS (hybrid)** | **55.66** |

The student surpasses its own teacher (**+6.79 pts**), and the hybrid objective
edges the best single objective — while decoder-side guidance clearly carries
most of the predictive value.

### 🧬 Beyond exact match — structural quality (paired 500-record SDBS analysis)

| Metric (Full view) | Source | SMART |
| :--- | ---: | ---: |
| Valid molecules | 51.4% | **100%** ✅ |
| Functional-group similarity | 0.1351 | **0.3109** |
| MCES distance ↓ | 16.294 | **9.070** |

Adaptation fixes *chemistry*, not just string matching — every adapted Full
output is a valid molecule. (Carbon-only MCES instead rises 8.272 → 9.484:
improvements are not uniform across views.)

### 🌪️ Stronger shift — Chemotion, 1,995 records

A harder acquisition shift, run as a **separate recorded protocol**:

| View | Source Top-1 | Source Top-5 | Source Top-10 | SMART Top-1 | SMART Top-5 | SMART Top-10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 3.609 | 6.366 | 8.020 | 3.759 | 7.368 | 8.972 |
| H + C | .201 | .401 | .652 | 4.060 | 7.419 | 8.672 |
| C + MS | 1.454 | 3.008 | 3.759 | 4.110 | 7.419 | **9.323** |
| Full | .100 | .201 | .351 | **4.511** | 7.018 | 8.271 |

Full Top-1 improves **0.100 → 4.511** (45×), yet C+MS owns the best Top-10 —
honest evidence that recovery has limits under strong shift, and complete
observations do not dominate at every budget.

### ⚙️ Recorded training configurations

| Component | Setting |
| :--- | :--- |
| 🧭 Router | AdamW · 100 epochs · batch 256 · lr 3e-4 · seed 3247 · 43,978 source episodes (35,183 train / 8,795 val; 2,048 weak / 30,819 medium / 11,111 strong) |
| ⚗️ SDBS Stage 1 | AdamW · 12 epochs · batch 16 · lr 1e-5 · bf16 · grad-clip 0.8 · seed 3247 |
| 🌀 Chemotion | Stage-1 epoch 19 · Stage 2 up to epoch 20 · epoch 18 selected by minimum pseudo-CE |
| 🔍 Checkpoint scheduling | Reference adaptive controller: ≤256 deterministic-hash probe records, interval 1–4, feasibility gate <0.80, drift-halving rule |

**🔁 Protocol notes (please read before comparing numbers).**
Repeated adaptation and scoring on the same cohort is *transductive*
evaluation; dispersion columns from the recorded runs are not repeat-seed
standard deviations or confidence intervals; no target-label hyperparameter
search is part of the intended protocol; the historical route summary assigns
≈97.5% of records to the carbon view, so a matched fixed-carbon-teacher
baseline remains the decisive test for routing itself.

## 🛠️ Installation

**Requirements:** Python ≥ 3.10 · PyTorch ≥ 2.4 · a CUDA GPU is strongly
recommended for training (evaluation of released predictions is CPU-friendly).

```bash
# 1) create the environment
conda create -n smart python=3.10 -y
conda activate smart

# 2) install the package (editable, with all dependencies)
pip install -e .
```

Core dependency versions:

| Package | Requirement | Package | Requirement |
| :--- | :--- | :--- | :--- |
| `torch` | ≥ 2.4.0 | `tokenizers` | ≥ 0.13.3 |
| `transformers` | ≥ 4.32.1, < 4.49.0 | `pytorch-lightning` | ≥ 2.1.0 |
| `datasets` | ≥ 2.14.4 | `hydra-core` | ≥ 1.3.2 |
| `rdkit` | ≥ 2024.9 | `pybaselines` | ≥ 1.2.0 |
| `scikit-learn` | ≥ 1.3.2 | `scipy` | ≥ 1.10.1 |
| `numpy` | ≥ 1.24.4 | `pandas` | ≥ 1.5.3 |
| `tensorboard` | ≥ 2.14.0 | `torchmetrics` | ≥ 1.2.0 |
| `loguru` | ≥ 0.7.3 | `pydantic` | ≥ 2.6.3 |

> ⚠️ **Artifacts are inputs, not downloads.** Dataset manifests, pretrained
> checkpoints, and fitted preprocessor artifacts are required to run the
> pipelines and are **not** distributed with this repository. The small files
> under `examples/` are interface fixtures only.

## 📦 Data Preparation

Start from [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) — it defines the
schemas every builder emits.

| Benchmark | Cohort | Builders |
| :--- | :--- | :--- |
| 🧪 **SDBS** | 3,669 records (evaluation) · 1,000 / 500 (development) | `sdbs_code/root_tools/build_sdbs_final.py`, `build_sdbs_final_ir1800.py`, subset builders |
| 🌪️ **Chemotion** | 1,995 records | `chemotion/` (preparation + reported-results table with provenance notes) |
| 🎓 **Education** | independent three-spectrum protocol | `root_tools/build_education_zeroshot.py`, `evaluate_education_zeroshot.py` |

```bash
python sdbs_code/root_tools/build_sdbs_final.py --help
python sdbs_code/root_tools/build_sdbs_final_ir1800.py --help
```

> 🔎 **Formula provenance.** The public SDBS and Chemotion builders derive the
> input molecular formula from the known structure. This is a data-preparation
> step: the adaptation objectives themselves never load ground-truth
> structures — only observable inputs and frozen-teacher pseudo labels are
> read during training. Report formula provenance together with any
> evaluation.

## 🚀 Usage

All runners share `--data-path`, `--preprocessor`, `--checkpoint`,
`--model-config`, and `--data-config`; every script documents its full
interface via `--help`.

**⚗️ Stage 1 — router-teacher / Full-student alignment**

```bash
python run_router_teacher_full_casp_stage1.py \
    --data-path <tokenized_dataset> \
    --checkpoint <source_checkpoint> \
    --route-manifest <router_route_directory> \
    --run-dir runs/stage1
```

**🌀 Stage 2 — stable pseudo-label refinement with modality dropout**

```bash
python run_stable_checkpoint_dropout_stage2.py \
    --data-path <tokenized_dataset> \
    --checkpoint <stage1_checkpoint> \
    --run-dir runs/stage2
```

**📏 Evaluation across matched modality views** — prediction is separated from
scoring; ground-truth SMILES are consumed only by the post-hoc scoring step:

```bash
python eval_stage2_modality_combinations.py \
    --data-path <tokenized_dataset> \
    --base-checkpoint <source_checkpoint> \
    --stage1-checkpoint runs/stage1 \
    --stage2-checkpoint runs/stage2 \
    --save-predictions
```

The end-to-end SDBS pipeline used during development is scripted in
[`sdbs_code/snapshot_tools/run_router_teacher_full_pseudo_ce_stage2_interval3.sh`](sdbs_code/snapshot_tools/run_router_teacher_full_pseudo_ce_stage2_interval3.sh)
(the recorded historical hard-CE, fixed-checkpoint variant — see
[Implementation Status](#-implementation-status)).

## 📁 Repository Structure

```
SMART/
├── 📄 README.md, LICENSE, pyproject.toml
├── ⚗️ run_router_teacher_full_casp_stage1.py   # Stage 1 alignment
├── 🌀 run_stable_checkpoint_dropout_stage2.py  # Stage 2 refinement
├── 🔁 run_routed_pseudolabel_ttt_earlystop.py  # routed TTT + label-free early stop
├── 🔀 run_stage2_router_opd_dropout.py         # alternative Stage 2 objective
├── 📏 eval_stage2_modality_combinations.py     # cross-view evaluation
├── 📏 eval_routed_ttt_sdbs500.py               # view-wise prediction + post-hoc scoring
├── data/                 # datasets, tokenization, preprocessing, collation
├── modeling/             # generation backbone, Router, research modules
├── generation/           # generation-time logit processing
├── trainer/              # training-loop components
├── cli/                  # training / prediction / retrieval-TTT entry points
├── configs/              # backbone, input, adaptation configurations
├── sdbs_code/            # SDBS builders + experiment utilities
├── chemotion/            # Chemotion prep + reported-results table
├── root_tools/           # Education builders & zero-shot evaluation
├── snapshot_tools/       # Education compatibility runner
├── optional_spectrallm/  # external-backbone (OpenNMT, SpectraLLM) controls
├── examples/             # interface fixtures (not benchmark data)
├── tests/                # contract tests
└── docs/                 # DATA_CONTRACT, IMPLEMENTATION_STATUS, codebase refs
```

## 🔍 Implementation Status

The paper's reported SDBS endpoint is attributed by its recorded provenance to
the hybrid **encoder InfoNCE + decoder JS** objective. The public runners in
this repository additionally implement earlier research objectives (NT-Xent,
CE, KL, hidden-state alignment, fingerprint listwise), and the scripted SDBS
shell pipeline reproduces the historical hard-pseudo-label CE variant with
fixed checkpoints. The protocol matrix — which entry point produced which
reported number, and what remains for a complete public integration of the
hybrid loss and adaptive checkpoint selection — is documented in
[`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md). Always
associate a reported result with its actual configuration and checkpoint
rather than inferring it from a script name.

## ✅ Tests and Checks

```bash
python scripts/audit_publication.py --strict-language   # release-content audit
python -m unittest discover -s tests -v                 # contract tests
```

Both run without model checkpoints or datasets. The audit verifies
English-only content, filenames, syntax, documentation links, and the absence
of hardcoded machine paths; passing it does not by itself establish
end-to-end molecular performance.


## 🙏 Acknowledgments

We thank the maintainers of the SDBS, Chemotion, and Education data resources
referenced in the paper.

---

<div align="center">


</div>

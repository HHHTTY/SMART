# Chemotion Zeroshot Tokenization Audit

## Scope

This note records checkpoint-compatible tokenization changes for the Chemotion IR1800 test set. The evaluated checkpoint is `checkpoints/epoch_24-step_122175.ckpt` (epoch 24); no model weights or vocabulary dimensions are changed. The current evaluation contains 1,995 rows and compares Formula+HNMR, Formula+CNMR, Formula+MSMS, and Formula+IR. Formula-only is intentionally excluded.

## Findings and handling

### Formula

The original SDBS Formula tokenizer produced active `<unk>` tokens in 520/1,995 Chemotion rows (767 active unknown tokens). Count tokens for known elements that exceed the learned vocabulary are now mapped to the nearest learned count for that same element, without changing token IDs. The mapped formula is kept compact (for example `C33H36N6O4`) before tokenization; whitespace-separated element fragments are invalid for this tokenizer.

This reduced active `<unk>` rows to 205/1,995 (213 active unknown tokens). The residual unknowns are principally elements absent from the checkpoint vocabulary, such as Si, Cu, Fe, and other less common elements. They are retained as `<unk>` rather than silently rewritten as a different element.

### HNMR and CNMR

Chemotion peak dictionaries expose `ppm` and `intensity`, whereas the checkpoint preprocessors expect structured H multiplets (`rangeMin`, `rangeMax`, `centroid`, `category`, `nH`) and C shifts (`delta (ppm)`). The compatibility wrapper normalizes these fields before using the original SDBS processing logic. Missing H ranges become a point peak at the measured ppm; absent multiplicity/integration metadata use neutral defaults (`m`, `1`). C peaks pass through their ppm values without intensity conditioning.

After the SDBS formatter runs, numeric shifts not in the learned vocabulary are mapped to the nearest numeric token already present for that modality. The prior audit estimated that this affects about 6.50% of H numeric tokens and 1.61% of C numeric tokens. This keeps the frozen embeddings valid, but it is quantization, not a chemically learned Chemotion vocabulary; large nearest-token errors remain a limitation.

### MSMS

Chemotion MS is already stored as alternating `m/z intensity` values. Its longest row contains 6,074 whitespace tokens, while the checkpoint MS tokenizer and preprocessor window is 392 tokens. Passing all peaks through caused multimodal generation to exceed the supported positional range and ended in a CUDA index assertion.

The compatibility preprocessor now selects up to 195 peaks by descending intensity, sorts the retained peaks by ascending m/z, maps numeric values to the checkpoint vocabulary, and tokenizes with truncation at the checkpoint's 392-token window. This preserves the strongest peaks while retaining the model's learned sequence-length contract. A targeted smoke including the 6,074-token row and other long MS rows completed for all four modality combinations with CUDA blocking enabled.

The earlier audit estimated that about 7.13% of MS numeric tokens required nearest-vocabulary mapping; some m/z mappings exceed 5 Da. The peak budget fixes sequence overflow, not this numeric resolution mismatch.

### IR

Chemotion IR uses the same 1,800-point representation as the selected IR1800 checkpoint, so no nearest-token remapping or peak-list conversion is applied. A small Formula+IR smoke completed successfully.

## Verification and experiment

- Formula vocabulary audit: 1,995 rows; active unknown rows reduced from 520 to 205.
- Four-combination edge smoke: rows selected to include the longest MS spectra; FH/FC/FM/FI all generated without a CUDA assertion using `CUDA_LAUNCH_BLOCKING=1`.
- Full zeroshot run: GPU3, checkpoint `epoch_24-step_122175.ckpt`, beam 10, bf16, 1,995 rows, combinations FH/FC/FM/FI, output directory `runs/chemotion_v9_formula_hc_mspeaklimit_zeroshot_noF_1995_gpu3_20260923/`.
- Zeroshot metrics:

| Combination | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Formula+HNMR | 0/1,995 (0.000%) | 1/1,995 (0.050%) | 3/1,995 (0.150%) |
| Formula+CNMR | 2/1,995 (0.100%) | 4/1,995 (0.201%) | 4/1,995 (0.201%) |
| Formula+MSMS | 5/1,995 (0.251%) | 9/1,995 (0.451%) | 12/1,995 (0.602%) |
| Formula+IR | 1/1,995 (0.050%) | 4/1,995 (0.201%) | 5/1,995 (0.251%) |

- Result JSON: `results.json`; launch log: `run.log`. Formula-only was intentionally not evaluated.

## Interpretation limits

These are checkpoint-compatibility transformations, not retraining or vocabulary expansion. Formula OOV elements remain unknown; H/C/MS numeric nearest-token mapping can distort values; and MS intensity top-195 selection may differ from the model's original peak distribution. The zeroshot results should therefore be read as a controlled compatibility experiment against the frozen epoch-24 SDBS model, not as an estimate of Chemotion-domain model quality after adaptation.

## Zeroshot result interpretation

All four Formula+modality combinations remain below 0.61% Top-10, with Formula+MSMS highest at 0.602%. These results do not indicate that the compatibility preprocessing has repaired the Chemotion zeroshot gap. The successful smoke tests establish that the processed inputs fit the checkpoint's tokenizer and positional limits; they do not establish that the resulting numeric tokens or spectra are semantically aligned with the SDBS training distribution. Follow-up diagnosis should inspect the actual post-preprocessing token sequences and compare them with SDBS inputs, particularly for C shifts and MS peak values/order, before treating further peak-count tuning as a likely fix.

### Comparability warning: previous Formula+CNMR baseline

The earlier 3.609% Top-1 / 8.020% Top-10 Formula+CNMR result used the
`v8_densec03_strictdense_hcpeakpick_csolventclean/test.parquet` dataset and
`preprocessor_chemotion_compat_v3_ms.pkl`. The v9 run's launch configuration
instead points to the unprocessed
`Chemotion_final_ir1800_20260922/test.parquet` dataset and
`preprocessor_chemotion_compat_v9_formula_hc_mspeaklimit.pkl`. In the raw data,
C inputs have a median of 100 entries per row, a maximum of 17,984 entries, and
66,450 shifts outside -5 to 230 ppm across the split. The strict peak-picked
data has a median of 13 peaks, maximum 53, and only 13 shifts outside that
range. Thus v9's 0.100% FC Top-1 is not a like-for-like regression against the
3.609% baseline; the C input itself is malformed for the model's peak-list
preprocessor. A fresh FC reproduction on the strict peak-picked data with the older v2
preprocessor completed and exactly reproduced the baseline: Top-1 3.609%,
Top-10 8.020% (1,995 rows, beam 10). Its artifacts are in
`runs/chemotion_fc_repro_v8strict_v2pre_1995_20260923/FC.json` on HPC. This
confirms that v9's near-zero FC score came from feeding the raw C trace table
to the peak-list preprocessor, not from a change to the checkpoint or an
unexplained evaluation regression.

The IR1800 input is a continuous 1,800-point spectrum and is not peak-picked.
H/MS inputs are already discrete peak observations; sample solvent-condition
fields are empty, so H/MS solvent-frequency or low-m/z deletion requires
separate candidate-level auditing rather than blanket removal. No such
candidate cleaning should replace the original dataset without a controlled
zeroshot comparison.

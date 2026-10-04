# Chemotion v8 Strict: 14-Combination Zero-Shot and Router-Teacher TTT

## Data and setup

- Dataset: `Chemotion_final_ir1800_20260922_peakpicked_model_compatible_v8_densec03_strictdense_hcpeakpick_csolventclean/test.parquet` (1,995 rows).
- Checkpoint: `epoch_24-step_122175.ckpt` from `paper_multitask_ms`.
- Preprocessor: `preprocessor_chemotion_compat_v3_ms.pkl`.
- Zero-shot: beam 10, Formula retained, no Formula-only evaluation. The panel is the 14 non-Full subsets of HNMR/CNMR/MSMS/IR plus Full. MSMS-containing evaluations use a 916-token limit.
- IR is already represented on the checkpoint's 1,800-point grid. H/C peak picking and C solvent cleaning are represented by the selected strict dataset version; MSMS remains peak-list input.

## Router distribution

The profiled router was run on the same 1,995 rows without reading ground-truth structures.

| Router-selected spectra | Rows | Share |
|---|---:|---:|
| CNMR | 1,531 | 76.74% |
| CNMR + MSMS | 331 | 16.59% |
| CNMR + MSMS + IR | 93 | 4.66% |
| MSMS | 32 | 1.60% |
| Other combinations | 8 | 0.40% |

Overall, 1,955/1,995 rows (94.0%) route through CNMR. Formula-compatible router pseudo-labels were available for 620 rows (31.1%).

## Zero-shot results

Top-k is the fraction of the 1,995 targets present in the first k beam candidates.

| Input combination | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Formula + HNMR | 0.150% | 0.201% | 0.301% |
| Formula + CNMR | 3.609% | 6.366% | 8.020% |
| Formula + IR | 0.050% | 0.251% | 0.301% |
| Formula + MSMS | 0.251% | 0.451% | 0.602% |
| Formula + HNMR + CNMR | 0.201% | 0.401% | 0.652% |
| Formula + HNMR + IR | 0.050% | 0.150% | 0.150% |
| Formula + HNMR + MSMS | 0.050% | 0.100% | 0.150% |
| Formula + HNMR + CNMR + IR | 0.050% | 0.351% | 0.551% |
| Formula + HNMR + CNMR + MSMS | 0.150% | 0.351% | 0.501% |
| Formula + HNMR + MSMS + IR | 0.050% | 0.150% | 0.201% |
| Formula + CNMR + IR | 1.404% | 3.158% | 3.910% |
| Formula + CNMR + MSMS | 1.454% | 3.008% | 3.759% |
| Formula + CNMR + MSMS + IR | 0.602% | 1.253% | 1.855% |
| Formula + MSMS + IR | 0.000% | 0.100% | 0.201% |
| Full | 0.100% | 0.201% | 0.351% |

Formula + CNMR is the strongest zero-shot setting. Adding HNMR, IR, or MSMS to CNMR lowers its Top-1 and Top-10 results in this checkpoint-compatible setup. The gap versus HNMR-only, IR-only, and Full is in the tens-fold range.

## TTT run

The run reuses `run_router_teacher_full_casp_stage1.py` and `run_stable_checkpoint_dropout_stage2.py`, with Chemotion MS/IR column names and early-stop settings in the run wrapper.

| Stage | Configuration and outcome |
|---|---|
| Stage 1 | No fixed epoch cap (`--epochs 0`), `lr=3e-5`, patience 5, minimum 6 epochs, `min_delta=0.001`, interval-3 snapshots. Stopped at epoch 19. Pseudo CE fell from 0.5804 at epoch 1 to 0.000673 at epoch 19. |
| Stable views | Full-view greedy predictions from epochs 3, 6, 9, 12, 15, and 18; 1,995 rows aligned across all six views. |
| Consensus filter | 206 canonical consensuses; 175 valid and formula-matching rows accepted; 31 formula mismatches; 552 invalid outputs; 1,237 rows disagreed across views. Ground-truth structures were not loaded. |
| Stage 2 | `exclude_full` modality dropout, `lr=1e-5`, early-stop patience 3, minimum 3 epochs, `min_delta=0.002`, safety cap 20. It reached epoch 20 before patience fired. Best training pseudo CE was 0.098717 at epoch 18; epoch 20 was 0.109629. |

The post-TTT evaluation uses the Stage 2 epoch-18 checkpoint, the lowest-pseudo-CE saved checkpoint.

## Post-TTT Comparison

The final SMART checkpoint was evaluated for all 14 non-Full modality subsets and Full, using the same 1,995 rows, beam width, and preprocessing as the zero-shot panel.

| Input combination | SMART Top-1 | SMART Top-5 | SMART Top-10 |
|---|---:|---:|---:|
| Formula + HNMR | 2.356% | 4.411% | 5.714% |
| Formula + CNMR | 4.110% | 7.419% | 9.323% |
| Formula + MSMS | 1.704% | 3.459% | 4.261% |
| Formula + IR | 1.855% | 3.860% | 5.564% |
| Formula + HNMR + CNMR | 4.060% | 7.419% | 8.672% |
| Formula + HNMR + MSMS | 2.256% | 4.411% | 5.664% |
| Formula + HNMR + IR | 2.406% | 4.662% | 5.564% |
| Formula + CNMR + MSMS | 3.759% | 7.368% | 8.972% |
| Formula + CNMR + IR | 3.910% | 7.419% | 9.273% |
| Formula + MSMS + IR | 1.554% | 3.409% | 4.361% |
| Formula + HNMR + CNMR + MSMS | 4.511% | 7.018% | 8.271% |
| Formula + HNMR + CNMR + IR | 4.160% | 7.218% | 8.622% |
| Formula + HNMR + MSMS + IR | 2.206% | 4.311% | 5.414% |
| Formula + CNMR + MSMS + IR | 4.160% | 7.068% | 8.521% |
| Full | 4.361% | 7.368% | 8.571% |

The highest SMART Top-1 is 4.511% for Formula + HNMR + CNMR + MSMS; Formula + CNMR has the highest Top-10 at 9.323%. Full reaches 4.361% / 7.368% / 8.571%, versus 0.100% / 0.201% / 0.351% zero-shot. These are transductive adaptation measurements on the same rows used for unlabeled adaptation, not an independent holdout estimate. Ground truth is used only for the final evaluation.

## Artifacts

- Zero-shot run: `/hpc2hdd/home/aimslab/ChengtangZhan/ttt/runs/chemotion_v8_strict_compatv3_14combos_zeroshot_20260923_gpu4/`
- Router routes: `/hpc2hdd/home/aimslab/ChengtangZhan/ttt/paper_multitask_ms_snapshot_20260914/runs/chemotion_v8_strict_compatv3_router_profiled_v6_routes_20260923_gpu3/`
- TTT run: `/hpc2hdd/home/aimslab/ChengtangZhan/ttt/paper_multitask_ms_snapshot_20260914/runs/chemotion_v8_strict_compatv3_router_teacher_ttt_earlystop_lr3e5_unbounded_20260923_gpu3/`
- Post-TTT evaluations: all 14 subset JSONs under `.../post_ttt_eval/final_modalities/` and `.../post_ttt_eval/full.json` under the TTT run directory.
- Comparison table: `/Users/zhanchengtang/Documents/ttt/Chemotion/chemotion_zero_shot_smart_table.tex` (PDF and PNG are alongside it).

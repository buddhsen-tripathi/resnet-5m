# All completed models

Generated from [`assets/experiments.json`](../assets/experiments.json) by [`scripts/plot_experiments.py`](../scripts/plot_experiments.py).

- 31 completed single-model training runs; 60-epoch architecture screens, teacher ensembles and the two over-budget round 11 teachers excluded.
- Values reproduce the round tables in README.md; each run uses its recorded evaluation protocol.
- No-TTA values generally come from mixed-precision training evaluation; R8 and R9 use final full-precision evaluation.
- R6 is 97.25% in its original training log and 97.26% in the later H200 full-precision re-evaluation.
- Kaggle scores refer to the recorded standard flip-TTA submission; alternative zoom/shift, BatchNorm-adaptation and balanced-decoding submissions are excluded.
- Missing Kaggle scores mean no recorded submission for that checkpoint and protocol.
- Seeds, hardware, augmentation and teachers vary across rounds; these are individual runs, not controlled comparisons or multi-seed estimates.
- The public leaderboard scores 1,000 images (50%); the other 1,000 form the private split.
- Parameter counts refer to the inference classifier. R9, R10 and R11 also train a 45,824-parameter projection head; their totals remain below 5M.
- Round 10 and 11 checkpoints are not release assets; R10_t12's inference export is in weights/.

| Round | Checkpoint | Classifier parameters | Epochs | CIFAR-10 final (%) | Flip TTA (%) | Kaggle public | Record |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | [A_64-128-256_343_se](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/A_64-128-256_343_se.pt) | 4,658,278 | 200 | 95.81 | 96.21 | 0.873 | [source](../logs/1-architecture/A_64-128-256_343_se.log) |
| 1 | [A_64-128-256_343_nose](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/A_64-128-256_343_nose.pt) | 4,622,410 | 200 | 96.19 | 96.55 | — | [source](../logs/1-architecture/A_64-128-256_343_nose.log) |
| 1 | [B_64-128-256_353_se](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/B_64-128-256_353_se.pt) | 4,955,886 | 200 | 96.23 | 96.58 | — | [source](../logs/1-architecture/B_64-128-256_353_se.log) |
| 1 | [C_80-160-320_232_se](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/C_80-160-320_232_se.pt) | 4,836,010 | 200 | 96.18 | 96.36 | — | [source](../logs/1-architecture/C_80-160-320_232_se.log) |
| 1 | [D_96-192-320_222_se](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/D_96-192-320_222_se.pt) | 4,939,894 | 200 | 96.06 | 96.21 | — | [source](../logs/1-architecture/D_96-192-320_222_se.log) |
| 2 | [C_ta](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/C_ta.pt) | 4,836,010 | 200 | 96.76 | 96.84 | — | [source](../logs/2-augmentation/C_ta.log) |
| 2 | [C_mix](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/C_mix.pt) | 4,836,010 | 200 | 96.68 | 97.01 | — | [source](../logs/2-augmentation/C_mix.log) |
| 2 | [C_ta_mix](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/C_ta_mix.pt) | 4,836,010 | 200 | 96.84 | 96.98 | — | [source](../logs/2-augmentation/C_ta_mix.log) |
| 2 | [C_ta_mix_noerase](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/C_ta_mix_noerase.pt) | 4,836,010 | 200 | 96.93 | 96.97 | 0.884 | [source](../logs/2-augmentation/C_ta_mix_noerase.log) |
| 2 | [A_nose_ta_mix](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/A_nose_ta_mix.pt) | 4,622,410 | 200 | 96.77 | 97.00 | — | [source](../logs/2-augmentation/A_nose_ta_mix.log) |
| 3 | [C_kd5](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/C_kd5.pt) | 4,836,010 | 200 | 96.13 | 96.33 | — | [source](../logs/3-distillation/C_kd5.log) |
| 3 | [KD_C_aug](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/KD_C_aug.pt) | 4,836,010 | 200 | 97.22 | 97.31 | 0.889 | [source](../logs/3-distillation/KD_C_aug.log) |
| 3 | [KD_A_nose_aug](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/KD_A_nose_aug.pt) | 4,622,410 | 200 | 97.27 | 97.23 | — | [source](../logs/3-distillation/KD_A_nose_aug.log) |
| 3 | [KD_C_aug_T1](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.1/KD_C_aug_T1.pt) | 4,836,010 | 200 | 97.12 | 97.25 | — | [source](../logs/3-distillation/KD_C_aug_T1.log) |
| 4 | [S2_highres](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/S2_highres.pt) | 4,995,226 | 200 | 97.18 | 97.49 | 0.885 | [source](../notebooks/round4_confirm.ipynb) |
| 4 | [S4_deep](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/S4_deep.pt) | 4,552,158 | 200 | 97.07 | 97.27 | 0.886 | [source](../notebooks/round4_confirm.ipynb) |
| 5 | [KD5_pair](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/KD5_pair.pt) | 4,995,226 | 200 | 97.31 | 97.49 | 0.887 | [source](../notebooks/round5_kd.ipynb) |
| 5 | [KD5_full](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/KD5_full.pt) | 4,995,226 | 200 | 97.49 | 97.60 | 0.893 | [source](../notebooks/round5_kd.ipynb) |
| 5 | [KD5_full_cutout](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/KD5_full_cutout.pt) | 4,995,226 | 200 | 97.25 | 97.28 | 0.896 | [source](../notebooks/round5_kd.ipynb) |
| 6 | [R6_base](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/R6_base.pt) | 4,995,226 | 200 | 97.25 | 97.26 | 0.898 | [source](../notebooks/round6_screen.ipynb) |
| 6 | [R6_sam](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/R6_sam.pt) | 4,995,226 | 200 | 97.07 | 97.04 | 0.891 | [source](../notebooks/round6_screen.ipynb) |
| 6 | [R6_sd_ema](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/R6_sd_ema.pt) | 4,995,226 | 200 | 97.03 | 97.03 | 0.887 | [source](../notebooks/round6_screen.ipynb) |
| 6 | [R6_all](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/R6_all.pt) | 4,995,226 | 200 | 96.83 | 96.90 | 0.883 | [source](../notebooks/round6_screen.ipynb) |
| 7 | [R7_zoom](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/R7_zoom.pt) | 4,995,226 | 200 | 96.87 | 97.03 | 0.891 | [source](../logs/7-zoom/R7_zoom.log) |
| 7 | [R7_zoom_strong](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/R7_zoom_strong.pt) | 4,995,226 | 200 | 96.89 | 96.91 | 0.889 | [source](../logs/7-zoom/R7_zoom_strong.log) |
| 7 | [R7_zoom_a7](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.2/R7_zoom_a7.pt) | 4,995,226 | 200 | 97.01 | 97.03 | 0.896 | [source](../logs/7-zoom/R7_zoom_a7.log) |
| 8 | [R8_base_300](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.3/R8_base_300.pt) | 4,995,226 | 300 | 97.25 | 97.46 | 0.895 | [source](../logs/8-duration/R8_base_300.log) |
| 9 | [R9_combined](https://github.com/buddhsen-tripathi/resnet-5m/releases/download/v0.3/R9_combined.pt) | 4,937,794 | 200 | 97.54 | 97.70 | 0.901 | [source](../logs/9-combined/R9_combined.log) |
| 10 | [R10_t12](../weights/R10_t12.pt) | 4,937,794 | 200 | 97.51 | 97.59 | 0.901 | [source](../logs/10-shift/R10_t12.log) |
| 10 | R10_shift | 4,937,794 | 200 | 97.48 | 97.68 | 0.894 | [source](../logs/10-shift/R10_shift.log) |
| 11 | R11_wide | 4,937,794 | 200 | 97.33 | 97.56 | 0.894 | [source](../logs/11-wide-teachers/R11_wide.log) |

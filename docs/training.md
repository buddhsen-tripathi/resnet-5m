# Reproducing the experiments

These commands train students from scratch. Download the teacher checkpoints from the [model releases](../README.md#model-weights) before running distillation. Bash and Zsh are supported. The commands preserve each round's recipe; seeds and hardware differ across rounds.

Rounds 1–3 have per-epoch logs; rounds 4–7 also have Colab notebooks containing the original training code. Rounds 8–9 record the H200 runtime package versions and source hashes in their log directories. These records matter when comparing small accuracy differences across devices.

```bash
uv sync
gh release download v0.1 -p '*.pt' -D checkpoints
gh release download v0.2 -p '*.pt' -D checkpoints
```

## Round 1: architecture

[Results and analysis](../README.md#round-1-architecture)

```bash
uv run python train.py --out checkpoints/A_64-128-256_343_se.pt
uv run python train.py --no-se --out checkpoints/A_64-128-256_343_nose.pt
uv run python train.py --blocks 3 5 3 --out checkpoints/B_64-128-256_353_se.pt
uv run python train.py --widths 80 160 320 --blocks 2 3 2 --out checkpoints/C_80-160-320_232_se.pt
uv run python train.py --widths 96 192 320 --blocks 2 2 2 --out checkpoints/D_96-192-320_222_se.pt
```

## Round 2: augmentation

[Results and analysis](../README.md#round-2-augmentation)

```bash
C=(--widths 80 160 320 --blocks 2 3 2)
uv run python train.py "${C[@]}" --autoaugment --out checkpoints/C_ta.pt
uv run python train.py "${C[@]}" --mix 0.5 --out checkpoints/C_mix.pt
uv run python train.py "${C[@]}" --autoaugment --mix 0.5 --out checkpoints/C_ta_mix.pt
uv run python train.py "${C[@]}" --autoaugment --mix 0.5 --erase-p 0 --out checkpoints/C_ta_mix_noerase.pt
uv run python train.py --no-se --autoaugment --mix 0.5 --out checkpoints/A_nose_ta_mix.pt
```

## Round 3: knowledge distillation

[Results and analysis](../README.md#round-3-knowledge-distillation)

```bash
R1=(checkpoints/A_64-128-256_343_se.pt checkpoints/A_64-128-256_343_nose.pt checkpoints/B_64-128-256_353_se.pt checkpoints/C_80-160-320_232_se.pt checkpoints/D_96-192-320_222_se.pt)
R2=(checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt)
C=(--widths 80 160 320 --blocks 2 3 2)
uv run python train.py "${C[@]}" --teacher "${R1[@]}" --out checkpoints/C_kd5.pt
uv run python train.py "${C[@]}" --autoaugment --mix 0.5 --erase-p 0 --teacher "${R2[@]}" --out checkpoints/KD_C_aug.pt
uv run python train.py --no-se --autoaugment --mix 0.5 --erase-p 0 --teacher "${R2[@]}" --out checkpoints/KD_A_nose_aug.pt
uv run python train.py "${C[@]}" --autoaugment --mix 0.5 --erase-p 0 --teacher "${R2[@]}" --kd-temp 1 --kd-alpha 1.0 --out checkpoints/KD_C_aug_T1.pt
```

## Round 4: architecture under strong augmentation

[Results and analysis](../README.md#round-4-architecture-under-strong-augmentation)

```bash
A=(--autoaugment --mix 0.5 --erase-p 0 --workers 6)
# Screen all five architectures for 60 epochs.
uv run python train.py --widths 80 160 320 --blocks 2 3 2 "${A[@]}" --epochs 60 --out checkpoints/screen_S0_C.pt
uv run python train.py --widths 96 168 232 --blocks 3 3 3 "${A[@]}" --epochs 60 --out checkpoints/screen_S1_mid.pt
uv run python train.py --widths 128 192 232 --blocks 4 3 2 --no-se "${A[@]}" --epochs 60 --out checkpoints/screen_S2_highres.pt
uv run python train.py --block bottleneck --widths 256 512 768 --blocks 5 5 3 "${A[@]}" --epochs 60 --out checkpoints/screen_S3_bneck.pt
uv run python train.py --block bottleneck --widths 256 448 640 --blocks 7 7 4 "${A[@]}" --epochs 60 --out checkpoints/screen_S4_deep.pt

# Train the selected architectures from scratch for 200 epochs.
uv run python train.py --widths 128 192 232 --blocks 4 3 2 --no-se "${A[@]}" --epochs 200 --out checkpoints/S2_highres.pt
uv run python train.py --block bottleneck --widths 256 448 640 --blocks 7 7 4 "${A[@]}" --epochs 200 --out checkpoints/S4_deep.pt
```

## Round 5: distill into the S2 architecture

[Results and analysis](../README.md#round-5-distill-into-the-s2-architecture)

```bash
A=(--widths 128 192 232 --blocks 4 3 2 --no-se --autoaugment --mix 0.5 --erase-p 0 --seed 1 --teacher-flip --epochs 200 --workers 4)
PAIR=(checkpoints/S2_highres.pt checkpoints/KD_C_aug.pt)
FULL=("${PAIR[@]}" checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt)
uv run python train.py "${A[@]}" --teacher "${PAIR[@]}" --out checkpoints/KD5_pair.pt
uv run python train.py "${A[@]}" --teacher "${FULL[@]}" --out checkpoints/KD5_full.pt
uv run python train.py "${A[@]}" --teacher "${FULL[@]}" --cutout --out checkpoints/KD5_full_cutout.pt
```

## Round 6: regularization screen

[Results and analysis](../README.md#round-6-regularization-screen)

```bash
A=(--widths 128 192 232 --blocks 4 3 2 --no-se --autoaugment --mix 0.5 --erase-p 0 --cutout --seed 1 --teacher-flip --epochs 200 --workers 3)
TEACHERS=(
  checkpoints/S2_highres.pt checkpoints/KD_C_aug.pt
  checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt
  checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt
  checkpoints/S4_deep.pt checkpoints/KD5_full.pt checkpoints/KD5_full_cutout.pt
)
uv run python train.py "${A[@]}" --teacher "${TEACHERS[@]}" --out checkpoints/R6_base.pt
uv run python train.py "${A[@]}" --teacher "${TEACHERS[@]}" --drop-path 0.1 --ema 0.9995 --out checkpoints/R6_sd_ema.pt
uv run python train.py "${A[@]}" --teacher "${TEACHERS[@]}" --sam 0.05 --out checkpoints/R6_sam.pt
uv run python train.py "${A[@]}" --teacher "${TEACHERS[@]}" --drop-path 0.1 --ema 0.9995 --sam 0.05 --out checkpoints/R6_all.pt
```

## Round 7: zoom augmentation

[Results and analysis](../README.md#round-7-zoom-augmentation)

```bash
A=(--widths 128 192 232 --blocks 4 3 2 --no-se --autoaugment --mix 0.5 --erase-p 0 --cutout --seed 1 --teacher-flip --epochs 200 --workers 4)
TEACHERS=(
  checkpoints/S2_highres.pt checkpoints/KD_C_aug.pt
  checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt
  checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt
  checkpoints/S4_deep.pt checkpoints/KD5_full.pt checkpoints/KD5_full_cutout.pt
)
uv run python train.py "${A[@]}" --teacher "${TEACHERS[@]}" --zoom 0.75 1.33 --out checkpoints/R7_zoom.pt
uv run python train.py "${A[@]}" --teacher "${TEACHERS[@]}" --zoom 0.5 1.5 --out checkpoints/R7_zoom_strong.pt
uv run python train.py "${A[@]}" --teacher "${TEACHERS[@]}" --zoom 0.75 1.33 --kd-alpha 0.7 --out checkpoints/R7_zoom_a7.pt
```

## Round 8: training duration

[Results and analysis](../README.md#round-8-training-duration)

```bash
TEACHERS=(
  checkpoints/S2_highres.pt checkpoints/KD_C_aug.pt
  checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt
  checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt
  checkpoints/S4_deep.pt checkpoints/KD5_full.pt checkpoints/KD5_full_cutout.pt
)
uv run python train.py --widths 128 192 232 --blocks 4 3 2 --no-se \
  --autoaugment --mix 0.5 --erase-p 0 --cutout \
  --teacher-flip --kd-alpha 0.9 --kd-temp 4 --label-smoothing 0.1 \
  --epochs 300 --seed 1 --workers 3 \
  --teacher "${TEACHERS[@]}" --out checkpoints/R8_base_300.pt
```

## Round 9: combined data-focused recipe

[Results and analysis](../README.md#round-9-combined-data-focused-recipe)

```bash
TEACHERS=(
  checkpoints/S2_highres.pt checkpoints/KD_C_aug.pt
  checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt
  checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt
  checkpoints/S4_deep.pt checkpoints/KD5_full.pt checkpoints/KD5_full_cutout.pt
)
uv run python train.py --widths 128 192 228 --blocks 4 3 2 --no-se \
  --augmix --mix 0.5 --erase-p 0 --cutout --cutout-size 8 \
  --supcon 0.1 --supcon-temp 0.07 --teacher-flip --kd-alpha 0.7 --kd-temp 4 \
  --label-smoothing 0 --epochs 200 --seed 1 --workers 8 \
  --teacher "${TEACHERS[@]}" --out checkpoints/R9_combined.pt
```

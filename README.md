# resnet-5m

A parameter-efficient ResNet for CIFAR-10, trained from scratch under a **5M trainable-parameter budget**.

Built for the [CIFAR-10 ResNet Architecture Challenge](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl) (NYU Tandon Deep Learning, Fall 2026): highest accuracy on a held-out image set, at most 5M trainable parameters, trained only on the 50k CIFAR-10 training images, no externally pretrained weights or external training data.

## Results

Our best Kaggle and CIFAR-10 scores come from `R9_combined`, a **4,937,794-parameter pre-activation ResNet distilled from an ensemble of ten of our own models**. Its training-only contrastive head brings the total trainable parameters to 4,983,618:

| run | CIFAR-10 test | flip TTA | Kaggle public |
|-----|---------------|----------|---------------|
| `R6_base` | 97.25% | 97.26% | 0.898 |
| `KD5_full_cutout` | 97.25% | 97.28% | 0.896 |
| `R8_base_300` | 97.25% | 97.46% | 0.895 |
| `KD5_full` | 97.49% | 97.60% | 0.893 |
| `R9_combined` | **97.54%** | **97.70%** | **0.901** |

`R9_combined` is the checkpoint to use for the best recorded Kaggle public score and CIFAR-10 test accuracy. It is a single model under 5M parameters. Ensembles are used as teachers, with only the student used for submission.

| Round | Question | Highest Kaggle public score in the round | Run |
|-------|----------|------------------------------------------|-----|
| [1](#round-1-architecture) | Depth, width and SE | 0.873 | A with SE |
| [2](#round-2-augmentation) | Stronger augmentation | 0.884 | `C_ta_mix_noerase` |
| [3](#round-3-knowledge-distillation) | Ensemble distillation | 0.889 | `KD_C_aug` |
| [4](#round-4-architecture-under-strong-augmentation) | Architecture with strong augmentation | 0.886 | `S4_deep` |
| [5](#round-5-distill-into-the-s2-architecture) | More teachers and Cutout | 0.896 | `KD5_full_cutout` |
| [6](#round-6-regularization-screen) | SAM, EMA and stochastic depth | 0.898 | `R6_base` |
| [7](#round-7-zoom-augmentation) | Zoom and distillation weight | 0.896 | `R7_zoom_a7` |
| [8](#round-8-training-duration) | 200 vs. 300 epochs | 0.895 | `R8_base_300` |
| [9](#round-9-combined-data-focused-recipe) | Smaller Cutout, softer KD, AugMix and contrastive learning | **0.901** | `R9_combined` |

Stronger augmentation improved CIFAR-10 accuracy by 0.5-0.75 percentage points in round 2. Distillation with strong augmentation added another 0.2-0.5 points in round 3. The higher-resolution S2 architecture and larger teacher sets were used in later rounds. Additional regularization and zoom did not beat `R6_base` in the tested recipes.

These are single-seed experiments. Small leaderboard gaps should be treated cautiously, and higher CIFAR-10 test accuracy did not consistently translate into a higher Kaggle score in rounds 4-8.

## Pretrained weights

The best model, `R9_combined` (0.901 Kaggle public), ships with the repo, so it can be used right after cloning. The round 8 duration experiment and earlier baselines are also available in `weights/`:

```bash
uv sync
uv run python evaluate.py weights/R9_combined.pt       # 97.70% with flip TTA
uv run python predict.py --checkpoint weights/R9_combined.pt
uv run python evaluate.py weights/R8_base_300.pt       # 97.46% with flip TTA
```

`weights/R8_base_300.pt` and `weights/R9_combined.pt` contain the exact student weights used for their Kaggle submissions. Configurations, counts and SHA-256 checksums are recorded in [`weights/round8-9.json`](weights/round8-9.json). These inference exports omit optimizer state and the R9 training-only projection head; they support evaluation, prediction and use as teachers, but cannot be resumed with `--resume`.

`weights/KD_C_aug.pt`, the best submitted model from rounds 1-3, is also included (97.31% CIFAR-10 accuracy with flip TTA, 0.889 Kaggle public). Evaluation downloads CIFAR-10 automatically. Prediction additionally requires the competition test files described in [Kaggle notes](#kaggle-notes).

Weights are attached to two releases, and file names match the run names in the experiment tables below. The [v0.1 release](https://github.com/buddhsen-tripathi/resnet-5m/releases/tag/v0.1) holds all 14 runs from rounds 1-3 (~19 MB each), and the [v0.2 release](https://github.com/buddhsen-tripathi/resnet-5m/releases/tag/v0.2) holds 12 runs from rounds 4-7 (~20 MB each). The 60-epoch screening runs S0_C, S1_mid and S3_bneck are not included:

```bash
gh release download v0.1 -D checkpoints                  # all runs from rounds 1-3
gh release download v0.2 -D checkpoints                  # runs from rounds 4-7
gh release download v0.2 -p "R6_*.pt" -D checkpoints     # a subset
```

They load in `predict.py`, `evaluate.py` and as distillation teachers (`--teacher`). They hold the weights and config only, not the optimizer state, so `--resume` cannot continue them; use them for inference or as teachers for a new student. Loading existing weights as a starting point for fine-tuning is not currently supported by `train.py`.

## Usage

```bash
uv sync
uv run python train.py --out checkpoints/model.pt   # downloads CIFAR-10 to data/
uv run python evaluate.py weights/R6_base.pt weights/KD_C_aug.pt   # each model and their ensemble
uv run python train.py --teacher weights/R6_base.pt --out checkpoints/student.pt
uv run python predict.py --checkpoint checkpoints/student.pt
```

`predict.py` reads `data/kaggle/test.csv` and `data/kaggle/test_images/`, averages each image with its horizontal flip (`--shift 1` also averages over 1-pixel translations and `--zoom 0.8 1.25` over scale factors; `evaluate.py` takes the same flags), and writes `submission.csv` (`id,label`). Checkpoints store the model config, so any run can be loaded without extra flags. `train.py --resume` continues from the `--out` checkpoint after an interruption; repeat the original architecture and training arguments with `--resume` added. The checkpoint must contain optimizer and scheduler state. CUDA, Apple MPS and CPU are selected automatically in that order; a CUDA GPU is recommended for the full experiments.

The default `train.py` command reproduces the original A architecture and baseline recipe. The best submitted model uses a slightly narrower S2 architecture and ten teachers; see the complete [round 9 command](#round-9-combined-data-focused-recipe). For Colab, use the notebooks linked in each round; for NYU HPC, see [`hpc/README.md`](hpc/README.md).

## Model

A configurable pre-activation ResNet with optional squeeze-and-excitation (SE) and ResNet-D shortcuts:

- **Pre-activation blocks** (BN-ReLU-conv x2, He et al. 2016): the skip path stays a pure identity, so gradients reach early layers unchanged.
- **SE channel attention**, when enabled, in every block (Hu et al. 2018): global context reweights channels for <1% extra parameters.
- **ResNet-D downsampling shortcut** (2x2 avg-pool then 1x1 conv, He et al. 2019): pooling aggregates each 2x2 region before projection.
- 3x3 stem, final BN-ReLU, global average pooling, linear classifier. Kaiming init. Input normalization lives inside the model.

Widths, blocks per stage and block type are configurable (`--widths`, `--blocks`, `--no-se`, `--block bottleneck`); each stage after the first halves the resolution (32 -> 16 -> 8). `train.py` refuses any configuration over 5M trainable parameters, including the optional contrastive projection head.

| config | block type | widths | blocks | SE | params |
|--------|------------|--------|--------|----|--------|
| A (CLI default) | basic | 64-128-256 | 3-4-3 | yes | 4,658,278 |
| A, no SE | basic | 64-128-256 | 3-4-3 | no | 4,622,410 |
| B | basic | 64-128-256 | 3-5-3 | yes | 4,955,886 |
| C | basic | 80-160-320 | 2-3-2 | yes | 4,836,010 |
| D | basic | 96-192-320 | 2-2-2 | yes | 4,939,894 |
| S2 (`R6_base`, `KD5_full`) | basic | 128-192-232 | 4-3-2 | no | 4,995,226 |
| S4 | bottleneck | 256-448-640 | 7-7-4 | yes | 4,552,158 |

`R6_base` uses S2 without SE, stochastic depth, EMA or SAM. Its training recipe adds TrivialAugment, CutMix/MixUp on half the batches, Cutout and flip-averaged distillation from ten teachers (temperature 4, distillation weight 0.9).

## Training recipe

- SGD (momentum 0.9, Nesterov), lr 0.2, batch 256, 5-epoch linear warmup then cosine decay, 200 epochs
- Weight decay 5e-4 (not on BN/bias), label smoothing 0.1
- Augmentation: random crop (reflect pad 4), horizontal flip, optional random zoom (`--zoom MIN MAX`), random erasing (`--erase-p`) or a fixed Cutout square (`--cutout`, side length `--cutout-size`, default 16); optional TrivialAugment (`--autoaugment`) or AugMix transforms (`--augmix`, without a separate Jensen-Shannon loss), and per-batch CutMix/MixUp (`--mix 0.5`)
- Knowledge distillation (`--teacher ckpt [ckpt ...]`): the student matches the averaged logits of the teachers on the *same augmented batch* (KL at temperature `--kd-temp`, weighted by `--kd-alpha` against the label loss); `--teacher-flip` also averages the teachers over the horizontal flip
- Supervised contrastive learning (`--supcon WEIGHT`, temperature `--supcon-temp`, default 0.07): two independently augmented, unmixed views share their original class labels. A 128-dimensional MLP projection head is trained alongside the classifier and saved for resuming, but is unused for inference. CutMix/MixUp remains confined to the classification/distillation branch. The implementation concatenates classification and contrastive views into one student forward pass, so they share batch-normalization statistics.
- Optional regularizers: stochastic depth (`--drop-path`), weight EMA (`--ema`, evaluates and saves the EMA model) and SAM (`--sam`, about twice the compute, not supported with float16 loss scaling)
- bf16 mixed precision (float16 with loss scaling on GPUs without bf16) and channels-last on CUDA; checkpoint written atomically every epoch

The official 10k CIFAR-10 test split is evaluated after each epoch and never used for gradient updates. The saved checkpoint is overwritten after each epoch, so it contains the final model rather than the best-scoring epoch. Test accuracy has informed choices between experiments, so these results are not an independent final evaluation of the selected recipe. Training uses only the 50k training images; CIFAR-10 test images and competition test images are excluded from training, including pseudo-labeling.

## Experiments

Rounds 1-7 and 9 use **200 epochs** for full runs; round 8 tests **300 epochs**. Round 4 first screens candidates for 60 epochs, then trains the selected architectures from scratch for 200; the screening and confirmation results are reported separately. Rounds 1-4 use seed 0; rounds 5-9 use seed 1. Hardware, worker counts and teacher sets also vary across rounds, so comparisons across rounds do not isolate a single change.

Rounds 1-3 ran on one NVIDIA B300, with full per-epoch logs in [`logs/`](logs). Rounds 4-7 used Colab A100 notebooks in [`notebooks/`](notebooks); only round 7's per-epoch logs are currently committed alongside the round 1-3 logs. The notebooks embed the model and training code used for their experiments.

Rounds 8-9 ran on one H200. Their logs include the runtime package versions, recipe manifests, full-precision evaluation and Kaggle submission records. Recreate their unsmoothed training curves with `uv run --no-project --with matplotlib python scripts/plot_training.py`; Matplotlib is only needed for plotting.

*Final* is test accuracy after the last epoch, *best* is the highest observed epoch, and *TTA* is the final checkpoint averaged with its horizontal flip (`evaluate.py`). CIFAR-10 values are percentages; Kaggle scores are fractions. Training evaluation uses CUDA mixed precision, while `evaluate.py` uses full precision, so small differences between devices or evaluation paths are possible.

Commands below use shell arrays and work in Bash and Zsh. Each round defines its own arguments. Download the required teacher checkpoints from the releases before running distillation.

### Round 1: architecture

How to spend 5M parameters: deeper (B) vs. wider (C, D), and whether SE helps (A with/without).

```bash
uv run python train.py --out checkpoints/A_64-128-256_343_se.pt
uv run python train.py --no-se --out checkpoints/A_64-128-256_343_nose.pt
uv run python train.py --blocks 3 5 3 --out checkpoints/B_64-128-256_353_se.pt
uv run python train.py --widths 80 160 320 --blocks 2 3 2 --out checkpoints/C_80-160-320_232_se.pt
uv run python train.py --widths 96 192 320 --blocks 2 2 2 --out checkpoints/D_96-192-320_222_se.pt
```

| run | widths | blocks | SE | params | final | best | TTA |
|-----|--------|--------|----|--------|-------|------|-----|
| A | 64-128-256 | 3-4-3 | yes | 4,658,278 | 95.81 | 95.90 | 96.21 |
| A | 64-128-256 | 3-4-3 | no | 4,622,410 | 96.19 | 96.25 | 96.55 |
| B | 64-128-256 | 3-5-3 | yes | 4,955,886 | **96.23** | 96.23 | **96.58** |
| C | 80-160-320 | 2-3-2 | yes | 4,836,010 | 96.18 | **96.26** | 96.36 |
| D | 96-192-320 | 2-2-2 | yes | 4,939,894 | 96.06 | 96.11 | 96.21 |
| ensemble of the 5 | | | | | | | 97.06 |

![Round 1 training curves](assets/1-architecture.png)

- All five land within 0.42 pts. Depth vs. width barely matters at this budget.
- A with SE finished lowest, but B, C and D (all with SE) match A without SE, so with one seed per config the SE effect is indistinguishable from run-to-run noise.
- Flip TTA adds 0.15-0.4 pts to every model.
- Kaggle public score of A (SE): **0.873**, about 8.5 points below its CIFAR-10 test accuracy, suggesting a distribution shift between the two evaluation sets.

### Round 2: augmentation

Same budget, stronger regularization: TrivialAugment (per-image), CutMix/MixUp (per-batch, half the batches), and whether random erasing is still useful on top.

```bash
C=(--widths 80 160 320 --blocks 2 3 2)
uv run python train.py "${C[@]}" --autoaugment --out checkpoints/C_ta.pt
uv run python train.py "${C[@]}" --mix 0.5 --out checkpoints/C_mix.pt
uv run python train.py "${C[@]}" --autoaugment --mix 0.5 --out checkpoints/C_ta_mix.pt
uv run python train.py "${C[@]}" --autoaugment --mix 0.5 --erase-p 0 --out checkpoints/C_ta_mix_noerase.pt
uv run python train.py --no-se --autoaugment --mix 0.5 --out checkpoints/A_nose_ta_mix.pt
```

| run | augmentation | params | final | best | TTA |
|-----|--------------|--------|-------|------|-----|
| C | baseline (round 1) | 4,836,010 | 96.18 | 96.26 | 96.36 |
| C | + TrivialAugment | 4,836,010 | 96.76 | 96.86 | 96.84 |
| C | + CutMix/MixUp | 4,836,010 | 96.68 | 96.73 | **97.01** |
| C | + both | 4,836,010 | 96.84 | 96.88 | 96.98 |
| C | + both, no erasing | 4,836,010 | **96.93** | **96.97** | 96.97 |
| A (no SE) | + both | 4,622,410 | 96.77 | 96.83 | 97.00 |
| ensemble of the 5 | | | | | 97.50 |

![Round 2 training curves](assets/2-augmentation.png)

- Every augmentation setting beats its baseline by 0.5-0.75 pts at the same 200 epochs; combining TrivialAugment with CutMix/MixUp is best.
- Strong augmentation trains slower early (lower accuracy until ~epoch 150) and catches up once the learning rate decays.
- Random erasing on top of both is neutral.
- Kaggle public score of C + both, no erasing: **0.884** (+1.1 over round 1).

### Round 3: knowledge distillation

The ensembles are better than any single model but are 5x over the budget. Distillation moves that accuracy into one model under 5M: the student is trained on the averaged logits of the teacher ensemble, computed on the same augmented batch it sees. Teachers are our own models trained from scratch on the same data, as the rules allow.

```bash
R1=(checkpoints/A_64-128-256_343_se.pt checkpoints/A_64-128-256_343_nose.pt checkpoints/B_64-128-256_353_se.pt checkpoints/C_80-160-320_232_se.pt checkpoints/D_96-192-320_222_se.pt)
R2=(checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt)
C=(--widths 80 160 320 --blocks 2 3 2)
uv run python train.py "${C[@]}" --teacher "${R1[@]}" --out checkpoints/C_kd5.pt
uv run python train.py "${C[@]}" --autoaugment --mix 0.5 --erase-p 0 --teacher "${R2[@]}" --out checkpoints/KD_C_aug.pt
uv run python train.py --no-se --autoaugment --mix 0.5 --erase-p 0 --teacher "${R2[@]}" --out checkpoints/KD_A_nose_aug.pt
uv run python train.py "${C[@]}" --autoaugment --mix 0.5 --erase-p 0 --teacher "${R2[@]}" --kd-temp 1 --kd-alpha 1.0 --out checkpoints/KD_C_aug_T1.pt
```

| run | augmentation | teacher | KD loss | params | final | best | TTA | same run without KD |
|-----|--------------|---------|---------|--------|-------|------|-----|---------------------|
| C | baseline | round 1 (97.06%) | T=4, alpha=0.9 | 4,836,010 | 96.13 | 96.13 | 96.33 | 96.18 |
| C | both, no erasing | round 2 (97.50%) | T=4, alpha=0.9 | 4,836,010 | 97.22 | 97.28 | **97.31** | 96.93 |
| A (no SE) | both, no erasing | round 2 (97.50%) | T=4, alpha=0.9 | 4,622,410 | **97.27** | **97.31** | 97.23 | 96.77 |
| C | both, no erasing | round 2 (97.50%) | T=1, alpha=1.0 | 4,836,010 | 97.12 | 97.15 | 97.25 | 96.93 |

![Round 3 training curves](assets/3-distillation.png)

- With baseline augmentation, distillation gives nothing (96.13 vs. 96.18): its early lead disappears once the learning rate decays.
- With strong augmentation and the stronger teacher it adds +0.2 to +0.5 pts, pushing a single sub-5M model past 97.2%. This round changes both augmentation and teacher quality relative to the baseline distillation run.
- Softened (T=4) vs. raw (T=1) teacher targets are within noise.
- Flip TTA has a smaller or negative effect on these distilled students; its benefit depends on the checkpoint.
- Kaggle public score of the distilled C: **0.889** (+0.5 over round 2).

![Final CIFAR-10 test accuracy of every run in rounds 1-3](assets/summary.png)

### Round 4: architecture under strong augmentation

Round 1 compared architectures with weak augmentation. This round repeats the search with TrivialAugment and CutMix/MixUp, including higher-resolution stages and bottleneck blocks. Five candidates were screened for 60 epochs ([`notebooks/round4_screening.ipynb`](notebooks/round4_screening.ipynb)), and the top two were confirmed for 200 ([`notebooks/round4_confirm.ipynb`](notebooks/round4_confirm.ipynb)).

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

| run | block | widths | blocks | SE | params | 60 epochs (final) | 60 epochs (best) | 200 epochs (final) | TTA | Kaggle |
|-----|-------|--------|--------|----|--------|-------------------|------------------|--------------------|-----|--------|
| S0_C | basic | 80-160-320 | 2-3-2 | yes | 4,836,010 | 95.39 | 95.50 | | | |
| S1_mid | basic | 96-168-232 | 3-3-3 | yes | 4,786,532 | 95.64 | 95.67 | | | |
| S2_highres | basic | 128-192-232 | 4-3-2 | no | 4,995,226 | **95.90** | **95.93** | **97.18** | **97.49** | 0.885 |
| S3_bneck | bottleneck | 256-512-768 | 5-5-3 | yes | 4,529,290 | 95.42 | 95.46 | | | |
| S4_deep | bottleneck | 256-448-640 | 7-7-4 | yes | 4,552,158 | 95.89 | 95.93 | 97.07 | 97.27 | **0.886** |

- Without any teacher, S2 nearly matches the distilled C on CIFAR-10 test (97.18 vs. 97.22) and beats it with flip TTA (97.49 vs. 97.31).
- Neither model beats the distilled C on Kaggle (0.889), so better CIFAR-10 accuracy did not carry over here.
- Differences of a few thousandths are close to the noise of the public leaderboard, which scores only part of the 2,000 test images.

### Round 5: distill into the S2 architecture

S2 is the best architecture without KD, so it became the student. The teachers are now our own models from rounds 2-4, averaged over the horizontal flip (`--teacher-flip`). `KD5_pair` uses S2 and the distilled C, while the other two runs use seven teachers. The notebook is [`notebooks/round5_kd.ipynb`](notebooks/round5_kd.ipynb).

```bash
A=(--widths 128 192 232 --blocks 4 3 2 --no-se --autoaugment --mix 0.5 --erase-p 0 --seed 1 --teacher-flip --epochs 200 --workers 4)
PAIR=(checkpoints/S2_highres.pt checkpoints/KD_C_aug.pt)
FULL=("${PAIR[@]}" checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt)
uv run python train.py "${A[@]}" --teacher "${PAIR[@]}" --out checkpoints/KD5_pair.pt
uv run python train.py "${A[@]}" --teacher "${FULL[@]}" --out checkpoints/KD5_full.pt
uv run python train.py "${A[@]}" --teacher "${FULL[@]}" --cutout --out checkpoints/KD5_full_cutout.pt
```

| run | teachers | Cutout | params | final | TTA | Kaggle |
|-----|----------|--------|--------|-------|-----|--------|
| KD5_pair | 2 | no | 4,995,226 | 97.31 | 97.49 | 0.887 |
| KD5_full | 7 | no | 4,995,226 | **97.49** | **97.60** | 0.893 |
| KD5_full_cutout | 7 | yes | 4,995,226 | 97.25 | 97.28 | **0.896** |

- Seven teachers beat two on both measures: +0.18 pts on CIFAR-10 test and +0.006 on Kaggle.
- Both seven-teacher students beat the previous best of 0.889 on Kaggle.
- Cutout lowers CIFAR-10 test accuracy by 0.24 pts but scores 0.003 higher on Kaggle, about 3 images on the public leaderboard. With one seed, that is too small to call either way.

### Round 6: regularization screen

Ten teachers (the seven from round 5, plus S4, KD5_full and KD5_full_cutout), with Cutout on every run. The notebook is [`notebooks/round6_screen.ipynb`](notebooks/round6_screen.ipynb).

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

| run | added | params | final | best | TTA | Kaggle |
|-----|-------|--------|-------|------|-----|--------|
| R6_base | none | 4,995,226 | **97.25** | **97.27** | **97.26** | **0.898** |
| R6_sam | SAM | 4,995,226 | 97.07 | 97.17 | 97.04 | 0.891 |
| R6_sd_ema | stochastic depth, EMA | 4,995,226 | 97.03 | 97.03 | 97.03 | 0.887 |
| R6_all | all three | 4,995,226 | 96.83 | 96.86 | 96.90 | 0.883 |

- SAM and the combined stochastic-depth/EMA setting each lowered both CIFAR-10 accuracy and the Kaggle score; stacking all three was worst. This is consistent with excessive regularization in this 200-epoch recipe. Stochastic depth and EMA were not tested separately.
- R6_base scores +0.002 over KD5_full_cutout on Kaggle at identical final accuracy (97.25), which is within leaderboard noise.
- Flip plus 1-pixel shift TTA left KD5_full_cutout at 0.896.

### Round 7: zoom augmentation

Many of the hidden images our models are least sure about show tiny objects, close-ups or unusual scenes. This round tests whether random zoom helps, along with a softer KD weight. The notebook is [`notebooks/round7_zoom.ipynb`](notebooks/round7_zoom.ipynb), and the per-epoch logs are in [`logs/7-zoom`](logs/7-zoom).

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

| run | zoom | KD alpha | params | final | best | TTA | Kaggle |
|-----|------|----------|--------|-------|------|-----|--------|
| R6_base | none | 0.9 | 4,995,226 | **97.25** | **97.27** | **97.26** | **0.898** |
| R7_zoom | 0.75-1.33 | 0.9 | 4,995,226 | 96.87 | 96.96 | 97.03 | 0.891 |
| R7_zoom_strong | 0.5-1.5 | 0.9 | 4,995,226 | 96.89 | 96.90 | 96.91 | 0.889 |
| R7_zoom_a7 | 0.75-1.33 | 0.7 | 4,995,226 | 97.01 | 97.06 | 97.03 | 0.896 |

![Round 7 training curves](assets/7-zoom.png)

- The tested random-zoom settings lowered both CIFAR-10 accuracy and the Kaggle score relative to `R6_base`. They did not resolve the gap between evaluation sets.
- Zoom TTA on R6_base also hurt, scoring 0.883.
- With mild zoom, the softer KD weight added 0.14 pts on CIFAR-10 and +0.005 on Kaggle. That is suggestive, but one seed cannot separate it from leaderboard noise.

### Round 8: training duration

The `R6_base` recipe was trained from scratch for 300 epochs, with its cosine schedule extended across all 300. Architecture, ten teachers, augmentation and seed 1 were retained. The run used one H200, took 82.74 minutes of epoch time, and resumed from epoch 113 after the VM was stopped. Full logs and runtime details are in [`logs/8-duration/`](logs/8-duration/).

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

| run | epochs | no TTA | flip TTA | Kaggle public |
|-----|--------|--------|----------|---------------|
| R6_base | 200 | 97.26 | 97.26 | **0.898** |
| R8_base_300 | 300 | 97.25 | **97.46** | 0.895 |

Both checkpoints were re-evaluated in full precision on the H200 for this table; the original mixed-precision training evaluation reports R6_base at 97.25%. The additional epochs improved flip-TTA accuracy by 0.20 percentage points, but did not improve the public Kaggle score. This is one run on different hardware/software from the original Colab baseline, not evidence that longer training never helps.

![R8 training loss and CIFAR-10 test accuracy across all 300 epochs](assets/8-duration.png)

### Round 9: combined data-focused recipe

One combined 200-epoch experiment, rather than separate ablations: 8x8 Cutout, KD weight 0.7, no label smoothing, AugMix transforms replacing TrivialAugment, and supervised contrastive loss with weight 0.1 and temperature 0.07. It uses the same ten frozen teachers as round 6 and seed 1. AugMix is used as an image transform; there is no separate AugMix Jensen-Shannon consistency penalty.

The last stage is narrowed from 232 to 228 channels so the 128-dimensional, two-layer contrastive projection head fits the budget. The classifier has **4,937,794 parameters**, the projection head **45,824**, and all trainable parameters total **4,983,618**. Teachers are frozen and the projection head is unused during inference.

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

The contrastive objective uses the original class labels and two unmixed views; classification and distillation use the potentially mixed batch. New checkpoints retain the projection head, optimizer, scheduler and main-process CPU/CUDA RNG states for resuming. Persistent data-loader workers' augmentation streams are not restored bit-for-bit after interruption.

Training completed all 200 epochs on one H200. Full-precision evaluation reached **97.54% without TTA** and **97.70% with horizontal-flip averaging**, compared with 97.26%/97.26% for R6_base and 97.25%/97.46% for R8_base_300 in the same evaluation. The flip-averaged R9 submission scored **0.901 on Kaggle public**, beating R6_base's 0.898 by 0.3 percentage points (six additional correct predictions out of 2,000). Results, submission metadata and runtime records are in [`logs/9-combined/`](logs/9-combined/). The combined experiment cannot attribute any gain or loss to an individual change.

| run | no TTA | flip TTA | Kaggle public |
|-----|--------|----------|---------------|
| R6_base | 97.26 | 97.26 | 0.898 |
| R8_base_300 | 97.25 | 97.46 | 0.895 |
| R9_combined | **97.54** | **97.70** | **0.901** |

![R9 total training objective and CIFAR-10 test accuracy across all 200 epochs](assets/9-combined.png)

The curves show unsmoothed per-epoch mixed-precision evaluation without TTA. R9's total loss includes the contrastive term, so its loss magnitude is not directly comparable with R8's.

## Kaggle notes

- The test set is 2,000 32x32 PNGs plus `test.csv` (`id,filename`); submissions are `id,label` with CIFAR-10 class indices. These files and account credentials are not included in the repository.
- For local prediction, place `test.csv` in `data/kaggle/` and the PNGs in `data/kaggle/test_images/`, or pass the containing directory with `--data-dir`. `predict.py` writes a CSV locally; it does not submit it to Kaggle.
- Bulk download is not available for this competition, and per-file API downloads are rate-limited (HTTP 429). Predictions can instead be made inside a private Kaggle notebook with the competition data attached and the checkpoint uploaded as a private dataset, running `predict.py --data-dir <competition input dir>`.

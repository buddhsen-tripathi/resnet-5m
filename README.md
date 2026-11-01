# resnet-5m

A parameter-efficient ResNet for CIFAR-10, trained from scratch under a **5M trainable-parameter budget**.

Built for the [CIFAR-10 ResNet Architecture Challenge](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl) (NYU Tandon Deep Learning, Fall 2026): highest accuracy on a held-out image set, at most 5M trainable parameters, trained only on the 50k CIFAR-10 training images, no pretrained weights or external data.

## Results

![Final CIFAR-10 test accuracy of every run in rounds 1-3](assets/summary.png)

Our best Kaggle score comes from a **4,995,226-parameter pre-activation ResNet distilled from an ensemble of ten of our own models**. `KD5_full` has the best CIFAR-10 accuracy:

| run | CIFAR-10 test | flip TTA | Kaggle public |
|-----|---------------|----------|---------------|
| `R6_base` | 97.25% | 97.26% | **0.898** |
| `KD5_full_cutout` | 97.25% | 97.28% | 0.896 |
| `KD5_full` | **97.49%** | **97.60%** | 0.893 |

Gaps this small are a few images on the public leaderboard.

What moved the needle, in order: stronger data augmentation (+0.5 to +0.75 pts), then knowledge distillation on top of it (+0.2 to +0.5 pts). Round 5 showed that a larger teacher set also matters on Kaggle, while round 6 showed that adding stochastic depth, EMA and SAM to this recipe hurts. Architecture choices within the budget mattered least (all within 0.5 pts), and from round 4 on, CIFAR-10 test accuracy stopped predicting the Kaggle score reliably.

## Pretrained weights

The best model from rounds 1-3 ships with the repo, so it can be used right after cloning:

```bash
uv run python evaluate.py weights/KD_C_aug.pt          # 97.31% with flip TTA
uv run python predict.py --checkpoint weights/KD_C_aug.pt
```

Weights for all 14 runs from rounds 1-3 (~19 MB each) are attached to the [v0.1 release](https://github.com/buddhsen-tripathi/resnet-5m/releases/tag/v0.1). File names match the run names in the experiment tables below:

```bash
gh release download v0.1 -D checkpoints                  # all runs
gh release download v0.1 -p "C_*.pt" -D checkpoints      # a subset
```

They load in `predict.py`, `evaluate.py` and as distillation teachers (`--teacher`). They hold the weights and config only, not the optimizer state, so `--resume` cannot continue them; start a new run instead.

## Usage

```bash
uv sync
uv run python train.py --out checkpoints/model.pt   # downloads CIFAR-10 to data/
uv run python evaluate.py checkpoints/a.pt checkpoints/b.pt   # flip-TTA accuracy of each and of their ensemble
uv run python train.py --teacher checkpoints/a.pt checkpoints/b.pt --out checkpoints/student.pt
uv run python predict.py --checkpoint checkpoints/student.pt
```

`predict.py` reads `data/kaggle/test.csv` and `data/kaggle/test_images/`, averages each image with its horizontal flip (`--shift 1` also averages over 1-pixel translations; `evaluate.py` takes the same flag), and writes `submission.csv` (`id,label`). Checkpoints store the model config, so any run can be loaded without extra flags. `train.py --resume` continues from the `--out` checkpoint after an interruption.

## Model

A pre-activation ResNet with squeeze-and-excitation (SE) and ResNet-D shortcuts:

- **Pre-activation blocks** (BN-ReLU-conv x2, He et al. 2016): the skip path stays a pure identity, so gradients reach early layers unchanged.
- **SE channel attention** in every block (Hu et al. 2018): global context reweights channels for <1% extra parameters.
- **ResNet-D downsampling shortcut** (2x2 avg-pool then 1x1 conv, He et al. 2019): a strided 1x1 conv would discard 3/4 of the activations.
- 3x3 stem, final BN-ReLU, global average pooling, linear classifier. Kaiming init. Input normalization lives inside the model.

Widths, blocks per stage and block type are configurable (`--widths`, `--blocks`, `--no-se`, `--block bottleneck`); each stage after the first halves the resolution (32 -> 16 -> 8). `train.py` refuses any configuration over 5M parameters.

| config | widths | blocks | params (with SE) | GMACs |
|--------|--------|--------|------------------|-------|
| A (default) | 64-128-256 | 3-4-3 | 4.66M | 0.72 |
| B | 64-128-256 | 3-5-3 | 4.96M | 0.80 |
| C | 80-160-320 | 2-3-2 | 4.84M | 0.78 |
| D | 96-192-320 | 2-2-2 | 4.94M | 0.86 |

## Training recipe

- SGD (momentum 0.9, Nesterov), lr 0.2, batch 256, 5-epoch linear warmup then cosine decay, 200 epochs
- Weight decay 5e-4 (not on BN/bias), label smoothing 0.1
- Augmentation: random crop (reflect pad 4), horizontal flip, random erasing (`--erase-p`) or a 16x16 Cutout square (`--cutout`); optional TrivialAugment (`--autoaugment`) and per-batch CutMix/MixUp (`--mix 0.5`)
- Knowledge distillation (`--teacher ckpt [ckpt ...]`): the student matches the averaged logits of the teachers on the *same augmented batch* (KL at temperature `--kd-temp`, weighted by `--kd-alpha` against the label loss); `--teacher-flip` also averages the teachers over the horizontal flip
- Optional regularizers: stochastic depth (`--drop-path`), weight EMA (`--ema`, evaluates and saves the EMA model) and SAM (`--sam`, about twice the compute, not supported with float16 loss scaling)
- bf16 mixed precision (float16 with loss scaling on GPUs without bf16) and channels-last on CUDA; checkpoint written atomically every epoch

The official 10k CIFAR-10 test split is used only to report accuracy, never for training or model selection inside a run.

## Experiments

Rounds 1-3: 200 epochs, seed 0, one NVIDIA B300 (five runs in parallel, ~8 s/epoch each), with full per-epoch logs in [`logs/`](logs). Rounds 4-6 ran on a Colab A100 from the self-contained notebooks in [`notebooks/`](notebooks); their per-epoch logs are not in the repo yet. *Final* is the test accuracy after the last epoch, *best* the highest epoch, *TTA* the final model averaged with its horizontal flip (`evaluate.py`).

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
- Kaggle public score of A (SE): **0.873**, about 8 pts below its CIFAR-10 test accuracy. Every team sees a similar drop, so the held-out set is a harder, shifted distribution.

### Round 2: augmentation

Same budget, stronger regularization: TrivialAugment (per-image), CutMix/MixUp (per-batch, half the batches), and whether random erasing is still useful on top.

```bash
C="--widths 80 160 320 --blocks 2 3 2"
uv run python train.py $C --autoaugment --out checkpoints/C_ta.pt
uv run python train.py $C --mix 0.5 --out checkpoints/C_mix.pt
uv run python train.py $C --autoaugment --mix 0.5 --out checkpoints/C_ta_mix.pt
uv run python train.py $C --autoaugment --mix 0.5 --erase-p 0 --out checkpoints/C_ta_mix_noerase.pt
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
R1="checkpoints/A_64-128-256_343_se.pt checkpoints/A_64-128-256_343_nose.pt checkpoints/B_64-128-256_353_se.pt checkpoints/C_80-160-320_232_se.pt checkpoints/D_96-192-320_222_se.pt"
R2="checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt"
C="--widths 80 160 320 --blocks 2 3 2"
uv run python train.py $C --teacher $R1 --out checkpoints/C_kd5.pt
uv run python train.py $C --autoaugment --mix 0.5 --erase-p 0 --teacher $R2 --out checkpoints/KD_C_aug.pt
uv run python train.py --no-se --autoaugment --mix 0.5 --erase-p 0 --teacher $R2 --out checkpoints/KD_A_nose_aug.pt
uv run python train.py $C --autoaugment --mix 0.5 --erase-p 0 --teacher $R2 --kd-temp 1 --kd-alpha 1.0 --out checkpoints/KD_C_aug_T1.pt
```

| run | augmentation | teacher | KD loss | params | final | best | TTA | same run without KD |
|-----|--------------|---------|---------|--------|-------|------|-----|---------------------|
| C | baseline | round 1 (97.06%) | T=4, alpha=0.9 | 4,836,010 | 96.13 | 96.13 | 96.33 | 96.18 |
| C | both, no erasing | round 2 (97.50%) | T=4, alpha=0.9 | 4,836,010 | 97.22 | 97.28 | **97.31** | 96.93 |
| A (no SE) | both, no erasing | round 2 (97.50%) | T=4, alpha=0.9 | 4,622,410 | **97.27** | **97.31** | 97.23 | 96.77 |
| C | both, no erasing | round 2 (97.50%) | T=1, alpha=1.0 | 4,836,010 | 97.12 | 97.15 | 97.25 | 96.93 |

![Round 3 training curves](assets/3-distillation.png)

- With baseline augmentation, distillation gives nothing (96.13 vs. 96.18): its early lead disappears once the learning rate decays.
- With strong augmentation and the stronger teacher it adds +0.2 to +0.5 pts, pushing a single sub-5M model past 97.2%. This matches "a good teacher is patient and consistent" (Beyer et al. 2022): distillation pays off when the teacher labels heavily augmented images.
- Softened (T=4) vs. raw (T=1) teacher targets are within noise.
- Flip TTA no longer helps the distilled students much: the teacher already averaged over many augmented views.
- Kaggle public score of the distilled C: **0.889** (+0.5 over round 2).

### Round 4: architecture under strong augmentation

Round 1 compared architectures with weak augmentation. This round repeats the search with TrivialAugment and CutMix/MixUp, including higher-resolution stages and bottleneck blocks. Five candidates were screened for 60 epochs ([`notebooks/round4_screening.ipynb`](notebooks/round4_screening.ipynb)), and the top two were confirmed for 200 ([`notebooks/round4_confirm.ipynb`](notebooks/round4_confirm.ipynb)).

```bash
A="--autoaugment --mix 0.5 --erase-p 0"
uv run python train.py --widths 80 160 320 --blocks 2 3 2 $A --epochs 60 --out checkpoints/S0_C.pt
uv run python train.py --widths 96 168 232 --blocks 3 3 3 $A --epochs 60 --out checkpoints/S1_mid.pt
uv run python train.py --widths 128 192 232 --blocks 4 3 2 --no-se $A --out checkpoints/S2_highres.pt
uv run python train.py --block bottleneck --widths 256 512 768 --blocks 5 5 3 $A --epochs 60 --out checkpoints/S3_bneck.pt
uv run python train.py --block bottleneck --widths 256 448 640 --blocks 7 7 4 $A --out checkpoints/S4_deep.pt
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
A="--widths 128 192 232 --blocks 4 3 2 --no-se --autoaugment --mix 0.5 --erase-p 0 --seed 1 --teacher-flip"
PAIR="checkpoints/S2_highres.pt checkpoints/KD_C_aug.pt"
FULL="$PAIR checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt"
uv run python train.py $A --teacher $PAIR --out checkpoints/KD5_pair.pt
uv run python train.py $A --teacher $FULL --out checkpoints/KD5_full.pt
uv run python train.py $A --teacher $FULL --cutout --out checkpoints/KD5_full_cutout.pt
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
A="--widths 128 192 232 --blocks 4 3 2 --no-se --autoaugment --mix 0.5 --erase-p 0 --cutout --seed 1 --teacher-flip"
TEACHERS="$FULL checkpoints/S4_deep.pt checkpoints/KD5_full.pt checkpoints/KD5_full_cutout.pt"
uv run python train.py $A --teacher $TEACHERS --out checkpoints/R6_base.pt
uv run python train.py $A --teacher $TEACHERS --drop-path 0.1 --ema 0.9995 --out checkpoints/R6_sd_ema.pt
uv run python train.py $A --teacher $TEACHERS --sam 0.05 --out checkpoints/R6_sam.pt
uv run python train.py $A --teacher $TEACHERS --drop-path 0.1 --ema 0.9995 --sam 0.05 --out checkpoints/R6_all.pt
```

| run | added | params | final | best | TTA | Kaggle |
|-----|-------|--------|-------|------|-----|--------|
| R6_base | none | 4,995,226 | **97.25** | **97.27** | **97.26** | **0.898** |
| R6_sam | SAM | 4,995,226 | 97.07 | 97.17 | 97.04 | 0.891 |
| R6_sd_ema | stochastic depth, EMA | 4,995,226 | 97.03 | 97.03 | 97.03 | 0.887 |
| R6_all | all three | 4,995,226 | 96.83 | 96.86 | 96.90 | 0.883 |

- Every added regularizer lowered both CIFAR-10 accuracy and the Kaggle score, and stacking them was worst. On top of distillation and heavy augmentation, they appear to over-regularize at 200 epochs.
- R6_base scores +0.002 over KD5_full_cutout on Kaggle at identical final accuracy (97.25), which is within leaderboard noise.
- Flip plus 1-pixel shift TTA left KD5_full_cutout at 0.896.

[`notebooks/round7_long.ipynb`](notebooks/round7_long.ipynb) is the next run: the R6_base recipe at 400 epochs.

## Kaggle notes

- The test set is 2,000 32x32 PNGs plus `test.csv` (`id,filename`); submissions are `id,label` with CIFAR-10 class indices.
- Bulk download is not available for this competition, and per-file API downloads are rate-limited (HTTP 429). Predictions can instead be made inside a private Kaggle notebook with the competition data attached and the checkpoint uploaded as a private dataset, running `predict.py --data-dir <competition input dir>`.

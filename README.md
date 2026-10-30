# cnn-maxxxing

A parameter-efficient ResNet for CIFAR-10, trained from scratch under a **5M trainable-parameter budget**.

Built for the [CIFAR-10 ResNet Architecture Challenge](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl) (NYU Tandon Deep Learning, Fall 2026): highest accuracy on a held-out image set, at most 5M trainable parameters, trained only on the 50k CIFAR-10 training images, no pretrained weights or external data.

## Usage

```bash
uv sync
uv run python train.py --out checkpoints/model.pt   # downloads CIFAR-10 to data/
uv run python evaluate.py checkpoints/a.pt checkpoints/b.pt   # flip-TTA accuracy of each and of their ensemble
uv run python train.py --teacher checkpoints/a.pt checkpoints/b.pt --out checkpoints/student.pt
uv run python predict.py --checkpoint checkpoints/student.pt
```

`predict.py` reads `data/kaggle/test.csv` and `data/kaggle/test_images/`, averages each image with its horizontal flip, and writes `submission.csv` (`id,label`). Checkpoints store the model config, so any run can be loaded without extra flags. `train.py --resume` continues from the `--out` checkpoint after an interruption.

## Model

A pre-activation ResNet with squeeze-and-excitation (SE) and ResNet-D shortcuts:

- **Pre-activation blocks** (BN-ReLU-conv x2, He et al. 2016): the skip path stays a pure identity, so gradients reach early layers unchanged.
- **SE channel attention** in every block (Hu et al. 2018): global context reweights channels for <1% extra parameters.
- **ResNet-D downsampling shortcut** (2x2 avg-pool then 1x1 conv, He et al. 2019): a strided 1x1 conv would discard 3/4 of the activations.
- 3x3 stem, final BN-ReLU, global average pooling, linear classifier. Kaiming init. Input normalization lives inside the model.

Widths and blocks per stage are configurable (`--widths`, `--blocks`, `--no-se`); each stage after the first halves the resolution (32 -> 16 -> 8). `train.py` refuses any configuration over 5M parameters.

| config | widths | blocks | params (with SE) | GMACs |
|--------|--------|--------|------------------|-------|
| A (default) | 64-128-256 | 3-4-3 | 4.66M | 0.72 |
| B | 64-128-256 | 3-5-3 | 4.96M | 0.80 |
| C | 80-160-320 | 2-3-2 | 4.84M | 0.78 |
| D | 96-192-320 | 2-2-2 | 4.94M | 0.86 |

## Training recipe

- SGD (momentum 0.9, Nesterov), lr 0.2, batch 256, 5-epoch linear warmup then cosine decay, 200 epochs
- Weight decay 5e-4 (not on BN/bias), label smoothing 0.1
- Augmentation: random crop (reflect pad 4), horizontal flip, random erasing (`--erase-p`); optional TrivialAugment (`--autoaugment`) and per-batch CutMix/MixUp (`--mix 0.5`)
- Knowledge distillation (`--teacher ckpt [ckpt ...]`): the student matches the averaged logits of the teachers on the *same augmented batch* (KL at temperature `--kd-temp`, weighted by `--kd-alpha` against the label loss)
- bf16 mixed precision and channels-last on CUDA; checkpoint written atomically every epoch

The official 10k CIFAR-10 test split is used only to report accuracy, never for training or model selection inside a run.

## Experiments

All runs: 200 epochs, seed 0, one NVIDIA B300 (five runs in parallel, ~8 s/epoch each). Full per-epoch logs are in [`logs/`](logs). *Final* is the test accuracy after the last epoch, *best* the highest epoch, *TTA* the final model averaged with its horizontal flip (`evaluate.py`).

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

## Kaggle notes

- The test set is 2,000 32x32 PNGs plus `test.csv` (`id,filename`); submissions are `id,label` with CIFAR-10 class indices.
- Bulk download is not available for this competition, and per-file API downloads are rate-limited (HTTP 429). Predictions can instead be made inside a private Kaggle notebook with the competition data attached and the checkpoint uploaded as a private dataset, running `predict.py --data-dir <competition input dir>`.

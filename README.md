# cnn-maxxxing

A parameter-efficient ResNet for CIFAR-10, trained from scratch under a **5M trainable-parameter budget**.

Built for the [CIFAR-10 ResNet Architecture Challenge](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl) (NYU Tandon Deep Learning, Fall 2026): highest accuracy on a held-out image set, at most 5M trainable parameters, trained only on the 50k CIFAR-10 training images, no pretrained weights or external data.

## Usage

```bash
uv sync
uv run python train.py --out checkpoints/model.pt   # downloads CIFAR-10 to data/
uv run python predict.py --checkpoint checkpoints/model.pt
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
- Augmentation: random crop (reflect pad 4), horizontal flip, random erasing (`--erase-p`); optional TrivialAugment (`--autoaugment`)
- bf16 mixed precision and channels-last on CUDA; checkpoint written atomically every epoch

The official 10k CIFAR-10 test split is used only to report accuracy, never for training or model selection inside a run.

## Kaggle notes

- The test set is 2,000 32x32 PNGs plus `test.csv` (`id,filename`); submissions are `id,label` with CIFAR-10 class indices.
- Bulk download is not available for this competition, and per-file API downloads are rate-limited (HTTP 429). Predictions can instead be made inside a private Kaggle notebook with the competition data attached and the checkpoint uploaded as a private dataset, running `predict.py --data-dir <competition input dir>`.

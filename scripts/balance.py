"""Balanced decoding: assign predictions so that every class receives the same number of images.

The competition's hidden set has a known design of 200 images per class. A model's raw argmax
predictions drift from that (R9 predicts 174 airplanes and 221 automobiles). Balanced decoding keeps
the model unchanged and only changes the decision rule: among all labelings with exactly N/10 images
per class, choose the one with the highest total log-probability (an optimal assignment). No labels
and no test-time training are involved; this is prior correction under label shift (Saerens et al. 2002).

Check on labeled CIFAR-10 test, where the class design is also balanced:
    uv run --with scipy python scripts/balance.py --check weights/R9_combined.pt
Write a balanced Kaggle submission (also saves the raw probabilities next to it):
    uv run --with scipy python scripts/balance.py --checkpoint weights/R9_combined.pt --data-dir data/kaggle --out submission_R9_balanced.csv
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ResNet, tta_logits  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def log_probs(checkpoint, x, batch=250):
    """Flip-TTA log-probabilities [N, 10] of a checkpoint on images x in [0, 1], NCHW."""
    dev = device()
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = ResNet(**ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    out = [torch.log_softmax(tta_logits(model, b.to(dev)).float(), 1).cpu() for b in x.split(batch)]
    del model
    return torch.cat(out).numpy()


def balance(logp, per_class=None):
    """Labels with exactly `per_class` images per class maximizing total log-probability."""
    n, k = logp.shape
    per_class = per_class or n // k
    if per_class * k != n:
        raise ValueError(f"{n} images cannot be split into {k} classes of {per_class}")
    cost = -np.repeat(logp, per_class, axis=1)  # column j belongs to class j // per_class
    rows, cols = linear_sum_assignment(cost)
    labels = np.empty(n, dtype=np.int64)
    labels[rows] = cols // per_class
    return labels


def counts(labels, k=10):
    return np.bincount(labels, minlength=k)


def check(checkpoint, data_dir, draws, seed):
    """Raw vs balanced accuracy on balanced 2,000-image subsets of the labeled CIFAR-10 test set."""
    from torchvision import datasets, transforms as T

    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=T.ToTensor())
    x = torch.stack([img for img, _ in test])
    y = np.array(test.targets)
    logp = log_probs(checkpoint, x)
    raw = logp.argmax(1)
    print(f"{checkpoint}: full test raw accuracy {100 * (raw == y).mean():.2f}%  predicted counts {counts(raw).tolist()}")

    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == c) for c in range(10)]
    deltas, changed = [], []
    # Five disjoint subsets covering the whole test set, then random balanced draws.
    perm = [rng.permutation(idx) for idx in by_class]
    subsets = [np.concatenate([p[200 * i:200 * (i + 1)] for p in perm]) for i in range(5)]
    subsets += [np.concatenate([rng.choice(idx, 200, replace=False) for idx in by_class]) for _ in range(draws)]
    for i, idx in enumerate(subsets):
        bal = balance(logp[idx])
        acc_raw, acc_bal = (raw[idx] == y[idx]).mean(), (bal == y[idx]).mean()
        deltas.append(100 * (acc_bal - acc_raw))
        changed.append(int((bal != raw[idx]).sum()))
        kind = "disjoint" if i < 5 else "random"
        print(f"  subset {i:2d} ({kind:8}): raw {100 * acc_raw:.2f}%  balanced {100 * acc_bal:.2f}%  "
              f"delta {deltas[-1]:+.2f} pts  changed {changed[-1]:3d} of 2000  raw counts {counts(raw[idx]).tolist()}")
    d = np.array(deltas)
    print(f"balanced minus raw over {len(d)} subsets: mean {d.mean():+.2f} pts, std {d.std(ddof=1):.2f}, "
          f"min {d.min():+.2f}, max {d.max():+.2f}; predictions changed per subset: mean {np.mean(changed):.1f}")
    print("verdict:", "balanced decoding does not hurt on CIFAR-10 test" if d.mean() >= -0.05 else "balanced decoding HURTS on CIFAR-10 test; do not submit it")


def submit(checkpoint, data_dir, out, per_class):
    data = Path(data_dir)
    rows = list(csv.DictReader(open(data / "test.csv")))
    imgs = np.stack([np.asarray(Image.open(data / "test_images" / r["filename"]).convert("RGB")) for r in rows])
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255
    logp = log_probs(checkpoint, x)
    raw = logp.argmax(1)
    bal = balance(logp, per_class)
    moved = np.flatnonzero(bal != raw)
    margin = np.sort(logp[moved], 1)
    print(f"raw counts      {dict(zip(CLASSES, counts(raw).tolist()))}")
    print(f"balanced counts {dict(zip(CLASSES, counts(bal).tolist()))}")
    print(f"changed {len(moved)} of {len(raw)} predictions; their raw top-1 probability: "
          f"median {np.exp(margin[:, -1]).mean():.2f}, max {np.exp(margin[:, -1]).max():.2f}" if len(moved) else "no changes")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows((r["id"], int(label)) for r, label in zip(rows, bal))
    np.savez(Path(out).with_suffix(".npz"), ids=np.array([r["id"] for r in rows]), log_probs=logp, raw=raw, balanced=bal)
    print(f"wrote {out} and {Path(out).with_suffix('.npz')}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", help="checkpoint to predict the Kaggle test set with")
    p.add_argument("--check", metavar="CHECKPOINT", help="evaluate balanced decoding on labeled CIFAR-10 test instead")
    p.add_argument("--data-dir", default=None, help="data/kaggle for --checkpoint, data (CIFAR-10) for --check")
    p.add_argument("--out", default="submission_balanced.csv")
    p.add_argument("--per-class", type=int, default=None, help="images per class (default: N / 10)")
    p.add_argument("--draws", type=int, default=20, help="random balanced subsets for --check")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if bool(args.checkpoint) == bool(args.check):
        p.error("give exactly one of --checkpoint or --check")
    if args.check:
        check(args.check, args.data_dir or "data", args.draws, args.seed)
    else:
        submit(args.checkpoint, args.data_dir or "data/kaggle", args.out, args.per_class)


if __name__ == "__main__":
    main()

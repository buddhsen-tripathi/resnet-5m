import argparse

import torch
from torchvision import datasets, transforms as T

from model import ResNet


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="CIFAR-10 test accuracy (with flip TTA) of checkpoints and of their logit-averaged ensemble")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=T.ToTensor())
    x = torch.stack([img for img, _ in test]).to(device)
    y = torch.tensor(test.targets, device=device)

    logits = []
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=device)
        model = ResNet(**ckpt["cfg"]).to(device).eval()
        model.load_state_dict(ckpt["model"])
        out = torch.cat([(model(b) + model(b.flip(3))) / 2 for b in x.split(500)])
        logits.append(out)
        print(f"{path}: {(out.argmax(1) == y).float().mean():.4f}")
    if len(logits) > 1:
        print(f"ensemble of {len(logits)}: {(torch.stack(logits).mean(0).argmax(1) == y).float().mean():.4f}")


if __name__ == "__main__":
    main()

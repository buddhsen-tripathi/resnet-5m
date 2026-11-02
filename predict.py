import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import ResNet, tta_logits


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Write a Kaggle submission CSV (id,label) for the held-out test images")
    p.add_argument("--data-dir", default="data/kaggle", help="folder with test.csv and test_images/")
    p.add_argument("--checkpoint", default="checkpoints/model.pt")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--shift", type=int, default=0, help="also average over translations of this many pixels")
    p.add_argument("--zoom", type=float, nargs="*", default=[], help="also average over these scale factors")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = ResNet(**ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    data = Path(args.data_dir)
    rows = list(csv.DictReader(open(data / "test.csv")))
    imgs = np.stack([np.asarray(Image.open(data / "test_images" / r["filename"]).convert("RGB")) for r in rows])
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255

    preds = []
    for batch in x.split(1024):
        batch = batch.to(device)
        preds += tta_logits(model, batch, args.shift, args.zoom).argmax(1).tolist()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows((r["id"], label) for r, label in zip(rows, preds))
    print(f"wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()

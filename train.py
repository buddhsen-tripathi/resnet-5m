import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from torchvision.transforms import v2

from model import MAX_PARAMS, ResNet, count_params


def get_args():
    p = argparse.ArgumentParser(description="Train a <5M-param ResNet on CIFAR-10")
    p.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--blocks", type=int, nargs="+", default=[3, 4, 3])
    p.add_argument("--se", action=argparse.BooleanOptionalAction, default=True, help="squeeze-and-excitation")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--autoaugment", action="store_true", help="add TrivialAugmentWide")
    p.add_argument("--mix", type=float, default=0.0, help="probability of applying CutMix or MixUp to a batch")
    p.add_argument("--erase-p", type=float, default=0.5, help="RandomErasing (cutout) probability")
    p.add_argument("--teacher", nargs="+", default=[], help="checkpoint(s) to distill from (logits are averaged)")
    p.add_argument("--kd-alpha", type=float, default=0.9, help="weight of the distillation loss vs. the label loss")
    p.add_argument("--kd-temp", type=float, default=4.0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--resume", action="store_true", help="continue from --out if it exists (e.g. after preemption)")
    return p.parse_args()


def loaders(args):
    aug = [T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()]
    if args.autoaugment:
        aug.append(T.TrivialAugmentWide())
    aug += [T.ToTensor(), T.RandomErasing(p=args.erase_p, scale=(0.02, 0.2), value=0.5)]
    train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=T.Compose(aug))
    # The official CIFAR-10 test split is used only to monitor generalization, never for training.
    test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=T.ToTensor())
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    return (
        DataLoader(train, args.batch_size, shuffle=True, drop_last=True, **kw),
        DataLoader(test, 1024, **kw),
    )


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, torch.bfloat16, enabled=amp):
            correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    cfg = dict(widths=args.widths, blocks=args.blocks, se=args.se)
    model = ResNet(**cfg).to(device, memory_format=torch.channels_last)
    n_params = count_params(model)
    print(f"params: {n_params:,}")
    assert n_params <= MAX_PARAMS, f"model has {n_params:,} params, limit is {MAX_PARAMS:,}"
    net = torch.compile(model) if args.compile else model

    train_loader, test_loader = loaders(args)
    # No weight decay on BN and bias parameters.
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0}],
        lr=args.lr, momentum=0.9, nesterov=True,
    )
    steps, warmup = args.epochs * len(train_loader), args.warmup_epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup)))
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    mix = v2.RandomChoice([v2.CutMix(num_classes=10), v2.MixUp(num_classes=10)])
    teachers = []
    for path in args.teacher:
        t = torch.load(path, map_location=device)
        teachers.append(ResNet(**t["cfg"]).to(device, memory_format=torch.channels_last).eval())
        teachers[-1].load_state_dict(t["model"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume and out.exists():
        ckpt = torch.load(out, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start = ckpt["epoch"]
        print(f"resumed from {out} at epoch {start}")
    for epoch in range(start, args.epochs):
        net.train()
        t0, loss_sum = time.time(), torch.zeros((), device=device)
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if torch.rand(()) < args.mix:
                x, y = mix(x, y)  # y becomes soft labels
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, torch.bfloat16, enabled=amp):
                logits = net(x)
                loss = criterion(logits, y)
                if teachers:
                    # The teacher labels the exact same augmented batch the student sees.
                    with torch.no_grad():
                        target = torch.stack([t(x) for t in teachers]).float().mean(0)
                    T = args.kd_temp
                    kd = F.kl_div(F.log_softmax(logits.float() / T, 1), F.softmax(target / T, 1), reduction="batchmean")
                    loss = args.kd_alpha * kd * T * T + (1 - args.kd_alpha) * loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach()
        acc = evaluate(net, test_loader, device, amp)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {loss_sum.item() / len(train_loader):.4f}  "
              f"test_acc {acc:.4f}  lr {sched.get_last_lr()[0]:.4f}  {time.time() - t0:.1f}s", flush=True)
        state = {"model": model.state_dict(), "cfg": cfg, "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "epoch": epoch + 1, "test_acc": acc}
        torch.save(state, out.with_suffix(".tmp"))
        os.replace(out.with_suffix(".tmp"), out)  # atomic, so preemption mid-save can't corrupt the checkpoint
    print(f"saved {out}")


if __name__ == "__main__":
    main()

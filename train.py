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

from model import BLOCKS, MAX_PARAMS, ResNet, count_params


def get_args():
    p = argparse.ArgumentParser(description="Train a <5M-param ResNet on CIFAR-10")
    p.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--blocks", type=int, nargs="+", default=[3, 4, 3])
    p.add_argument("--block", choices=BLOCKS, default="basic", help="residual block type")
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
    p.add_argument("--cutout", action="store_true", help="erase a 16x16 square from every image (overrides --erase-p)")
    p.add_argument("--zoom", type=float, nargs=2, metavar=("MIN", "MAX"), help="random scale range, at least 0.5")
    p.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth rate of the last block")
    p.add_argument("--ema", type=float, default=0.0, help="weight EMA decay per step; evaluates and saves the EMA model")
    p.add_argument("--sam", type=float, default=0.0, help="SAM neighborhood size rho (about 2x the compute)")
    p.add_argument("--teacher", nargs="+", default=[], help="checkpoint(s) to distill from (logits are averaged)")
    p.add_argument("--teacher-flip", action="store_true", help="also average teacher logits over the horizontal flip")
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
    if args.zoom:
        # Reflect padding first, so zooming out never shows empty borders.
        scale = T.RandomAffine(0, scale=args.zoom, interpolation=T.InterpolationMode.BILINEAR)
        aug = [T.Pad(16, padding_mode="reflect"), scale, T.CenterCrop(32)] + aug
    if args.autoaugment:
        aug.append(T.TrivialAugmentWide())
    erase = dict(p=1.0, scale=(0.25, 0.25), ratio=(1, 1)) if args.cutout else dict(p=args.erase_p, scale=(0.02, 0.2))
    aug += [T.ToTensor(), T.RandomErasing(value=0.5, **erase)]
    train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=T.Compose(aug))
    # The official CIFAR-10 test split is used only to monitor generalization, never for training.
    test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=T.ToTensor())
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    return (
        DataLoader(train, args.batch_size, shuffle=True, drop_last=True, **kw),
        DataLoader(test, 1024, **kw),
    )


@torch.no_grad()
def evaluate(model, loader, device, amp, amp_dtype):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, amp_dtype, enabled=amp):
            correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    amp = device.type == "cuda"
    # Older GPUs such as the V100 have no bf16; float16 needs loss scaling to avoid gradient underflow.
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    torch.backends.cudnn.benchmark = True

    assert not (args.sam and scaler.is_enabled()), "--sam is not supported with float16 loss scaling"

    cfg = dict(widths=args.widths, blocks=args.blocks, se=args.se, block=args.block, drop_path=args.drop_path)
    model = ResNet(**cfg).to(device, memory_format=torch.channels_last)
    n_params = count_params(model)
    print(f"params: {n_params:,}")
    assert n_params <= MAX_PARAMS, f"model has {n_params:,} params, limit is {MAX_PARAMS:,}"
    net = torch.compile(model) if args.compile else model
    ema = None
    if args.ema:
        ema = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema), use_buffers=True
        )
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

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
        model.load_state_dict(ckpt.get("raw", ckpt["model"]))
        if ema:
            ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
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
            target = None
            if teachers:
                # The teacher labels the exact same augmented batch the student sees.
                with torch.no_grad(), torch.autocast(device.type, amp_dtype, enabled=amp):
                    target = torch.stack([t(x) for t in teachers]).float().mean(0)
                    if args.teacher_flip:
                        target = (target + torch.stack([t(x.flip(3)) for t in teachers]).float().mean(0)) / 2

            def compute_loss():
                with torch.autocast(device.type, amp_dtype, enabled=amp):
                    logits = net(x)
                    loss = criterion(logits, y)
                    if target is not None:
                        T = args.kd_temp
                        kd = F.kl_div(F.log_softmax(logits.float() / T, 1), F.softmax(target / T, 1), reduction="batchmean")
                        loss = args.kd_alpha * kd * T * T + (1 - args.kd_alpha) * loss
                return loss

            loss = compute_loss()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.sam:
                # SAM (Foret et al., 2021): take the gradient at the worst nearby weights, then step from the original ones.
                params = [p for p in model.parameters() if p.grad is not None]
                scale = args.sam / (torch.stack([p.grad.norm() for p in params]).norm() + 1e-12)
                perturb = [p.grad * scale for p in params]
                with torch.no_grad():
                    torch._foreach_add_(params, perturb)
                for m in bns:
                    m.momentum = 0  # the batch was already counted in the BN statistics
                opt.zero_grad(set_to_none=True)
                compute_loss().backward()
                for m in bns:
                    m.momentum = 0.1
                with torch.no_grad():
                    torch._foreach_sub_(params, perturb)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update_parameters(model)
            loss_sum += loss.detach()
        acc = evaluate(ema if ema else net, test_loader, device, amp, amp_dtype)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {loss_sum.item() / len(train_loader):.4f}  "
              f"test_acc {acc:.4f}  lr {sched.get_last_lr()[0]:.4f}  {time.time() - t0:.1f}s", flush=True)
        state = {"model": (ema.module if ema else model).state_dict(), "cfg": cfg, "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch + 1, "test_acc": acc}
        if ema:
            state |= {"raw": model.state_dict(), "ema": ema.state_dict()}
        torch.save(state, out.with_suffix(".tmp"))
        os.replace(out.with_suffix(".tmp"), out)  # atomic, so preemption mid-save can't corrupt the checkpoint
    print(f"saved {out}")


if __name__ == "__main__":
    main()

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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
    augmentation = p.add_mutually_exclusive_group()
    augmentation.add_argument("--autoaugment", action="store_true", help="add TrivialAugmentWide")
    augmentation.add_argument("--augmix", action="store_true", help="use AugMix transforms (without a separate JS loss)")
    p.add_argument("--mix", type=float, default=0.0, help="probability of applying CutMix or MixUp to a batch")
    p.add_argument("--erase-p", type=float, default=0.5, help="RandomErasing (cutout) probability")
    p.add_argument("--cutout", action="store_true", help="erase a fixed square from every image (overrides --erase-p)")
    p.add_argument("--cutout-size", type=int, default=16, help="Cutout square side in pixels (1-31)")
    p.add_argument("--supcon", type=float, default=0.0, help="weight of supervised contrastive loss on two unmixed views")
    p.add_argument("--supcon-temp", type=float, default=0.07)
    p.add_argument("--zoom", type=float, nargs=2, metavar=("MIN", "MAX"), help="random scale range, at least 0.5")
    p.add_argument("--resample", type=float, default=0.0,
                   help="probability of upscaling then downscaling each image with random filters")
    p.add_argument("--resample-size", type=int, nargs=2, default=[48, 64], metavar=("MIN", "MAX"),
                   help="intermediate side length range for --resample")
    p.add_argument("--jitter", type=float, default=0.0, help="brightness and contrast jitter strength")
    p.add_argument("--jitter-p", type=float, default=0.5, help="probability of applying --jitter")
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
    p.add_argument("--allow-over-budget", action="store_true",
                   help="train a teacher above the 5M student limit (never a submission candidate)")
    args = p.parse_args()
    if not 1 <= args.cutout_size <= 31:
        p.error("--cutout-size must be between 1 and 31")
    if args.supcon < 0 or args.supcon_temp <= 0:
        p.error("--supcon must be nonnegative and --supcon-temp must be positive")
    if not 0 <= args.resample <= 1 or not 32 <= args.resample_size[0] <= args.resample_size[1]:
        p.error("--resample must be in [0, 1] and --resample-size must satisfy 32 <= MIN <= MAX")
    if args.jitter < 0 or not 0 <= args.jitter_p <= 1:
        p.error("--jitter must be nonnegative and --jitter-p must be in [0, 1]")
    return args


def check_budget(n_params, allow_over_budget=False):
    """Students must fit the 5M budget; a teacher may exceed it only when asked explicitly."""
    if n_params > MAX_PARAMS and not allow_over_budget:
        raise AssertionError(f"model has {n_params:,} params, limit is {MAX_PARAMS:,} (teachers: --allow-over-budget)")
    if n_params > MAX_PARAMS:
        print(f"over-budget teacher: {n_params:,} params; this checkpoint must not be submitted as a student")


class TwoViews:
    """Independent augmentations of one image, retaining its original class label."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)


class RandomResample:
    """Upscale, then downscale back with independently chosen filters.

    Imitates images that reached 32x32 through a different resizing pipeline than CIFAR-10's.
    """

    FILTERS = (Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX, Image.HAMMING)

    def __init__(self, p, size_range=(48, 64)):
        self.p, self.size_range = p, size_range

    def _filter(self):
        return self.FILTERS[torch.randint(len(self.FILTERS), ()).item()]

    def __call__(self, image):
        if torch.rand(()).item() >= self.p:
            return image
        side = torch.randint(self.size_range[0], self.size_range[1] + 1, ()).item()
        return image.resize((side, side), self._filter()).resize(image.size, self._filter())


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """SupCon (Khosla et al., 2020) on [batch, views, features], excluding self-pairs."""
    if features.ndim != 3 or features.size(1) < 2 or labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("SupCon requires at least two views per image and one original label per image")
    if temperature <= 0:
        raise ValueError("SupCon temperature must be positive")
    with torch.autocast(features.device.type, enabled=False):
        z = F.normalize(features.float(), dim=-1).flatten(0, 1)
        classes = labels.repeat_interleave(features.size(1))
        self_pairs = torch.eye(len(z), dtype=torch.bool, device=z.device)
        positives = classes[:, None].eq(classes[None, :]) & ~self_pairs
        similarities = (z @ z.T) / temperature
        normalizer = torch.logsumexp(similarities.masked_fill(self_pairs, -torch.inf), dim=1, keepdim=True)
        log_prob = similarities - normalizer
        return -((log_prob * positives).sum(1) / positives.sum(1)).mean()


def loaders(args):
    aug = [T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()]
    if args.zoom:
        # Reflect padding first, so zooming out never shows empty borders.
        scale = T.RandomAffine(0, scale=args.zoom, interpolation=T.InterpolationMode.BILINEAR)
        aug = [T.Pad(16, padding_mode="reflect"), scale, T.CenterCrop(32)] + aug
    if args.jitter:
        aug.insert(0, T.RandomApply([T.ColorJitter(brightness=args.jitter, contrast=args.jitter)], p=args.jitter_p))
    if args.resample:
        aug.insert(0, RandomResample(args.resample, tuple(args.resample_size)))
    if args.autoaugment:
        aug.append(T.TrivialAugmentWide())
    if args.augmix:
        aug.append(T.AugMix())
    area = (args.cutout_size / 32) ** 2
    erase = dict(p=1.0, scale=(area, area), ratio=(1, 1)) if args.cutout else dict(p=args.erase_p, scale=(0.02, 0.2))
    aug += [T.ToTensor(), T.RandomErasing(value=0.5, **erase)]
    transform = T.Compose(aug)
    train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=TwoViews(transform) if args.supcon else transform)
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
    projector = nn.Sequential(nn.Linear(args.widths[-1], 128), nn.ReLU(), nn.Linear(128, 128)).to(device) if args.supcon else None
    n_params = count_params(model) + (count_params(projector) if projector is not None else 0)
    print(f"params: {n_params:,}")
    if projector is not None:
        print(f"classifier: {count_params(model):,}; training-only projection head: {count_params(projector):,}")
    check_budget(n_params, args.allow_over_budget)
    net = torch.compile(model) if args.compile else model
    ema = None
    if args.ema:
        ema = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema), use_buffers=True
        )
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

    train_loader, test_loader = loaders(args)
    # No weight decay on BN and bias parameters.
    parameters = list(model.parameters()) + (list(projector.parameters()) if projector is not None else [])
    decay = [p for p in parameters if p.ndim > 1]
    no_decay = [p for p in parameters if p.ndim <= 1]
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
        teachers.append(ResNet(**t["cfg"]).to(device, memory_format=torch.channels_last).eval().requires_grad_(False))
        teachers[-1].load_state_dict(t["model"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume and out.exists():
        ckpt = torch.load(out, map_location=device)
        saved_cfg = {"block": "basic", "drop_path": 0.0, **ckpt["cfg"]}
        saved_cfg["widths"], saved_cfg["blocks"] = list(saved_cfg["widths"]), list(saved_cfg["blocks"])
        assert saved_cfg == cfg, "resume requires the checkpoint's model configuration"
        assert ("projector" in ckpt) == (projector is not None), "resume requires the original --supcon setting"
        if "train_args" in ckpt:
            runtime_options = {"out", "resume", "workers", "compile", "data_dir"}
            for key, value in vars(args).items():
                if key not in runtime_options:
                    assert ckpt["train_args"][key] == value, f"resume requires the original --{key.replace('_', '-')}"
        model.load_state_dict(ckpt.get("raw", ckpt["model"]))
        if projector is not None:
            projector.load_state_dict(ckpt["projector"])
        if ema:
            ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "rng" in ckpt:
            torch.set_rng_state(ckpt["rng"].cpu())
        if device.type == "cuda" and "cuda_rng" in ckpt:
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng"]])
        start = ckpt["epoch"]
        print(f"resumed from {out} at epoch {start}")
    for epoch in range(start, args.epochs):
        net.train()
        t0, loss_sum = time.time(), torch.zeros((), device=device)
        classification_sum, contrastive_sum = torch.zeros((), device=device), torch.zeros((), device=device)
        for x, y in train_loader:
            y = y.to(device, non_blocking=True)
            views = [view.to(device, non_blocking=True) for view in x] if projector is not None else None
            original_labels = y.clone() if views is not None else None
            x = views[0].clone() if views is not None else x.to(device, non_blocking=True)
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
                    contrastive = torch.zeros((), device=device)
                    if views is None:
                        logits = net(x)
                    else:
                        all_logits, features = net(torch.cat([x, *views]), return_features=True)
                        logits = all_logits[:len(x)]
                        embeddings = projector(features[len(x):]).reshape(2, len(x), -1).transpose(0, 1)
                        contrastive = supervised_contrastive_loss(embeddings, original_labels, args.supcon_temp)
                    loss = criterion(logits, y)
                    if target is not None:
                        T = args.kd_temp
                        kd = F.kl_div(F.log_softmax(logits.float() / T, 1), F.softmax(target / T, 1), reduction="batchmean")
                        loss = args.kd_alpha * kd * T * T + (1 - args.kd_alpha) * loss
                    classification = loss
                    loss = classification + args.supcon * contrastive
                return loss, classification.detach(), contrastive.detach()

            loss, classification, contrastive = compute_loss()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.sam:
                # SAM (Foret et al., 2021): take the gradient at the worst nearby weights, then step from the original ones.
                params = [p for p in parameters if p.grad is not None]
                scale = args.sam / (torch.stack([p.grad.norm() for p in params]).norm() + 1e-12)
                perturb = [p.grad * scale for p in params]
                with torch.no_grad():
                    torch._foreach_add_(params, perturb)
                for m in bns:
                    m.momentum = 0  # the batch was already counted in the BN statistics
                opt.zero_grad(set_to_none=True)
                compute_loss()[0].backward()
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
            classification_sum += classification
            contrastive_sum += contrastive
        acc = evaluate(ema if ema else net, test_loader, device, amp, amp_dtype)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {loss_sum.item() / len(train_loader):.4f}  "
              f"test_acc {acc:.4f}  lr {sched.get_last_lr()[0]:.4f}  {time.time() - t0:.1f}s", flush=True)
        if projector is not None:
            print(f"  classification_kd {classification_sum.item() / len(train_loader):.4f}  "
                  f"supcon {contrastive_sum.item() / len(train_loader):.4f}", flush=True)
        state = {"model": (ema.module if ema else model).state_dict(), "cfg": cfg, "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch + 1, "test_acc": acc, "train_args": vars(args), "trainable_params": n_params,
                 "rng": torch.get_rng_state()}
        if projector is not None:
            state["projector"] = projector.state_dict()
        if device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        if ema:
            state |= {"raw": model.state_dict(), "ema": ema.state_dict()}
        torch.save(state, out.with_suffix(".tmp"))
        os.replace(out.with_suffix(".tmp"), out)  # atomic, so preemption mid-save can't corrupt the checkpoint
    print(f"saved {out}")


if __name__ == "__main__":
    main()

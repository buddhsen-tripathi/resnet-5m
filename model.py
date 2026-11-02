import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import stochastic_depth

MAX_PARAMS = 5_000_000
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class SqueezeExcite(nn.Module):
    """Channel attention (Hu et al., 2018): reweight channels using globally pooled context."""

    def __init__(self, c, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // reduction, 1)
        self.fc2 = nn.Conv2d(c // reduction, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))


class PreActBlock(nn.Module):
    """Pre-activation basic block (He et al., 2016): BN-ReLU-conv x2, keeping the skip path a clean identity.

    Downsampling shortcuts use avg-pool then 1x1 conv (ResNet-D, He et al., 2019) so no activations are discarded.
    """

    def __init__(self, cin, cout, stride=1, se=True, drop_path=0.0):
        super().__init__()
        self.drop_path = drop_path
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.se = SqueezeExcite(cout) if se else nn.Identity()
        self.shortcut = None
        if stride != 1 or cin != cout:
            pool = [nn.AvgPool2d(stride)] if stride != 1 else []
            self.shortcut = nn.Sequential(*pool, nn.Conv2d(cin, cout, 1, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        skip = x if self.shortcut is None else self.shortcut(out)
        out = self.conv2(F.relu(self.bn2(self.conv1(out))))
        return stochastic_depth(self.se(out), self.drop_path, "row", self.training) + skip


class PreActBottleneck(nn.Module):
    """Pre-activation bottleneck (He et al., 2016): 1x1 down, 3x3, 1x1 up, for more depth per parameter."""

    def __init__(self, cin, cout, stride=1, se=True, reduction=4, drop_path=0.0):
        super().__init__()
        self.drop_path = drop_path
        mid = cout // reduction
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, mid, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, cout, 1, bias=False)
        self.se = SqueezeExcite(cout) if se else nn.Identity()
        self.shortcut = None
        if stride != 1 or cin != cout:
            pool = [nn.AvgPool2d(stride)] if stride != 1 else []
            self.shortcut = nn.Sequential(*pool, nn.Conv2d(cin, cout, 1, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        skip = x if self.shortcut is None else self.shortcut(out)
        out = self.conv2(F.relu(self.bn2(self.conv1(out))))
        out = self.conv3(F.relu(self.bn3(out)))
        return stochastic_depth(self.se(out), self.drop_path, "row", self.training) + skip


BLOCKS = {"basic": PreActBlock, "bottleneck": PreActBottleneck}


class ResNet(nn.Module):
    """CIFAR ResNet: 3x3 stem, then one stage per width; each stage after the first halves resolution.

    Takes images in [0, 1]; normalization is done inside the model so inference needs no extra preprocessing.
    """

    def __init__(self, widths=(64, 128, 256), blocks=(3, 4, 3), se=True, num_classes=10, block="basic", drop_path=0.0):
        super().__init__()
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))
        self.stem = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        layers, cin, total = [], widths[0], sum(blocks)
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                # Stochastic depth grows linearly with depth (Huang et al., 2016).
                rate = drop_path * len(layers) / max(total - 1, 1)
                layers.append(BLOCKS[block](cin, w, stride=2 if i > 0 and j == 0 else 1, se=se, drop_path=rate))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.layers(self.stem((x - self.mean) / self.std))
        x = F.relu(self.bn(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def zoom_view(x, factor):
    """Rescale around the center, keeping the image size: crops when zooming in, reflect-pads when zooming out."""
    size = x.size(-1)
    new = round(size * factor)
    x = F.interpolate(x, size=new, mode="bilinear", align_corners=False, antialias=factor < 1)
    if new >= size:
        start = (new - size) // 2
        return x[..., start:start + size, start:start + size]
    before = (size - new) // 2
    after = size - new - before
    return F.pad(x, (before, after, before, after), "reflect")


def tta_logits(model, x, shift=0, zooms=()):
    """Average logits over the horizontal flip, translations of `shift` pixels and the `zooms` scale factors."""
    size = x.size(-1)
    padded = F.pad(x, (shift,) * 4, "reflect")
    offsets = [(0, 0)] + ([(shift, 0), (-shift, 0), (0, shift), (0, -shift)] if shift else [])
    views = [padded[..., shift + dy:shift + dy + size, shift + dx:shift + dx + size] for dx, dy in offsets]
    views += [zoom_view(x, z) for z in zooms]
    return sum(model(v) + model(v.flip(3)) for v in views) / (2 * len(views))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

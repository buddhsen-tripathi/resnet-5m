import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, cin, cout, stride=1, se=True):
        super().__init__()
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
        return self.se(out) + skip


class ResNet(nn.Module):
    """CIFAR ResNet: 3x3 stem, then one stage per width; each stage after the first halves resolution.

    Takes images in [0, 1]; normalization is done inside the model so inference needs no extra preprocessing.
    """

    def __init__(self, widths=(64, 128, 256), blocks=(3, 4, 3), se=True, num_classes=10):
        super().__init__()
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))
        self.stem = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        layers, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(PreActBlock(cin, w, stride=2 if i > 0 and j == 0 else 1, se=se))
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

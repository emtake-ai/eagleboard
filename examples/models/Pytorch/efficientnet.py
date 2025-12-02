# efficientnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Swish activation (same as SiLU in PyTorch)
# ---------------------------------------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# ---------------------------------------------------------
# Squeeze-and-Excite block
# ---------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        squeezed = in_channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeezed, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


# ---------------------------------------------------------
# MBConv block (EfficientNet’s inverted residual)
# ---------------------------------------------------------
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel, stride):
        super().__init__()
        mid = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        # 1×1 expansion
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, mid, 1, bias=False))
            layers.append(nn.BatchNorm2d(mid))
            layers.append(Swish())

        # Depthwise convolution
        layers.append(nn.Conv2d(mid, mid, kernel, stride,
                                padding=kernel // 2,
                                groups=mid, bias=False))
        layers.append(nn.BatchNorm2d(mid))
        layers.append(Swish())

        # SE block
        layers.append(SEBlock(mid))

        # 1×1 projection
        layers.append(nn.Conv2d(mid, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out


# ---------------------------------------------------------
# EfficientNet-B0
# ---------------------------------------------------------
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  # (32,112,112)
            nn.BatchNorm2d(32),
            Swish()
        )

        # Stage configuration for EfficientNet-B0
        # (expand_ratio, channels, repeats, kernel, stride)
        cfg = [
            (1,   16, 1, 3, 1),
            (6,   24, 2, 3, 2),
            (6,   40, 2, 5, 2),
            (6,   80, 3, 3, 2),
            (6,  112, 3, 5, 1),
            (6,  192, 4, 5, 2),
            (6,  320, 1, 3, 1),
        ]

        layers = []
        in_channels = 32

        for expand_ratio, out_channels, repeats, kernel, stride in cfg:
            for i in range(repeats):
                s = stride if i == 0 else 1
                layers.append(MBConv(in_channels, out_channels,
                                     expand_ratio, kernel, s))
                in_channels = out_channels

        self.blocks = nn.Sequential(*layers)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            Swish()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)


    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------
if __name__ == "__main__":
    model = EfficientNetB0(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output:", y.shape)  # torch.Size([1,1000])

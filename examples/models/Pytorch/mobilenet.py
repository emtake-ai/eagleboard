# mobilenet_v1.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Depthwise Separable Convolution Block (DWConv → PWConv)
# ---------------------------------------------------------
class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DWConvBlock, self).__init__()

        # Depthwise 3×3 convolution
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )

        # Pointwise 1×1 convolution
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


# ---------------------------------------------------------
# MobileNet v1 Model
# ---------------------------------------------------------
class MobileNetV1(nn.Module):
    """
    Original MobileNet v1 (2017)
    Paper: https://arxiv.org/abs/1704.04861
    """
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        # Input conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # Depthwise + pointwise blocks following the original table
        self.layers = nn.Sequential(
            DWConvBlock(32, 64, stride=1),

            DWConvBlock(64, 128, stride=2),
            DWConvBlock(128, 128, stride=1),

            DWConvBlock(128, 256, stride=2),
            DWConvBlock(256, 256, stride=1),

            DWConvBlock(256, 512, stride=2),
            DWConvBlock(512, 512, stride=1),
            DWConvBlock(512, 512, stride=1),
            DWConvBlock(512, 512, stride=1),
            DWConvBlock(512, 512, stride=1),
            DWConvBlock(512, 512, stride=1),

            DWConvBlock(512, 1024, stride=2),
            DWConvBlock(1024, 1024, stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------
if __name__ == "__main__":
    model = MobileNetV1(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output:", y.shape)  # torch.Size([1, 1000])

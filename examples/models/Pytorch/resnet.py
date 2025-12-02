# resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# BasicBlock (for ResNet-18, ResNet-34)
# ---------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---------------------------------------------------------
# Bottleneck Block (for ResNet-50, ResNet-101, ResNet-152)
# ---------------------------------------------------------
class Bottleneck(nn.Module):
    expansion = 4  # output = planes * 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # 1x1, reduce
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3x3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1x1, expand
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---------------------------------------------------------
# Main ResNet
# ---------------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        """
        block : BasicBlock or Bottleneck
        layers: list like [2,2,2,2] or [3,4,6,3]
        """
        super(ResNet, self).__init__()

        self.in_planes = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Stages (Res2_x, Res3_x, Res4_x, Res5_x)
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    # Create one stage (e.g., Res3_x)
    def _make_layer(self, block, planes, blocks, stride):
        downsample = None

        # If changing size or using bottleneck expansion -> project shortcut
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block of the stage
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)   # 56×56
        x = self.layer2(x)   # 28×28
        x = self.layer3(x)   # 14×14
        x = self.layer4(x)   # 7×7

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ---------------------------------------------------------
# Factory functions
# ---------------------------------------------------------
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------
if __name__ == "__main__":
    model = resnet50()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output:", y.shape)   # (1,1000)

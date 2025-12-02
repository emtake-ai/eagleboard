# googlenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Basic Convolution Block (Conv → BN → ReLU)
# ---------------------------------------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# ---------------------------------------------------------
# Inception Module (1x1, 3x3, 5x5, pool-project)
# ---------------------------------------------------------
class Inception(nn.Module):
    def __init__(self, in_channels,
                 ch1x1, 
                 ch3x3_reduce, ch3x3, 
                 ch5x5_reduce, ch5x5, 
                 pool_proj):
        super(Inception, self).__init__()

        # 1×1 branch
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # 1×1 → 3×3 branch
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_reduce, kernel_size=1),
            BasicConv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )

        # 1×1 → 5×5 branch
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5_reduce, kernel_size=1),
            BasicConv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2)
        )

        # maxpool → 1×1 branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], 1)


# ---------------------------------------------------------
# Auxiliary Classifier (used only during training)
# ---------------------------------------------------------
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),         # → 4×4
            BasicConv2d(in_channels, 128, kernel_size=1),   # → 128×4×4
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------
# GoogLeNet Main Model
# ---------------------------------------------------------
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        # ---------------- Stem -------------------
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)   # 112x112
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)                  # 56x56

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)                  # 28x28

        # ---------------- Inception 3a, 3b -------------------
        self.inception3a = Inception(192,  
                                     64, 
                                     96, 128, 
                                     16, 32, 
                                     32)
        self.inception3b = Inception(256, 
                                     128, 
                                     128, 192, 
                                     32, 96, 
                                     64)

        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)                  # 14x14

        # ---------------- Inception 4a–4e -------------------
        self.inception4a = Inception(480,
                                     192,
                                     96, 208,
                                     16, 48,
                                     64)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)

        self.inception4b = Inception(512,
                                     160,
                                     112, 224,
                                     24, 64,
                                     64)
        self.inception4c = Inception(512,
                                     128,
                                     128, 256,
                                     24, 64,
                                     64)
        self.inception4d = Inception(512,
                                     112,
                                     144, 288,
                                     32, 64,
                                     64)

        if aux_logits:
            self.aux2 = InceptionAux(528, num_classes)

        self.inception4e = Inception(528,
                                     256,
                                     160, 320,
                                     32, 128,
                                     128)

        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)                  # 7x7

        # ---------------- Inception 5a, 5b -------------------
        self.inception5a = Inception(832,
                                     256,
                                     160, 320,
                                     32, 128,
                                     128)

        self.inception5b = Inception(832,
                                     384,
                                     192, 384,
                                     48, 128,
                                     128)

        # ---------------- Classifier -------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)


    def forward(self, x):
        # Stem
        x = self.conv1(x)       # (64,112,112)
        x = self.maxpool1(x)    # (64,56,56)

        x = self.conv2(x)
        x = self.conv3(x)       # (192,56,56)
        x = self.maxpool2(x)    # (192,28,28)

        # Inception 3
        x = self.inception3a(x) # (256,28,28)
        x = self.inception3b(x) # (480,28,28)
        x = self.maxpool3(x)    # (480,14,14)

        # Inception 4
        x = self.inception4a(x) # (512,14,14)

        aux1_out = None
        if self.aux_logits and self.training:
            aux1_out = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        aux2_out = None
        if self.aux_logits and self.training:
            aux2_out = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)    # (832,7,7)

        # Inception 5
        x = self.inception5a(x) # (832,7,7)
        x = self.inception5b(x) # (1024,7,7)

        # Classifier
        x = self.avgpool(x)     # (1024,1,1)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux1_out, aux2_out  # (main, aux1, aux2)

        return x


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------
if __name__ == "__main__":
    model = GoogLeNet(num_classes=1000, aux_logits=True)
    x = torch.randn(1, 3, 224, 224)
    model.train()
    y = model(x)
    print("Output shapes:")
    print("Main:", y[0].shape)
    print("Aux1:", y[1].shape)
    print("Aux2:", y[2].shape)

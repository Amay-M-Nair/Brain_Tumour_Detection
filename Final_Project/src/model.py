"""The from-scratch CNN, carried over unchanged from Days 3-5.

ConvBlock -> BrainTumourCNN (feature extractor) -> MLPHead (classifier),
assembled as BrainTumourNet. No pretrained weights anywhere: every layer here
was derived by hand in the reference notebooks.
"""
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU.

    bias=False because BatchNorm immediately subtracts the batch mean, which
    cancels any constant the convolution's bias would have added.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BrainTumourCNN(nn.Module):
    """Feature extractor: 1 -> 32 -> 64 -> 128 -> 256 channels, ending in a
    global average pool so the head's input size is independent of image size.

    deep=True inserts a second convolution at stages 3 and 4. That does not add
    resolution, it adds receptive field: the plain stack sees only 38 px of a
    128 px scan at the deepest layer, which is too little context to judge a
    lesion's margin or its position relative to the midline.
    """

    def __init__(self, deep=False):
        super().__init__()
        self.deep = deep
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        if deep:
            self.block3b = ConvBlock(128, 128)
            self.block4b = ConvBlock(256, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap  = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.block3(x)
        if self.deep:
            x = self.block3b(x)
        x = self.pool(x)
        x = self.block4(x)
        if self.deep:
            x = self.block4b(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)


class MLPHead(nn.Module):
    """256 -> 128 -> 64 -> num_classes, returning raw logits.

    No softmax: nn.CrossEntropyLoss applies log-softmax internally, and
    applying it twice flattens the gradients.
    """

    def __init__(self, in_features=256, num_classes=4, dropout=0.4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.75))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


class BrainTumourNet(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4, deep=False):
        super().__init__()
        self.features = BrainTumourCNN(deep=deep)
        self.head     = MLPHead(256, num_classes, dropout)

    def forward(self, x):
        return self.head(self.features(x))


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def receptive_field(deep=False):
    """Receptive field in pixels at the deepest convolution.

    Walks the stack tracking (rf, jump): a 3x3 conv adds 2*jump, a 2x2 stride-2
    pool adds 1*jump and then doubles the jump.
    """
    rf, jump = 1, 1
    layers = (["conv", "pool", "conv", "pool", "conv", "conv", "pool", "conv", "conv"]
              if deep else
              ["conv", "pool", "conv", "pool", "conv", "pool", "conv"])
    for layer in layers:
        if layer == "conv":
            rf += 2 * jump
        else:
            rf += jump
            jump *= 2
    return rf

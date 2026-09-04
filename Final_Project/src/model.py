"""The network, assembled from components derived in the reference notebooks.

Nothing here is new: those notebooks built convolution three times over (nested
loops, im2col, nn.Conv2d) and derived ConvBlock, the four-stage stack and the
head. Same names, so the lineage stays traceable.

No pretrained backbone anywhere. Every weight starts from Kaiming init and is
learned from this dataset; a fine-tuned ResNet would score higher and
demonstrate nothing about whether the layers were understood.
"""
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "elu": nn.ELU}

# Kaiming's gain is derived for ReLU; reusing it for ELU would leave the network
# at the wrong scale, making an activation comparison a test of initialisation.
KAIMING_NONLINEARITY = {"relu": "relu", "elu": "linear"}


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> activation.

    bias=False because BatchNorm immediately subtracts the batch mean, cancelling
    any constant the bias added -- a parameter to initialise, store and update
    for no change in the function.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, activation="relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; "
                             f"use one of {sorted(ACTIVATIONS)}")
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = ACTIVATIONS[activation](inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in",
                                nonlinearity=KAIMING_NONLINEARITY[activation])
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BrainTumourCNN(nn.Module):
    """Feature extractor: 1 -> 32 -> 64 -> 128 -> 256, ending in a global average pool.

    deep=True adds a second convolution at stages 3 and 4 -- not resolution,
    receptive field: 38px of a 128px scan becomes 62px. 38px is too little
    context to judge a lesion's margin or its position relative to the midline.
    A prediction, tested in Phase 4.

    Global average pool rather than flatten means the head's input size does not
    depend on input resolution, so the resolution ablation needs no
    architectural change.
    """

    def __init__(self, deep=False, activation="relu"):
        super().__init__()
        self.deep = deep
        self.activation = activation
        block = lambda i, o: ConvBlock(i, o, activation=activation)

        self.block1 = block(1, 32)
        self.block2 = block(32, 64)
        self.block3 = block(64, 128)
        self.block4 = block(128, 256)
        if deep:
            self.block3b = block(128, 128)
            self.block4b = block(256, 256)
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
        return self.gap(x).flatten(1)


class MLPHead(nn.Module):
    """256 -> 128 -> 64 -> num_classes, returning raw logits.

    No softmax: CrossEntropyLoss applies log-softmax internally, and applying it
    twice flattens the gradients -- a bug that trains without erroring.
    """

    def __init__(self, in_features=256, num_classes=3, dropout=0.4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.75))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


class BrainTumourNet(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4, deep=False, activation="relu"):
        super().__init__()
        self.features = BrainTumourCNN(deep=deep, activation=activation)
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
    pool adds 1*jump then doubles it. Computed rather than quoted, because this
    number is the argument for the deeper variant.
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

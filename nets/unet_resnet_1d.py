import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        out_channels = channels * self.expansion

        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)

        self.conv3 = nn.Conv1d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.relu(out + identity)
        return out


class ResNet50Encoder1D(nn.Module):
    """
    1D adaptation of the standard ResNet-50 encoder.

    Stage depths are the canonical ResNet-50 layout: [3, 4, 6, 3].
    """

    def __init__(self, input_channels=9):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=3, stride=1)   # 256, /4
        self.layer2 = self._make_layer(128, blocks=4, stride=2)  # 512, /8
        self.layer3 = self._make_layer(256, blocks=6, stride=2)  # 1024, /16
        self.layer4 = self._make_layer(512, blocks=3, stride=2)  # 2048, /32

    def _make_layer(self, channels, blocks, stride):
        layers = [Bottleneck1D(self.in_channels, channels, stride=stride)]
        self.in_channels = channels * Bottleneck1D.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck1D(self.in_channels, channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.stem(x)       # 64, /2
        x = self.pool(x0)       # /4
        x1 = self.layer1(x)     # 256, /4
        x2 = self.layer2(x1)    # 512, /8
        x3 = self.layer3(x2)    # 1024, /16
        x4 = self.layer4(x3)    # 2048, /32
        return x0, x1, x2, x3, x4


class DecoderBlock1D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = nn.Sequential(
            nn.Conv1d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class ResNet50UNet1D(nn.Module):
    """
    1D U-Net using a true ResNet-50 bottleneck encoder.

    Input:
        [batch, 9, n_bases]

    Output:
        logits [batch, 1, n_bases]
    """

    def __init__(self, input_channels=9):
        super().__init__()
        self.encoder = ResNet50Encoder1D(input_channels=input_channels)

        self.up4 = DecoderBlock1D(2048, 1024, 512)
        self.up3 = DecoderBlock1D(512, 512, 256)
        self.up2 = DecoderBlock1D(256, 256, 128)
        self.up1 = DecoderBlock1D(128, 64, 64)

        self.final_refine = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        input_length = x.shape[-1]
        x0, x1, x2, x3, x4 = self.encoder(x)

        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)

        x = F.interpolate(
            x,
            size=input_length,
            mode="linear",
            align_corners=False,
        )
        x = self.final_refine(x)
        return self.outc(x)

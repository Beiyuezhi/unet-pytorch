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


class ResNetEncoder1D(nn.Module):
    """
    Generic 1D ResNet bottleneck encoder.

    ResNet-50:  [3, 4, 6, 3]
    ResNet-101: [3, 4, 23, 3]
    """

    def __init__(self, input_channels=9, layers=(3, 4, 6, 3)):
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers must contain exactly four stage depths")

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

        self.layer1 = self._make_layer(64, blocks=layers[0], stride=1)   # 256, /4
        self.layer2 = self._make_layer(128, blocks=layers[1], stride=2)  # 512, /8
        self.layer3 = self._make_layer(256, blocks=layers[2], stride=2)  # 1024, /16
        self.layer4 = self._make_layer(512, blocks=layers[3], stride=2)  # 2048, /32

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


class ResNet50Encoder1D(ResNetEncoder1D):
    def __init__(self, input_channels=9):
        super().__init__(input_channels=input_channels, layers=(3, 4, 6, 3))


class ResNet101Encoder1D(ResNetEncoder1D):
    def __init__(self, input_channels=9):
        super().__init__(input_channels=input_channels, layers=(3, 4, 23, 3))


class AttentionGate1D(nn.Module):
    """
    Attention U-Net style gate for a 1D skip connection.

    The decoder feature acts as the gating signal and learns a per-position,
    per-skip attention mask before the encoder feature is concatenated.
    """

    def __init__(self, skip_channels, gating_channels, inter_channels):
        super().__init__()
        self.theta = nn.Conv1d(
            skip_channels, inter_channels, kernel_size=1, bias=False
        )
        self.phi = nn.Conv1d(
            gating_channels, inter_channels, kernel_size=1, bias=False
        )
        self.norm = nn.BatchNorm1d(inter_channels)
        self.psi = nn.Conv1d(inter_channels, 1, kernel_size=1, bias=True)

    def forward(self, skip, gating):
        if gating.shape[-1] != skip.shape[-1]:
            gating = F.interpolate(
                gating,
                size=skip.shape[-1],
                mode="linear",
                align_corners=False,
            )

        attention = self.theta(skip) + self.phi(gating)
        attention = F.relu(self.norm(attention), inplace=True)
        attention = torch.sigmoid(self.psi(attention))
        return skip * attention


class DecoderBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_attention=False,
    ):
        super().__init__()
        self.reduce = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.attention = (
            AttentionGate1D(
                skip_channels=skip_channels,
                gating_channels=out_channels,
                inter_channels=max(out_channels // 2, 16),
            )
            if use_attention
            else None
        )
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
        if self.attention is not None:
            skip = self.attention(skip, x)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class ResNetUNet1D(nn.Module):
    """
    1D U-Net with a bottleneck ResNet encoder.

    Input:
        [batch, 9, n_bases]

    Output:
        logits [batch, 1, n_bases]
    """

    def __init__(
        self,
        input_channels=9,
        layers=(3, 4, 6, 3),
        attention_gates=False,
    ):
        super().__init__()
        self.encoder = ResNetEncoder1D(
            input_channels=input_channels,
            layers=layers,
        )

        self.up4 = DecoderBlock1D(
            2048, 1024, 512, use_attention=attention_gates
        )
        self.up3 = DecoderBlock1D(
            512, 512, 256, use_attention=attention_gates
        )
        self.up2 = DecoderBlock1D(
            256, 256, 128, use_attention=attention_gates
        )
        self.up1 = DecoderBlock1D(
            128, 64, 64, use_attention=attention_gates
        )

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


class ResNet50UNet1D(ResNetUNet1D):
    def __init__(self, input_channels=9, attention_gates=False):
        super().__init__(
            input_channels=input_channels,
            layers=(3, 4, 6, 3),
            attention_gates=attention_gates,
        )


class ResNet101UNet1D(ResNetUNet1D):
    def __init__(self, input_channels=9, attention_gates=False):
        super().__init__(
            input_channels=input_channels,
            layers=(3, 4, 23, 3),
            attention_gates=attention_gates,
        )

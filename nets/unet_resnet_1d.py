import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA1D(nn.Module):
    """Efficient Channel Attention for 1D feature maps."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        kernel_size = max(kernel_size, 3)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).squeeze(-1).unsqueeze(1)
        y = torch.sigmoid(self.conv(y)).squeeze(1).unsqueeze(-1)
        return x * y


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
        return self.relu(out + identity)


class ResNetEncoder1D(nn.Module):
    """
    ResNet-50  -> [3, 4, 6, 3]
    ResNet-101 -> [3, 4, 23, 3]
    """

    def __init__(self, input_channels=9, layers=(3, 4, 6, 3)):
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

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.eca0 = ECA1D(64)
        self.eca1 = ECA1D(256)
        self.eca2 = ECA1D(512)
        self.eca3 = ECA1D(1024)
        self.eca4 = ECA1D(2048)

    def _make_layer(self, channels, blocks, stride):
        layers = [Bottleneck1D(self.in_channels, channels, stride=stride)]
        self.in_channels = channels * Bottleneck1D.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck1D(self.in_channels, channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.eca0(self.stem(x))
        x = self.pool(x0)
        x1 = self.eca1(self.layer1(x))
        x2 = self.eca2(self.layer2(x1))
        x3 = self.eca3(self.layer3(x2))
        x4 = self.eca4(self.layer4(x3))
        return x0, x1, x2, x3, x4


class DilatedContext1D(nn.Module):
    """Multi-scale context block with dilation 1/2/4/8."""

    def __init__(self, channels=2048, hidden_channels=512):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        branch_channels = hidden_channels // 4
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        branch_channels,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for d in (1, 2, 4, 8)
            ]
        )

        self.project = nn.Sequential(
            nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.reduce(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.project(x)
        return self.relu(x + identity)


class AttentionGate1D(nn.Module):
    """Attention U-Net gate for a 1D encoder skip."""

    def __init__(self, skip_channels, gating_channels, inter_channels):
        super().__init__()
        self.theta = nn.Conv1d(
            skip_channels, inter_channels, kernel_size=1, bias=False
        )
        self.phi = nn.Conv1d(
            gating_channels, inter_channels, kernel_size=1, bias=False
        )
        self.norm = nn.BatchNorm1d(inter_channels)
        self.psi = nn.Conv1d(inter_channels, 1, kernel_size=1)

    def forward(self, skip, gating):
        if gating.shape[-1] != skip.shape[-1]:
            gating = F.interpolate(
                gating,
                size=skip.shape[-1],
                mode="linear",
                align_corners=False,
            )

        attn = self.theta(skip) + self.phi(gating)
        attn = F.relu(self.norm(attn), inplace=True)
        attn = torch.sigmoid(self.psi(attn))
        return skip * attn


class DecoderBlock1D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.attention = AttentionGate1D(
            skip_channels=skip_channels,
            gating_channels=out_channels,
            inter_channels=max(out_channels // 2, 16),
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
        skip = self.attention(skip, x)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class EnhancedResNetUNet1D(nn.Module):
    """
    Full AB1 model:
      ResNet50/101 encoder
      + ECA
      + Dilated Context
      + Attention U-Net decoder
      + segmentation head
      + start/end boundary heads
    """

    def __init__(self, input_channels=9, layers=(3, 4, 6, 3)):
        super().__init__()
        self.encoder = ResNetEncoder1D(
            input_channels=input_channels,
            layers=layers,
        )
        self.context = DilatedContext1D(2048, 512)

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

        self.segmentation_head = nn.Conv1d(32, 1, kernel_size=1)
        self.boundary_head = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 2, kernel_size=1),
        )

    def forward(self, x):
        input_length = x.shape[-1]
        x0, x1, x2, x3, x4 = self.encoder(x)
        x4 = self.context(x4)

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
        features = self.final_refine(x)

        return {
            "seg_logits": self.segmentation_head(features),
            "boundary_logits": self.boundary_head(features),
        }


class ResNet50UNet1D(EnhancedResNetUNet1D):
    def __init__(self, input_channels=9):
        super().__init__(
            input_channels=input_channels,
            layers=(3, 4, 6, 3),
        )


class ResNet101UNet1D(EnhancedResNetUNet1D):
    def __init__(self, input_channels=9):
        super().__init__(
            input_channels=input_channels,
            layers=(3, 4, 23, 3),
        )

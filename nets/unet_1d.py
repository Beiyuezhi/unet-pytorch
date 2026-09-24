import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv1d(nn.Module):
    """Standard double-convolution block for 1D U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            DoubleConv1d(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up1d(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv = DoubleConv1d(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    """Standard 1D U-Net baseline for AB1 usable-region segmentation."""

    def __init__(self, input_channels: int = 9, base_channels: int = 32):
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.inc = DoubleConv1d(input_channels, c1)
        self.down1 = Down1d(c1, c2)
        self.down2 = Down1d(c2, c3)
        self.down3 = Down1d(c3, c4)
        self.down4 = Down1d(c4, c5)

        self.up1 = Up1d(c5, c4, c4)
        self.up2 = Up1d(c4, c3, c3)
        self.up3 = Up1d(c3, c2, c2)
        self.up4 = Up1d(c2, c1, c1)

        self.outc = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

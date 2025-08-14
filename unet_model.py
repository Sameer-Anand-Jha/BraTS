import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.enc1 = DoubleConv(n_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512 if use_skip else 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256 if use_skip else 128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128 if use_skip else 64, 64)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x2 = self.enc2(x2)
        x3 = self.pool(x2)
        x3 = self.enc3(x3)
        x4 = self.pool(x3)
        bottleneck = self.bottleneck(x4)
        u3 = self.up3(bottleneck)
        if self.use_skip: u3 = torch.cat([u3, x3], dim=1)
        d3 = self.dec3(u3)
        u2 = self.up2(d3)
        if self.use_skip: u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        if self.use_skip: u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1)
        out = self.final(d1)
        return out, bottleneck
"""U-Net from scratch + optional pretrained-encoder U-Net (segmentation_models_pytorch)."""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, features=(32, 64, 128, 256, 512)):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        c = in_channels
        for f in features[:-1]:
            self.downs.append(DoubleConv(c, f))
            c = f
        self.bottleneck = DoubleConv(features[-2], features[-1])
        for f in reversed(features[:-1]):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))
        self.head = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[-(i // 2 + 1)]
            x = self.ups[i + 1](torch.cat([skip, x], dim=1))
        return self.head(x)  # logits


def build_model(arch="unet", pretrained=False):
    if arch == "unet":
        return UNet()
    if arch in ("resnet34", "resnet50"):
        import segmentation_models_pytorch as smp
        return smp.Unet(arch, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)
    raise ValueError(f"Unknown arch: {arch}")

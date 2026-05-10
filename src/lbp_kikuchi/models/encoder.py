import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    """
    ResNet18 adapted to grayscale EBSD patterns.

    Input:  (B, 1, H, W)  — any H, W; resized to (img_size, img_size) on the fly
    Output: (B, out_dim)

    First conv weights are averaged over the RGB dimension to preserve pretrained
    spatial structure when collapsing 3 channels → 1.

    Args:
        out_dim:  Output embedding dimension.
        img_size: Spatial size the pattern is resized to before ResNet18.
                  Use 224 (ResNet standard) for a 6× speedup vs 480×640.
                  None disables resizing (uses raw pattern size — very slow).
    """

    def __init__(self, out_dim: int = 128, img_size: int = 224):
        super().__init__()
        self.img_size = img_size

        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        w = net.conv1.weight.data  # (64, 3, 7, 7)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.conv1.weight.data = w.mean(dim=1, keepdim=True)

        net.fc = nn.Linear(512, out_dim)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.img_size is not None and (x.shape[-2] != self.img_size or x.shape[-1] != self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        return self.net(x)

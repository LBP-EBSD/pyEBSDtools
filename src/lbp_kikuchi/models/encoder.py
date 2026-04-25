import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """
    ResNet18 adapted to grayscale EBSD patterns.

    Input:  (B, 1, H, W)
    Output: (B, out_dim)

    First conv weights are averaged over the RGB dimension to preserve pretrained
    spatial structure when collapsing 3 channels → 1.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()

        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        w = net.conv1.weight.data  # (64, 3, 7, 7)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.conv1.weight.data = w.mean(dim=1, keepdim=True)

        net.fc = nn.Linear(512, out_dim)
        self.net = net

    def forward(self, x):
        return self.net(x)

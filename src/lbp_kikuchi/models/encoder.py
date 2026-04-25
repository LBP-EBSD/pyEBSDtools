# src/lbp_kikuchi/models/encoder.py

import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()

        # backbone = models.resnet18(pretrained=False)
        # backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        backbone = models.resnet18(pretrained=True)
        
        # average weights across RGB → 1 channel
        w = backbone.conv1.weight
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = w.mean(dim=1, keepdim=True)
        
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.backbone(x)      # (B,512,1,1)
        x = x.flatten(1)
        return self.fc(x)
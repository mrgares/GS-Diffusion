import torch.nn as nn
import torchvision.models as models

class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(size=(352, 640)) # Assuming input size is 352x640
        )

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        return out
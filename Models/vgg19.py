import torch.nn as nn
from torchvision import models


class VGG19(nn.Module):

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        weights = (
            models.VGG19_Weights.DEFAULT
            if pretrained else None
        )

        self.backbone = models.vgg19(weights=weights)

        in_features = self.backbone.classifier[-1].in_features

        self.backbone.classifier[-1] = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_cam_target_layer(self):
        return self.backbone.features[-1]

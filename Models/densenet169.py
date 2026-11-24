import torch.nn as nn
from torchvision import models


class DenseNet169(nn.Module):

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        weights = (
            models.DenseNet169_Weights.DEFAULT
            if pretrained else None
        )

        self.backbone = models.densenet169(weights=weights)

        in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_cam_target_layer(self):
        return self.backbone.features.denseblock4.denselayer32.conv2

import torch.nn as nn
from torchvision import models


class MobileNetV3(nn.Module):

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        weights = (
            models.MobileNet_V3_Large_Weights.DEFAULT
            if pretrained else None
        )

        self.backbone = models.mobilenet_v3_large(weights=weights)

        # MobileNetV3-Large feature output is 960-dimensional.
        in_features = self.backbone.classifier[0].in_features

        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_cam_target_layer(self):
        return self.backbone.features[-1]

import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained else None
        )

        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_cam_target_layer(self):
        return self.backbone.layer4[-1].conv3

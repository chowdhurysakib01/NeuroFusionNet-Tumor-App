import torch
import torch.nn as nn

from Models.dual_backbone import DualBackboneBase


class DenseNet169ResNet50Fusion(DualBackboneBase):

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__(pretrained)

        self.classifier = nn.Sequential(
            nn.Linear(1664 + 2048, 512),

            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        dense_features, res_features = self.extract_features(x)

        fused = torch.cat(
            [dense_features, res_features],
            dim=1
        )

        return self.classifier(fused)

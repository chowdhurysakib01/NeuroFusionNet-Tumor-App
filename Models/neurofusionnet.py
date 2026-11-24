import torch
import torch.nn as nn

from Models.dual_backbone import DualBackboneBase


class NeuroFusionNet(DualBackboneBase):

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__(pretrained)

        self.dense_projection = nn.Sequential(
            nn.Linear(1664, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True)
        )

        self.resnet_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1024),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),

            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        dense_features, res_features = self.extract_features(x)

        dense_features = self.dense_projection(dense_features)

        res_features = self.resnet_projection(res_features)

        concatenated = torch.cat(
            [dense_features, res_features],
            dim=1
        )

        attention_weights = self.attention(concatenated)

        attended_features = concatenated * attention_weights

        fused_features = concatenated + attended_features

        return self.classifier(fused_features)

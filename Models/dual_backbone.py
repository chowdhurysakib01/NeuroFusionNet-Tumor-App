import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class DualBackboneBase(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        dense_weights = (
            models.DenseNet169_Weights.DEFAULT
            if pretrained else None
        )

        resnet_weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained else None
        )

        self.densenet = models.densenet169(
            weights=dense_weights
        )

        self.resnet = models.resnet50(
            weights=resnet_weights
        )

        self.densenet.classifier = nn.Identity()
        self.resnet.fc = nn.Identity()

    def extract_features(self, x):

        dense_features = self.densenet.features(x)

        dense_features = F.relu(
            dense_features,
            inplace=False
        )

        dense_features = F.adaptive_avg_pool2d(
            dense_features,
            (1, 1)
        )

        dense_features = torch.flatten(
            dense_features,
            1
        )

        res_features = self.resnet.conv1(x)
        res_features = self.resnet.bn1(res_features)
        res_features = self.resnet.relu(res_features)
        res_features = self.resnet.maxpool(res_features)

        res_features = self.resnet.layer1(res_features)
        res_features = self.resnet.layer2(res_features)
        res_features = self.resnet.layer3(res_features)
        res_features = self.resnet.layer4(res_features)

        res_features = self.resnet.avgpool(res_features)

        res_features = torch.flatten(
            res_features,
            1
        )

        return dense_features, res_features

    def get_cam_target_layers(self):

        return [
            self.densenet.features.denseblock4.denselayer32.conv2,
            self.resnet.layer4[-1].conv3
        ]

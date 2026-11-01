"""CIFAR ResNet-18 (3x3 stem, no maxpool), normalization inside the model so inputs are in [0,1].
Follows sample_repos/Unlearn-Saliency-master/Classification/models/ResNet.py."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mu.data import CIFAR10_MEAN, CIFAR10_STD


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10):
        super().__init__()
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        """512-d penultimate embedding (avgpool output)."""
        x = (x - self.mean) / self.std
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        return self.fc(self.features(x))


def resnet18(num_classes=10):
    return ResNet((2, 2, 2, 2), num_classes)

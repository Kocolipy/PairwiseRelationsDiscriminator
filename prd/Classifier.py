import torch
from torch import nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.mlp = torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = []
        for i in range(2):
            y = self.cnn(x[:, 3*i:3*i + 3, :])
            features.append(y)
        features = torch.cat(features, 1)
        x = self.mlp(features)
        return x


class DiffResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.mlp = torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = self.cnn(x[:, 0:3, :])
        features = features.add(-1*self.cnn(x[:, 3:6, :]))
        x = self.mlp(features)
        return x


class L1ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.mlp = torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = self.cnn(x[:, 0:3, :])
        features = features.add(-1*self.cnn(x[:, 3:6, :]))
        features = torch.abs(features)
        x = self.mlp(features)
        return x


class L2ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.mlp = torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = self.cnn(x[:, 0:3, :])
        features = features.add(-1*self.cnn(x[:, 3:6, :]))
        features = torch.pow(features, 2)
        x = self.mlp(features)
        return x


class DotResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        a = self.cnn(x[:, 0:3, :])
        anorm = torch.nn.functional.normalize(a, p=2, dim=1)

        b = self.cnn(x[:, 3:6, :])
        bnorm = torch.nn.functional.normalize(b, p=2, dim=1)

        dot = (anorm*bnorm).sum(1)
        return (torch.abs(dot).clamp(0, 1)).unsqueeze(-1)


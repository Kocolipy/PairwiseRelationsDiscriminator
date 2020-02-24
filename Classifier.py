import torch
from torch import nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18()
        self.mlp = torch.nn.Sequential(
            nn.Linear(2000, 2000, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = []
        for i in range(2):
            features.append(self.cnn(x[:, 3*i:3*i + 3, :]))
        features = torch.cat(features, 1)
        x = self.mlp(features)
        return x

        # return self.cnn(x)

class CNN(nn.Module):
    """"
    Wild Rel Network
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = torch.nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(6 * 512, 6 * 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(6 * 512, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass
    def forward(self, x):
        features = []
        for i in range(6):
            features.append(self.cnn(x[:,i:i+1,:].float()))
        features = torch.cat(features, 1)
        features = torch.flatten(features, start_dim=1)
        x = self.classifier(features)
        return x

class CNNSplit(nn.Module):
    """"
    Wild Rel Network with Row features extractor
    """
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.rowfeats = nn.Sequential(
            nn.Linear(3 * 512, 3 * 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(3 * 512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass
    def forward(self, x):
        features = None
        for i in range(2):
            row = []
            for j in range(3):
                row.append(self.cnn(x[:, 3*i + j:3*i + j +1, :].float()))
            row = torch.flatten(torch.cat(row, 1), start_dim=1)
            if features is None:
                features = self.rowfeats(row)
            else:
                features = features.add(-1*self.rowfeats(row))
        x = self.classifier(features)
        return x

class CNN2(nn.Module):
    """"
    Large channels
    """
    def __init__(self):
        super(CNN2, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 160, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(160, 160 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(160 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(160 * 2, 160 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(160 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(160 * 4, 160 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(160 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(160 * 8, 1, 4, 1, 0, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(6 * 49, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass
    def forward(self, x):
        features = []
        for i in range(6):
            features.append(self.cnn(x[:,i:i+1,:].float()))
        features = torch.cat(features, 1)
        features = torch.flatten(features, start_dim=1)
        x = self.classifier(features)
        return x


class FFNN(nn.Module):
    def __init__(self, p=0.5):
        super(FFNN, self).__init__()
        self.classifier = torch.nn.Sequential(
            nn.Linear(128*6, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(num_features=2048),
            nn.Dropout(p),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(p),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class FFNN2(nn.Module):
    def __init__(self, p=0.5):
        super(FFNN2, self).__init__()
        self.classifier = torch.nn.Sequential(
            nn.Linear(768, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(num_features=2048),
            nn.Dropout(p),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(p),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

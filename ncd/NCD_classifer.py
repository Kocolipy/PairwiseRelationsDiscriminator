import torch
from torch import nn
import torchvision.models as models

class NCDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.ffnn = torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = []
        for i in range(x.shape[1]):
            features.append(self.cnn(x[:, i, :]))

        centroid = 0.5*(features[0] + features[1])
        features = torch.stack(features, 1)

        decentre = features - centroid.unsqueeze(1).repeat(1, 10, 1)

        output = []
        for i in range(decentre.shape[1]):
            output.append(self.ffnn(decentre[:, i, :]))

        return torch.stack(output, 1)
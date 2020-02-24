import torch
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from torch import nn
import numpy as np
import os
import pathlib
import sys

import RavenDataLoader

class Reshaper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.reshape(x, self.dim)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(2592, 128),
            nn.LeakyReLU()
        )
        self.mu = nn.Linear(128, 20)
        self.var = nn.Linear(128, 20)

        self.decoder = nn.Sequential(
            nn.Linear(20, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 3200),
            nn.LeakyReLU(),
            Reshaper((-1, 32, 10, 10)),
            nn.ConvTranspose2d(32, 32, 4, 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 4, 4),
            nn.LeakyReLU(),
            nn.Sigmoid()
        )
    def sample(self, mu, logvar):
        # Sample from distribution with parameters mu, logvar.
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size(), device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        return mu + std*eps

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        logvar = self.var(encoded)

        decoded = self.decoder(self.sample(mu, logvar))
        # decoded = torch.where(torch.isnan(decoded), torch.zeros_like(decoded), decoded)
        return decoded, mu[:, np.newaxis,:], logvar[:, np.newaxis,:]

class DenoisingRAE(nn.Module):
    def __init__(self):
        super(DenoisingRAE, self).__init__()

        self.cnn = models.resnet18()
        self.encoder = nn.Sequential(
            nn.Linear(1000, 128),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 400),
            nn.LeakyReLU(),
            Reshaper((-1, 1, 20, 20)),
            nn.ConvTranspose2d(1, 20, 4, 2, 1, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(20, 160, 4, 2, 1, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(160, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        encoded = self.encoder(self.cnn(x))
        return self.decoder(encoded)


class DenoisingAE(nn.Module):
    def __init__(self):
        super(DenoisingAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(512, 128),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 400),
            nn.LeakyReLU(),
            Reshaper((-1, 1, 20, 20)),
            nn.ConvTranspose2d(1, 20, 4, 2, 1, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(20, 160, 4, 2, 1, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(160, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


if __name__ == '__main__':
    cwd = pathlib.Path(os.getcwd())

    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    hyperparams = {"batch_size": 20,
                   "raven_mode": "All",
                   "num_epochs": 100,
                   "lr": 0.001,
                   "wd": 0.0001,
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    learning_rate = 0.02

    # data_path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]
    data_path = cwd / "data" / hyperparams["raven_mode"]

    # Define model, loss and optimizer
    model = DenoisingRAE()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hyperparams["wd"])

    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    # Load Training Data
    train_loader = RavenDataLoader.SampleLoader(data_path / "train", hyperparams)
    val_loader = RavenDataLoader.SampleLoader(data_path / "val", hyperparams)

    # Load from previous checkpoint
    ckpt_path = data_path / "ae_ckpt"
    ckpt_path.mkdir(exist_ok=True)
    if hyperparams["ckpt"]:
        checkpoint = torch.load(str(ckpt_path / str(hyperparams["ckpt"])))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loading Checkpoint", hyperparams["ckpt"])

    # Move model and optimiser to CUDA if available
    model.to(device)
    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    for epoch in range(hyperparams["num_epochs"]):
        losses = {"train": 0.0, "val": 0.0}
        scheduler.step()
        print(scheduler.get_lr())

        # Training Phase
        model.train()
        optimizer.zero_grad()
        for data in train_loader:
            data = data.to(device).float()
            output = []
            for i in range(data.shape[1]):
                output.append(model(data[:, i:i + 1, :]))
            output = torch.cat(output, 1)

            loss = criterion(output, data)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses["train"] += loss.item()

            losses["train"] = losses["train"] / len(train_loader)
        print("Training Loss for epoch {0}: {1}".format(
            epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0)+1,
            losses["train"])
        )

        # Validation Phase
        with torch.no_grad():
            model.eval()

            for data in val_loader:
                data = data.to(device).float()
                output = []
                for i in range(data.shape[1]):
                    output.append(model(data[:, i:i + 1, :]))
                output = torch.cat(output, 1)

                loss = criterion(output, data)
                losses["val"] += loss.item()

                losses["val"] = losses["val"] / len(val_loader)
            print("Validation Loss for epoch {0}: {1}".format(
                epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1,
                losses["val"])
            )

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': losses["train"],
            'validation_loss': losses["val"],
        }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))
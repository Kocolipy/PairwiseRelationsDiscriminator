import torch
from torch.utils.data import DataLoader

import os
import pathlib
import sys

import Autoencoder
import Classifier
import RavenDataLoader
import utils

if __name__ == '__main__':
    cwd = pathlib.Path(os.getcwd())

    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    hyperparams = {"batch_size": 20,
                   "raven_mode": "All",
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    data_path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]
    #  data_path = cwd / "data" / hyperparams["raven_mode"]

    ckpt_file = data_path / "ae_ckpt" / "6"

    ae = utils.loadAutoEncoder(Autoencoder.DenoisingRAE, ckpt_file, device)

    data_loader = RavenDataLoader.SampleLoader(data_path / "train", hyperparams)

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device).float()

            output = []
            for i in range(data.shape[1]):
                output.append(ae(data[:, i:i + 1, :]))
            output = torch.cat(output, 1)

            utils.displaySample(data[0,:].cpu())
            utils.displaySample(output[0,:].cpu())
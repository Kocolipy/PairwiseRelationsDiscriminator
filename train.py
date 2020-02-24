from collections import Counter
import numpy as np
import os
import pathlib
import sys
import torch
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

import Autoencoder
import Classifier
import RavenDataLoader
import utils

if __name__ == '__main__':
    cwd = pathlib.Path(os.getcwd())

    torch.multiprocessing.freeze_support()
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    hyperparams = {"batch_size": 20,
                   "raven_mode": "All",
                   "num_epochs": 80,
                   "lr": 0.001,
                   "wd": 0.0001,
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    data_path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]
    # data_path = cwd / "data" / hyperparams["raven_mode"]

    # # Load the autoencoder
    # aeckpt = data_path / "ae_ckpt" / "150"
    # ae = utils.loadAutoEncoder(Autoencoder.DenoisingAE, aeckpt, device)

    # Create the dataloader for training
    train_loader = RavenDataLoader.DualRowsRealFakeLoader(data_path, hyperparams)

    # Create the dataloader for validation loss (2 set of 3 tiles [discriminator task])
    valrows_loader = RavenDataLoader.DualRowsLoader(data_path/"val", hyperparams)

    # Create the dataloader for validation performance (Raven task)
    val_loader = RavenDataLoader.ValidationLoader(data_path/"val", hyperparams)

    # Define Model
    classifier = Classifier.ResNet()
    # classifier = Classifier.CNN()
    # classifier = Classifier.CNN2()
    # classifier = Classifier.FFNN()

    # Optimiser
    learning_rate = 0.002
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=hyperparams["wd"])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    # Loss function
    criterion = torch.nn.BCELoss()
    criterion_val = torch.nn.BCELoss()
    # criterion = torch.nn.MSELoss()

    # Load from checkpoint
    ckpt_path = data_path / "ckpt"
    ckpt_path.mkdir(exist_ok=True)
    if hyperparams["ckpt"]:
        checkpoint = torch.load(str(ckpt_path / str(hyperparams["ckpt"])))
        classifier.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loading Checkpoint", hyperparams["ckpt"])

    # Move model and optimiser to CUDA if available
    classifier.to(device)
    # if torch.cuda.is_available():
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if torch.is_tensor(v):
    #                 state[k] = v.cuda()

    for epoch in range(hyperparams["num_epochs"]):
        # Training Phase
        print("Training Epoch", epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1)
        print("-----------------------------------")
        # scheduler.step()
        training_loss = 0
        classifier.train()
        for real_data, real_label, fake_data, fake_label in train_loader:
            optimizer.zero_grad()

            # Move data to device
            real_data = real_data.to(device).float()
            real_label = real_label[:, np.newaxis].float()
            real_label = real_label.to(device)
            fake_data = fake_data.to(device).float()
            fake_label = fake_label[:, np.newaxis].float()
            fake_label = fake_label.to(device)

            # Real Data Batch
            # real_data = ae.encode(real_data)
            real_output = classifier(real_data)
            real_loss = criterion(real_output, real_label)
            # print("real", real_output.cpu().tolist())

            # Fake Data Batch
            # fake_data = ae.encode(fake_data)
            fake_output = classifier(fake_data)
            fake_loss = criterion(fake_output, fake_label)
            # print("fake", fake_output.cpu().tolist())

            # Compute loss and backpropagate gradients
            # loss = (real_loss + fake_loss)/2
            loss = fake_loss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        training_loss = training_loss / len(train_loader)
        print("Training Loss: {0:.5f}".format(training_loss))

        # # Validation Phase
        # classifier.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for data, label in valrows_loader:
        #         data = data.to(device).float()
        #         label = label[:, np.newaxis].float()
        #         label = label.to(device)
        #
        #         # data = ae.encode(data)
        #         output = classifier(data)
        #         loss = criterion_val(output, label)
        #         val_loss += loss.item()
        #     val_loss = val_loss / len(valrows_loader)
        #     print("Validation Loss: {0:.5f}".format(val_loss))
        #
        #     val = []
        #     for count, (data, label) in enumerate(val_loader):
        #         data = data.to(device).float()
        #         label = label[:, np.newaxis].int()
        #         label = label.to(device)
        #
        #         # Data is of dimension (batch_size, answer_panel, 2 (first and second set), 6, image_w, image_h)
        #         min_scores = []
        #         avg_scores = []
        #         for option in range(data.shape[1]):
        #             vals = []
        #             for sample_set in range(data.shape[2]):
        #                 # encoded = ae.encode(data[:, option, sample_set, :])
        #                 # out = classifier(encoded)
        #
        #                 out = classifier(data[:, option, sample_set, :])
        #                 # print(out.tolist())
        #                 vals.append(out)
        #
        #             # Take the score of the two sample sets
        #             min_scores.append(torch.min(torch.stack(vals), dim=0).values)
        #             avg_scores.append(torch.mean(torch.stack(vals), dim=0))
        #         # Arrange the panels according to their scores
        #         ranks = np.argsort(-1 * np.array(torch.stack(min_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
        #
        #         # Get the position of the gold label
        #         scores = (torch.tensor(ranks) == (label - 1).cpu()).nonzero()[:, 1]
        #         val += scores.tolist()
        #
        #     # Compute RSME from the label positions
        #     rmse = np.sqrt(sum(map(lambda x: x ** 2, val)) / len(val))
        #     print("RSME:", rmse)
        #
        #     # Display the raw positions
        #     summary = Counter(val)
        #     print(summary)
        #     print()

        torch.save({
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss,
            # 'validation_loss': val_loss,
            # 'rsme': rmse,
            # 'summary': summary
        }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))

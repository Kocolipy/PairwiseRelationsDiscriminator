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
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    hyperparams = {"batch_size": 32,
                   "raven_mode": "All",
                   "num_epochs": 80,
                   "lr": 0.001,
                   "wd": 0.0001,
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    # data_path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]
    data_path = cwd / "data" / hyperparams["raven_mode"]

    # # Load the autoencoder
    # aeckpt = data_path / "ae_ckpt" / "150"
    # ae = utils.loadAutoEncoder(Autoencoder.DenoisingAE, aeckpt, device)

    # Create the dataloader for training
    # train_loader = RavenDataLoader.DualRowsRealFakeLoader(data_path, hyperparams)
    train_loader = RavenDataLoader.RandomRealFakeLoader(data_path/"train", hyperparams)

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
    learning_rate = 0.0002
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

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
        print(checkpoint["training_loss"])
        print("Loading Checkpoint", hyperparams["ckpt"])

    # Move model and optimiser to CUDA if available
    classifier.to(device)
    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.eval()

    for epoch in range(hyperparams["num_epochs"]):
        # Training Phase
        print("Training Epoch", epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1)
        print("-----------------------------------")
        training_loss = 0
        classifier.train()
        classifier.apply(set_bn_eval)

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
            loss = (real_loss + fake_loss)/2
            # loss = real_loss
            # print(loss.item())
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss = training_loss / len(train_loader)
        print("Training Loss: {0:.5f}".format(training_loss))

        if (epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1) > 30 and (epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1) % 5 == 0:
            # scheduler.step()
            # Validation Phase
            classifier.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, label in valrows_loader:
                    data = data.to(device).float()
                    label = label[:, np.newaxis].float()
                    label = label.to(device)

                    # data = ae.encode(data)
                    output = classifier(data)
                    loss = criterion_val(output, label)
                    val_loss += loss.item()
                val_loss = val_loss / len(valrows_loader)
                print("Validation Loss: {0:.5f}".format(val_loss))

                min_val = []
                avg_val = []
                sqdist_val = []
                combined_val = []
                for count, (data, label) in enumerate(val_loader):
                    data = data.to(device).float()
                    label = label[:, np.newaxis].int()
                    label = label.to(device)

                    # Data is of dimension (batch_size, answer_panel, 2 (first and second set), 6, image_w, image_h)
                    min_scores = []
                    avg_scores = []
                    sqdist_scores = []
                    combined_scores = []

                    combined_qns = (classifier.cnn(data[:, 0, 0, :3, :]) + classifier.cnn(data[:, 0, 1, :3, :]))/2
                    for option in range(data.shape[1]):
                        vals = []
                        option_feat = classifier.cnn(data[:, option, 0, 3:, :])

                        # features = combined_qns.add(-1 * option_feat)
                        features = torch.cat([combined_qns, option_feat], 1)
                        combined_scores.append(classifier.mlp(features))

                        for sample_set in range(data.shape[2]):
                            # encoded = ae.encode(data[:, option, sample_set, :])
                            # out = classifier(encoded)
                            out = classifier(data[:, option, sample_set, :])
                            # print(out.tolist())
                            vals.append(out)

                        # Take the score of the two sample sets
                        min_scores.append(torch.min(torch.stack(vals), dim=0).values)
                        avg_scores.append(torch.mean(torch.stack(vals), dim=0))
                        sqdist_scores.append(torch.mean((1 - torch.stack(vals))**2, dim=0)*2)

                    # Arrange the panels according to their scores
                    min_ranks = np.argsort(-1 * np.array(torch.stack(min_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                    avg_ranks = np.argsort(-1 * np.array(torch.stack(avg_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                    sqdist_ranks = np.argsort(np.array(torch.stack(sqdist_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                    combined_ranks = np.argsort(-1 * np.array(torch.stack(combined_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()

                    # Get the position of the gold label
                    min_scores = (torch.tensor(min_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                    avg_scores = (torch.tensor(avg_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                    sqdist_scores = (torch.tensor(sqdist_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                    combined_scores = (torch.tensor(combined_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                    min_val += min_scores.tolist()
                    avg_val += avg_scores.tolist()
                    sqdist_val += sqdist_scores.tolist()
                    combined_val += combined_scores.tolist()

                # Compute RSME from the label positions
                min_rmse = np.sqrt(sum(map(lambda x: x ** 2, min_val)) / len(min_val))
                print("Min RSME:", min_rmse)
                min_summary = Counter(min_val)
                print(min_summary)

                avg_rmse = np.sqrt(sum(map(lambda x: x ** 2, avg_val)) / len(avg_val))
                print("Avg RSME:", avg_rmse)
                avg_summary = Counter(avg_val)
                print(avg_summary)

                sqdist_rmse = np.sqrt(sum(map(lambda x: x ** 2, sqdist_val)) / len(sqdist_val))
                print("Squared Dist RSME:", sqdist_rmse)
                sqdist_summary = Counter(sqdist_val)
                print(sqdist_summary)

                combined_rmse = np.sqrt(sum(map(lambda x: x ** 2, combined_val)) / len(combined_val))
                print("Combined Features RSME:", combined_rmse)
                combined_summary = Counter(combined_val)
                print(combined_summary)
                print()

            torch.save({
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': training_loss,
                'validation_loss': val_loss,
                'rsme': combined_rmse,
                'summary': combined_summary
            }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))
        else:
            torch.save({
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': training_loss,
            }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))
            print()
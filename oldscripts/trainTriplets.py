from collections import Counter
import numpy as np
import os
import pathlib
import sys
import torch
import torchvision.models as models

import Classifier
import RavenDataLoader
import utils

if __name__ == '__main__':
    cwd = pathlib.Path(os.getcwd())

    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    learning_rate = 0.0002

    hyperparams = {"batch_size": 32,
                   "raven_mode": "All",
                   "num_epochs": 120,
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    # data_path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]
    data_path = cwd / "data" / hyperparams["raven_mode"]

    # Create the dataloader for training
    train_loader = RavenDataLoader.TripletsLoader(data_path/"train", hyperparams)

    # Create the dataloader for validation performance (Raven task)
    val_loader = RavenDataLoader.TripletsLoader(data_path/"val", hyperparams)

    valtest_loader = RavenDataLoader.ValidationLoader(data_path/"val", hyperparams)

    # Define Model
    classifier = Classifier.TripletNet()

    # Optimiser
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    # Loss function
    criterion = torch.nn.TripletMarginLoss()
    distanceLoss = torch.nn.PairwiseDistance()

    # Load from checkpoint
    ckpt_path = data_path / "triplet_ckpt"
    ckpt_path.mkdir(exist_ok=True)
    if hyperparams["ckpt"]:
        checkpoint = torch.load(str(ckpt_path / str(hyperparams["ckpt"])))
        classifier.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loading Checkpoint", hyperparams["ckpt"])

    # Move model and optimiser to CUDA if available
    classifier.to(device)
    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    for epoch in range(hyperparams["num_epochs"]):
        # Training Phase
        print("Training Epoch", epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1)
        print("-----------------------------------")
        training_loss = 0
        classifier.train()
        for a, p, n in train_loader:
            optimizer.zero_grad()

            # Move data to device
            a = a.to(device).float()
            p = p.to(device).float()
            n = n.to(device).float()

            a_vec = classifier(a)
            p_vec = classifier(p)
            n_vec = classifier(n)

            # Compute loss and backpropagate gradients
            loss = criterion(a_vec, p_vec, n_vec)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        training_loss = training_loss / len(train_loader)
        print("Training Loss: {0:.10f}".format(training_loss))

        if (epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1) % 5 == 0:
            # Validation Phase
            classifier.eval()
            val_loss = 0.0
            with torch.no_grad():
                for a, p, n in val_loader:
                    # Move data to device
                    a = a.to(device).float()
                    p = p.to(device).float()
                    n = n.to(device).float()

                    a_vec = classifier(a)
                    p_vec = classifier(p)
                    n_vec = classifier(n)

                    # Compute loss and backpropagate gradients
                    loss = criterion(a_vec, p_vec, n_vec)
                    val_loss += loss.item()

                val_loss = val_loss / len(val_loader)
                print("Validation Loss:", val_loss)

                combined_val = []
                for count, (data, label) in enumerate(valtest_loader):
                    data = data.to(device).float()
                    label = label[:, np.newaxis].int()
                    label = label.to(device)

                    # Data is of dimension (batch_size, answer_panel, 2 (first and second set), 6, image_w, image_h)
                    combined_scores = []

                    combined_qns = (classifier.cnn(data[:, 0, 0, :3, :]) + classifier.cnn(data[:, 0, 1, :3, :])) / 2
                    for option in range(data.shape[1]):
                        option_feat = classifier.cnn(data[:, option, 0, 3:, :])

                        combined_scores.append(distanceLoss(combined_qns, option_feat))
                    # Arrange the panels according to their scores
                    combined_ranks = np.argsort(np.array(torch.stack(combined_scores).cpu()).transpose((1, 0)),
                                                axis=1).squeeze()

                    # Get the position of the gold label
                    combined_scores = (torch.tensor(combined_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                    combined_val += combined_scores.tolist()

                # Compute RSME from the label positions
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
                }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))
        else:
            torch.save({
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': training_loss,
            }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))
            print()
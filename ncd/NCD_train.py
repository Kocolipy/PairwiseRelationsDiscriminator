from collections import Counter
import numpy as np
import os
import pathlib
import sys
import torch

import Classifier
import RavenDataLoader

if __name__ == '__main__':
    cwd = pathlib.Path(os.getcwd())

    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    hyperparams = {"batch_size": 32,
                   "raven_mode": "All",
                   "num_epochs": 80,
                   "lr": 0.0002,
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    path = cwd / "data" / hyperparams["raven_mode"]
    # path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]
    data_path = path / "train"
    val_path = path / "val"

    # Create the dataloader for training
    train_loader = RavenDataLoader.NCDLoader(data_path, hyperparams)
    val_loader = RavenDataLoader.NCDValidationLoader(val_path, hyperparams)

    # Define Model
    classifier = Classifier.NCDNet()

    # Optimiser
    optimizer = torch.optim.Adam(classifier.parameters(), lr=hyperparams["lr"])

    # Loss function
    criterion = torch.nn.BCELoss()

    # Load from checkpoint
    ckpt_path = path / "ckpt"
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

        label = torch.tensor((1, 1, 0, 0, 0, 0, 0, 0, 0, 0)).repeat(hyperparams["batch_size"], 1)
        label = label.to(device).float()

        for data in train_loader:
            if data.shape[0] != hyperparams["batch_size"]:
                # To handle the last batch (usually less than batch size)
                label = label[:data.shape[0]]

            optimizer.zero_grad()

            # Move data to device
            data = data.to(device).float()

            output = classifier(data).squeeze()
            # print(output.cpu().tolist())
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        training_loss = training_loss / len(train_loader)
        print("Training Loss: {0:.5f}".format(training_loss))

        # Validation Phase
        classifier.eval()
        val = []
        with torch.no_grad():
            for data, label in val_loader:
                data = data.to(device).float()
                # print(label.tolist())
                label = label[:, np.newaxis].float()
                label = label.to(device)

                output = classifier(data).squeeze()
                # print(output.cpu().tolist())
                ranks = np.argsort(-1 * np.array(output[:, 2:].cpu()), axis=1)
                scores = (torch.tensor(ranks) == (label - 1).cpu()).nonzero()[:, 1]
                # print(scores.tolist())
                val += scores.tolist()

        # Compute RSME from the label positions
        rmse = np.sqrt(sum(map(lambda x: x ** 2, val)) / len(val))
        print("RSME:", rmse)

        # Display the raw positions
        summary = Counter(val)
        print(summary)
        print()

        torch.save({
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss,
            'rsme': rmse,
            'summary': summary
        }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))

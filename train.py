from collections import Counter
import numpy as np
import os
import pathlib
import sys
import torch

import Classifier
import RavenDataLoader

def set_bn_eval(m):
    # Freeze batchnorm layers in model
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


if __name__ == '__main__':
    cwd = pathlib.Path(os.getcwd())
    torch.multiprocessing.freeze_support()

    # Make use of GPU if available
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    hyperparams = {"batch_size": 32,
                   "raven_mode": "All",
                   "num_epochs": 200,
                   "dataset_type": "full", # "test": only the 14,000 test examples;"train": only the 42,000 training examples; "full": the entire 70,000 examples
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    # Set up data directory
    data_path = cwd / "data" / hyperparams["raven_mode"]
    # data_path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]

    # Create the dataloader to load data for training
    train_loader = RavenDataLoader.DiscriminatorDataloader(data_path, hyperparams)

    # Define Model
    classifier = Classifier.DotResNet()

    # Optimiser
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0002)

    # Loss function
    criterion = torch.nn.BCELoss()

    # Load from checkpoint
    ckpt_path = data_path / "dot_ckpt"
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

    for epoch in range(hyperparams["num_epochs"]):
        # Training Phase
        print("Training Epoch", epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1)
        print("-----------------------------------")
        training_loss = 0

        # Set model to training mode and freeze batchnorm layers
        classifier.train()
        classifier.apply(set_bn_eval)

        for real_data, real_label, fake_data, fake_label in train_loader:
            optimizer.zero_grad()

            # Move data to device
            real_data = real_data.to(device).float()
            real_label = real_label[:, np.newaxis].float().to(device)
            fake_data = fake_data.to(device).float()
            fake_label = fake_label[:, np.newaxis].float().to(device)

            # Real Data Batch
            real_output = classifier(real_data)
            real_loss = criterion(real_output, real_label)

            # Fake Data Batch
            fake_output = classifier(fake_data)
            fake_loss = criterion(fake_output, fake_label)

            # Compute loss and backpropagate gradients
            loss = (real_loss + fake_loss)/2
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss = training_loss / len(train_loader)
        print("Training Loss: {0:.5f}".format(training_loss))

        # Once training has proceeded long enough, we check training accuracy and only every 5 epoch
        if (epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1) >= 150 and (epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1) % 5 == 0:
            with torch.no_grad():
                num_correct = 0.0

                # Set model to eval mode
                classifier.eval()
                for real_data, real_label, fake_data, fake_label in train_loader:
                    # Move data to device
                    real_data = real_data.to(device).float()
                    real_label = real_label[:, np.newaxis].float().to(device)
                    fake_data = fake_data.to(device).float()
                    fake_label = fake_label[:, np.newaxis].float().to(device)

                    real_output = classifier(real_data)
                    fake_output = classifier(fake_data)

                    num_correct += sum(real_output > fake_output).item() / real_data.shape[0]

                training_acc = num_correct / len(train_loader)
                print("Training Acc: {0:.5f}".format(training_acc))

            # Save checkpoint
            torch.save({
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': training_loss,
                'training_acc': training_acc,
            }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))

        else:
            # Save checkpoint
            torch.save({
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': training_loss,
            }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))
        print()
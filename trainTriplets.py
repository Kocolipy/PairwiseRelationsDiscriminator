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

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Find the pairwise distance or euclidean distance of two output feature vectors
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean(label* torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def distance(x, y):
    euclidean_distance = torch.nn.functional.pairwise_distance(x, y)
    return torch.pow(euclidean_distance, 2)

if __name__ == '__main__':
    cwd = pathlib.Path(os.getcwd())

    torch.multiprocessing.freeze_support()
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    learning_rate = 0.014

    hyperparams = {"batch_size": 20,
                   "raven_mode": "All",
                   "image_size" : 80,
                   "num_epochs": 80,
                   "lr": 0.001,
                   "wd": 0.0001,
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    # data_path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]
    data_path = cwd / "data" / hyperparams["raven_mode"]

    # Create the dataloader for training
    train_loader = RavenDataLoader.DualRowsRealFakeLoader(data_path, hyperparams)

    # Create the dataloader for validation loss (2 set of 3 tiles [discriminator task])
    valrows_loader = RavenDataLoader.DualRowsLoader(data_path/"val", hyperparams)

    # Create the dataloader for validation performance (Raven task)
    val_loader = RavenDataLoader.ValidationTripletsLoader(data_path/"val", hyperparams)

    # Define Model
    classifier = Classifier.ResNet()
    # classifier = Classifier.CNN()

    # Optimiser
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=hyperparams["wd"])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    # Loss function
    criterion = ContrastiveLoss()
    criterion_val = ContrastiveLoss()
    # criterion =  torch.nn.TripletMarginLoss()
    # criterion_val = torch.nn.TripletMarginLoss()
    # criterion = torch.nn.BCELoss()
    # criterion_val = torch.nn.BCELoss()

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
    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    for epoch in range(hyperparams["num_epochs"]):
        # Training Phase
        print("Training Epoch", epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 1)
        print("-----------------------------------")
        scheduler.step()
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
            output_1 = classifier(real_data[:, 0:3, :])
            output_2 = classifier(real_data[:, 3:6, :])
            real_loss = criterion(output_1, output_2, real_label)
            # print("real", real_output.cpu().tolist())

            # Fake Data Batch
            # fake_data = ae.encode(fake_data)
            output_1 = classifier(fake_data[:, 0:3, :])
            output_2 = classifier(fake_data[:, 3:6, :])
            fake_loss = criterion(output_1, output_2, fake_label)
            # print("fake", fake_output.cpu().tolist())

            # Compute loss and backpropagate gradients
            loss = real_loss + fake_loss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()/2
        training_loss = training_loss / len(train_loader)
        print("Training Loss: {0:.5f}".format(training_loss))

        # Validation Phase
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            val = []
            for count, (first, second, answers, label) in enumerate(val_loader):
                first = first.to(device).float()
                second = second.to(device).float()
                answers = answers.to(device).float()
                label = label[:, np.newaxis].int()
                label = label.to(device)

                # answers is of dimension (batch_size, answer_panels, 3, image_w, image_h)
                output_first = classifier(first)
                output_second = classifier(second)

                # Calculate validation losses
                loss = criterion_val(output_first, output_second, torch.ones([20], dtype=float, device=device))
                val_loss += loss.item()

                output_answers = []
                for option in range(answers.shape[1]):
                    out = classifier(answers[:, option, :])
                    output_answers.append(out)
                score_first = [distance(output_first, answer) for answer in output_answers]
                score_second = [distance(output_second, answer) for answer in output_answers]

                # Change sum to other operations
                scores = [sum(x) for x in zip(score_first, score_second)]

                scores = torch.stack(scores).transpose(1, 0)

                # Arrange the panels according to their scores
                ranks = np.argsort(np.array(scores.cpu()), axis=1)

                # Get the position of the gold label
                scores = (torch.tensor(ranks) == (label - 1).cpu()).nonzero()[:, 1]
                val += scores.tolist()

            val_loss = val_loss / len(val_loader)
            print("Validation Loss:", val_loss)

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
                'validation_loss': val_loss,
                'rsme': rmse,
                'summary': summary
            }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))

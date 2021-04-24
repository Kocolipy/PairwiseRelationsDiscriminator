import argparse
from collections import Counter
import os
import pathlib
import sys

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from prd.Classifier import L1ResNet
from prd.RavenDataLoader import DiscriminatorDataloader


def set_bn_eval(m):
    # Freeze batchnorm layers in model
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--ckpt_dir', type=str, default="ckpt")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dataset_type', type=str, default="center", choices=["center", "2x2", "3x3", "oic", "oig", "lr", "ud"])
    parser.add_argument('--num_train_workers', type=int, default=6)
    parser.add_argument('--num_inference_workers', type=int, default=2)
    args = parser.parse_args()

    cwd = pathlib.Path(os.getcwd())
    torch.multiprocessing.freeze_support()

    # Make use of GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set up data directory
    data_path = cwd / args.data_dir

    # Create the dataloader to load data for training
    train_loader = SpecificConfigurationDataloader(data_path, args)

    # Define Model
    transform = torch.nn.Sequential(
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225, 0.229, 0.224, 0.225))
    )
    classifier = L1ResNet()

    # Optimiser
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0002)

    # Loss function
    criterion = torch.nn.BCELoss()

    # Load from checkpoint
    ckpt_path = data_path / args.ckpt_dir
    ckpt_path.mkdir(exist_ok=True)
    if args.ckpt is not None:
        checkpoint = torch.load(str(ckpt_path / str(args.ckpt)))
        classifier.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loading Checkpoint {args.ckpt}")
        print(f"Training loss: {checkpoint['training_loss']}")

    # Move model and optimiser to CUDA if available
    classifier.to(device)
    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    for epoch in range((1 if args.ckpt is None else args.ckpt + 1), args.epochs + 1):
        # Training Phase
        print(f"Training Epoch {epoch}")
        print("-----------------------------------")
        training_loss = 0

        # Set model to training mode and freeze batchnorm layers
        classifier.train()
        classifier.apply(set_bn_eval)

        for real_data, real_label, fake_data, fake_label in tqdm(train_loader):
            optimizer.zero_grad()

            # Move data to device
            real_data = transform(real_data.to(device).float())
            real_label = real_label[:, np.newaxis].float().to(device)
            fake_data = transform(fake_data.to(device).float())
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
        print(f"Training Loss: {training_loss:.5f}")

        # Save checkpoint
        torch.save({
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss,
        }, str(ckpt_path / str(epoch)))
        print()

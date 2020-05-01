import os
import pathlib

# importing the libraries
import numpy as np

# for reading and displaying images

# for creating validation set

# for evaluating the model
import sys

# PyTorch libraries and modules
import torch
from torch.utils.data import DataLoader
from collections import Counter

import Classifier
import RavenDataLoader

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    hyperparams = {"batch_size": 20,
                   "raven_mode": "3x3Grid",
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    cwd = pathlib.Path(os.getcwd())
    data_path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]
    # data_path = cwd / "data" / hyperparams["raven_mode"]

    ckpt_path = data_path / "ckpt"
    data_source = data_path / "train"
    dataset = RavenDataLoader.RavenDataset(data_source)
    # dataset = RavenDataset.RavenFullDataset(data_source)
    data_loader = DataLoader(dataset, batch_size=hyperparams["batch_size"],
                        shuffle=True, num_workers=hyperparams["batch_size"])

    # Load Autoencoder
    # ae = Autoencoder.ReducedAutoencoder()
    # checkpoint = torch.load(cwd / "data" / hyperparams["raven_mode"] / "ae_ckpt" / "35")
    # # checkpoint = torch.load(r"D:\RAVEN\Fold_1\{0}\ae_ckpt\200".format(hyperparams["raven_mode"]))
    # ae.load_state_dict(checkpoint["model_state_dict"])
    # ae.to(device)


    # Define Model
    # cnn = CNN.FFNN()
    cnn = Classifier.CNN()
    # cnn = CNN.CNN2()
    # cnn = CNN.FullCNN()
    cnn.eval()

    # Load from checkpoint
    if hyperparams["ckpt"]:
        checkpoint = torch.load(str(ckpt_path / str(hyperparams["ckpt"])))
        # checkpoint = torch.load(str(ckpt_path / str(hyperparams["ckpt"])))
        cnn.load_state_dict(checkpoint["model_state_dict"])
        print("Loading Checkpoint", hyperparams["ckpt"])

    # Move model and optimiser to CUDA if available
    cnn.to(device)

    num_min_correct = 0.0
    num_mean_correct = 0.0
    results = []
    val = []
    with torch.no_grad():
        cnn.eval()
        for count, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label[:, np.newaxis].int()
            label = label.to(device)

            # Data is of dimension (batch_size, answer_panel, 2 (first and second set), 6, image_w, image_h)
            min_scores = []
            avg_scores = []
            for option in range(data.shape[1]):
                vals = []
                for sample_set in range(data.shape[2]):
                    # encoded = []
                    # for i in range(data.shape[3]):
                    #     encoded.append(ae.encoder(data[:, option, sample_set, i:i+1, :].float()))
                    # encoded = torch.cat(encoded, 3)
                    # encoded = torch.flatten(encoded, start_dim=1)
                    out = cnn(data[:, option, sample_set, :].float())
                    # print(out.tolist())
                    vals.append(out)

                # Take the min of the two sample sets
                min_scores.append(torch.min(torch.stack(vals), dim=0).values)
                avg_scores.append(torch.mean(torch.stack(vals), dim=0))
            # Take the index of the highest scoring answer panel
            # results.append(min_scores[label[0,0]-1].item())
            ranks = np.argsort(-1 * np.array(torch.stack(min_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
            scores = (torch.tensor(ranks) == (label - 1).cpu()).nonzero()[:, 1]
            val += scores.tolist()
            min_scores = torch.max(torch.stack(min_scores), dim=0).indices + 1
            avg_scores = torch.max(torch.stack(avg_scores), dim=0).indices + 1
            min_correct = [1 if min_scores[j, :] == label[j, :] else 0 for j in range(data.shape[0])]
            avg_correct = [1 if avg_scores[j, :] == label[j, :] else 0 for j in range(data.shape[0])]

            num_min_correct += sum(min_correct)/len(min_correct)
            num_mean_correct += sum(avg_correct)/len(avg_correct)
            print(count, num_min_correct, num_mean_correct, len(data_loader))
        print("Accuracy of Min:", num_min_correct / len(data_loader) * 100)
        print("Accuracy of Mean:", num_mean_correct / len(data_loader) * 100)
        val_loss = np.sqrt(sum(map(lambda x: x ** 2, val)) / len(val))
        print("RSME:", val_loss)
        summary = Counter(val)
        print(summary)
        print()

        # import matplotlib.pyplot as plt
        # plt.hist(val, bins='auto')
        # plt.show()

        # plt.savefig(str(cwd/ "dist"))
    # # Validate for full set
    # num_correct = 0.0
    # with torch.no_grad():
    #     for count, (data, label) in enumerate(data_loader):
    #         data = data.to(device)
    #         label = label[:, np.newaxis].int()
    #         label = label.to(device)
    #
    #         # Data is of dimension (batch_size, 8 (options), 16 (question panel + answer panel), image_w, image_h)
    #         scores = []
    #         for option in range(data.shape[1]):
    #             scores.append(cnn(data[:, option, :]))
    #         # Take the index of the highest scoring answer panel
    #         scores = torch.max(torch.stack(scores), dim=0).indices + 1
    #         correct = [1 if scores[j, :] == label[j, :] else 0 for j in range(data.shape[0])]
    #         num_correct += sum(correct)/len(correct)
    #
    #         print(count, num_correct, len(data_loader))
    #     print("Accuracy:", num_correct/len(data_loader)*100)

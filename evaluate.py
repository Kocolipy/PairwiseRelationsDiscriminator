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

    # Make use of GPU if available
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    hyperparams = {"batch_size": 32,
                   "raven_mode": "All",
                   "num_epochs": 200,
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    # Set up data directory
    data_path = cwd / "data" / hyperparams["raven_mode"]
    # data_path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]

    # Create the dataloader for validation performance (Raven task)
    test_loader = RavenDataLoader.ValidationLoader(data_path/"val", hyperparams)

    # Define Model
    classifier = Classifier.L1ResNet()

    # Load from checkpoint
    # ckpt_path = data_path / "ckpt"
    ckpt_path = data_path / "saved_ckpt" / "L1Full0.75"

    for epoch in range(10):
        ckpt_no = (hyperparams["ckpt"] if hyperparams["ckpt"] else 0) + 5*epoch
        with torch.no_grad():
            # if hyperparams["ckpt"]:
            #     checkpoint = torch.load(str(ckpt_path / str(ckpt_no)))
            #     classifier.load_state_dict(checkpoint["model_state_dict"])
            #     print(checkpoint["training_acc"])
            checkpoint = torch.load(str(ckpt_path))
            classifier.load_state_dict(checkpoint["model_state_dict"])
            classifier.to(device)
            classifier.eval()

            # Training Phase
            print("Evaluating Epoch", ckpt_no)
            print("-----------------------------------")

            # Different Inference Approaches
            min_val = []
            avg_val = []
            sqdist_val = []
            combined_val = []
            for count, (data, label) in enumerate(test_loader):
                # Data is of dimension (batch_size, answer_panel, 2 (first and second set), 6, image_w, image_h)
                data = data.to(device).float()
                label = label[:, np.newaxis].int()
                label = label.to(device)

                min_scores = []
                avg_scores = []
                sqdist_scores = []
                combined_scores = []

                # Combine the features of the first two sets together to create a combined features
                combined_qns = (classifier.cnn(data[:, 0, 0, :3, :]) + classifier.cnn(data[:, 0, 1, :3, :]))/2
                for option in range(data.shape[1]):
                    vals = []
                    option_feat = classifier.cnn(data[:, option, 0, 3:, :])

                    # Combine the combined features and the candidate answer feature
                    features = combined_qns.add(-1 * option_feat)
                    features = torch.abs(features)
                    # features = torch.cat([combined_qns, option_feat], 1)
                    combined_scores.append(classifier.mlp(features))

                    for sample_set in range(data.shape[2]):
                        out = classifier(data[:, option, sample_set, :])
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
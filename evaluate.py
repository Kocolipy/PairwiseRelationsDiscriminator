from collections import Counter
import numpy as np
import os
import pathlib
import torch

import Classifier
import RavenDataLoader

if __name__ == '__main__':
    cwd = pathlib.Path(os.getcwd())
    torch.multiprocessing.freeze_support()

    # Make use of GPU if available
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    hyperparams = {"batch_size": 32}

    # Set up data directory
    data_path = cwd / "data"

    # Define Model
    classifier = Classifier.L1ResNet()

    # Load model
    model_path = data_path / "ckpt" / "modelA"

    mean_acc = [0.0, 0.0, 0.0, 0.0, 0.0]
    # Evaluate based on configurations
    for config in ["center", "2x2", "3x3", "lr", "ud", "oic", "oig"]:

        # Create the dataloader for evaluation performance (Raven task)
        test_loader = RavenDataLoader.InferenceLoader(data_path/"test", hyperparams, config)

        with torch.no_grad():
            checkpoint = torch.load(str(model_path))
            classifier.load_state_dict(checkpoint["model_state_dict"])
            classifier.to(device)
            classifier.eval()

            # Training Phase
            print("Evaluating", config)
            print("-----------------------------------")

            # Different Inference Approaches
            min_val = []
            max_val = []
            avg_val = []
            sqdist_val = []
            combined_val = []
            for count, (data, label) in enumerate(test_loader):
                # Data is of dimension (batch_size, answer_panel, 2 (first and second set), 6, image_w, image_h)
                data = data.to(device).float()
                label = label[:, np.newaxis].int()
                label = label.to(device)

                min_scores = []
                max_scores = []
                avg_scores = []
                sqdist_scores = []
                combined_scores = []

                # Combine the features of the first two sets together to create a combined features
                combined_qns = (classifier.cnn(data[:, 0, 0, :3, :]) + classifier.cnn(data[:, 0, 1, :3, :]))/2
                for option in range(data.shape[1]):
                    vals = []
                    option_feat = classifier.cnn(data[:, option, 0, 3:, :])

                    # Combine the combined features and the candidate answer feature

                    # # Dot prod
                    # combined_qns = torch.nn.functional.normalize(combined_qns, p=2, dim=1)
                    # option_feat = torch.nn.functional.normalize(option_feat, p=2, dim=1)
                    # dot = (combined_qns * option_feat).sum(1)
                    # features = (torch.abs(dot).clamp(0, 1)).unsqueeze(-1)
                    # combined_scores.append(features)

                    features = combined_qns.add(-1 * option_feat)
                    features = torch.abs(features)

                    # # L2 distance
                    # features = torch.pow(features, 2)

                    # # Concatenation
                    # features = torch.cat([combined_qns, option_feat], 1)

                    combined_scores.append(classifier.mlp(features))

                    for sample_set in range(data.shape[2]):
                        out = classifier(data[:, option, sample_set, :])
                        vals.append(out)

                    # Take the score of the two sample sets
                    min_scores.append(torch.min(torch.stack(vals), dim=0).values)
                    max_scores.append(torch.max(torch.stack(vals), dim=0).values)
                    avg_scores.append(torch.mean(torch.stack(vals), dim=0))
                    sqdist_scores.append(torch.mean((1 - torch.stack(vals))**2, dim=0)*2)

                # Arrange the panels according to their scores
                min_ranks = np.argsort(-1 * np.array(torch.stack(min_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                max_ranks = np.argsort(-1 * np.array(torch.stack(max_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                avg_ranks = np.argsort(-1 * np.array(torch.stack(avg_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                sqdist_ranks = np.argsort(np.array(torch.stack(sqdist_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                combined_ranks = np.argsort(-1 * np.array(torch.stack(combined_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()

                # Get the position of the gold label
                min_scores = (torch.tensor(min_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                max_scores = (torch.tensor(max_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                avg_scores = (torch.tensor(avg_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                sqdist_scores = (torch.tensor(sqdist_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                combined_scores = (torch.tensor(combined_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                min_val += min_scores.tolist()
                max_val += max_scores.tolist()
                avg_val += avg_scores.tolist()
                sqdist_val += sqdist_scores.tolist()
                combined_val += combined_scores.tolist()

            # Compute accuracy from the label positions
            min_summary = Counter(min_val)
            print("Min:", min_summary)
            print("Accuracy:", min_summary[0] / 2000.0)
            mean_acc[0] += min_summary[0] / 14000.0
            print()

            max_summary = Counter(max_val)
            print("Max:", max_summary)
            print("Accuracy:", max_summary[0] / 2000.0)
            mean_acc[1] += max_summary[0] / 14000.0
            print()

            avg_summary = Counter(avg_val)
            print("Avg:", avg_summary)
            print("Accuracy:", avg_summary[0] / 2000.0)
            mean_acc[2] += avg_summary[0] / 14000.0
            print()

            sqdist_summary = Counter(sqdist_val)
            print("Squared Dist:", sqdist_summary)
            print("Accuracy:", sqdist_summary[0] / 2000.0)
            mean_acc[3] += sqdist_summary[0] / 14000.0
            print()

            combined_summary = Counter(combined_val)
            print("Combined Features:", combined_summary)
            print("Accuracy:", combined_summary[0] / 2000.0)
            mean_acc[4] += combined_summary[0] / 14000.0
            print()
            print()

    print("------- Overall Performance -------")
    print("Min:", mean_acc[0])
    print("Max:", mean_acc[1])
    print("Avg:", mean_acc[2])
    print("Squared Dist:", mean_acc[3])
    print("Combined Features:", mean_acc[4])

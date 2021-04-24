import argparse
from collections import Counter, defaultdict
import random
import os
import pathlib

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from prd.Classifier import L1ResNet
from prd.RavenDataLoader import InferenceLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--ckpt_dir', type=str, default="ckpt")
    parser.add_argument('--start_ckpt', type=int, default=150)
    parser.add_argument('--end_ckpt', type=int, default=200)
    parser.add_argument('--num_inference_workers', type=int, default=2)
    args = parser.parse_args()

    cwd = pathlib.Path(os.getcwd())
    torch.multiprocessing.freeze_support()
    
    # Make use of GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set up data directory
    data_path = cwd / args.data_dir

    # Define Model
    transform = torch.nn.Sequential(
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225, 0.229, 0.224, 0.225))
    )
    classifier = L1ResNet()

    # Load model
    model_path = data_path / args.ckpt_dir
    picks = random.choices(range(args.start_ckpt, args.end_ckpt+1), k = 2)

    scores = defaultdict(list)

    for p in picks:
        model_fullpath = model_path / str(p)

        with torch.no_grad():
            checkpoint = torch.load(str(model_fullpath))
            classifier.load_state_dict(checkpoint["model_state_dict"])
            classifier.to(device)
            classifier.eval()
            
            print(f"Model {p} loaded")
        
            # Evaluate based on configurations
            for config in ["center", "2x2", "3x3", "lr", "ud", "oic", "oig"]:
                # Create the dataloader for evaluation performance (Raven task)
                test_loader = InferenceLoader(data_path/"test", args, config)

                # Training Phase
                print(f"Evaluating {config} ...")
                avg_val = []
                for count, (data, label) in enumerate(tqdm(test_loader)):
                    # Data is of dimension (batch_size, answer_panel, 2 (first and second set), 6, image_w, image_h)
                    shape = data.shape
                    data = transform(data.float().reshape(-1, *shape[3:]))
                    data = data.reshape(*shape[0:4], 224, 224).to(device)
                    label = label[:, np.newaxis].int().to(device)

                    avg_scores = []
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

                        # # L1 distance
                        features = combined_qns.add(-1 * option_feat)
                        features = torch.abs(features)

                        # # L2 distance
                        # features = combined_qns.add(-1 * option_feat)
                        # features = torch.pow(features, 2)

                        # # Concatenation
                        # features = torch.cat([combined_qns, option_feat], 1)

                        # combined_scores.append(classifier.mlp(features))

                        for sample_set in range(data.shape[2]):
                            out = classifier(data[:, option, sample_set, :])
                            vals.append(out)

                        # Take the score of the two sample sets
                        # min_scores.append(torch.min(torch.stack(vals), dim=0).values)
                        # max_scores.append(torch.max(torch.stack(vals), dim=0).values)
                        avg_scores.append(torch.mean(torch.stack(vals), dim=0))
                        # sqdist_scores.append(torch.mean((1 - torch.stack(vals))**2, dim=0)*2)

                    # Arrange the panels according to their scores
                    # min_ranks = np.argsort(-1 * np.array(torch.stack(min_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                    # max_ranks = np.argsort(-1 * np.array(torch.stack(max_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                    avg_ranks = np.argsort(-1 * np.array(torch.stack(avg_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                    # sqdist_ranks = np.argsort(np.array(torch.stack(sqdist_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()
                    # combined_ranks = np.argsort(-1 * np.array(torch.stack(combined_scores).cpu()).transpose((1, 0, 2)), axis=1).squeeze()

                    # Get the position of the gold label
                    # min_scores = (torch.tensor(min_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                    # max_scores = (torch.tensor(max_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                    avg_scores = (torch.tensor(avg_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                    # sqdist_scores = (torch.tensor(sqdist_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                    # combined_scores = (torch.tensor(combined_ranks) == (label - 1).cpu()).nonzero()[:, 1]
                    
                    # min_val += min_scores.tolist()
                    # max_val += max_scores.tolist()
                    avg_val += avg_scores.tolist()
                    # sqdist_val += sqdist_scores.tolist()
                    # combined_val += combined_scores.tolist()

                # Compute accuracy from the label positions
                # min_summary = Counter(min_val)
                # scores[config].append(min_summary[0]/ len(min_summary))

                # max_summary = Counter(max_val)
                # scores[config].append(max_summary[0]/ len(max_summary))

                avg_summary = Counter(avg_val)
                scores[config].append(avg_summary[0]/ len(avg_val))

                # sqdist_summary = Counter(sqdist_val)
                # scores[config].append(sqdist_summary[0]/ len(avg_val))

                # combined_summary = Counter(combined_val)
                # scores[config].append(combined_summary[0]/ len(avg_val))
            print()

    print("------- Overall Performance -------")
    for k,v in scores.items():
        print(f"Config {k}: Accuracy of {np.mean(np.array(v))}")
    print(f"Overall Accuracy: {np.mean(np.array(list(scores.values())))}")
    

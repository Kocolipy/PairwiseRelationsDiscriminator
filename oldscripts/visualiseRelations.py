import torch

import os
import pathlib
import pickle
import numpy as np
from skimage.transform import resize
from collections import defaultdict
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import Classifier

if __name__ == '__main__':
    cwd = pathlib.Path(os.getcwd())

    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    hyperparams = {"batch_size": 5,
                   "raven_mode": "All"}

    data_path = pathlib.Path(r"D:\RAVEN\Fold_1") / hyperparams["raven_mode"]
    #  data_path = cwd / "data" / hyperparams["raven_mode"]

    data_source = pathlib.Path(r"D:\RAVEN\Fold_1") / "Center" / "train"

    # Load ckpt
    ckpt_file = data_path / "ckpt" / "90"
    classifier = Classifier.ResNet()
    checkpoint = torch.load(ckpt_file)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.to(device)

    files = list(os.listdir(str(data_source)))[:1000]
    data = []
    # df = pd.DataFrame(columns=['Color', "Number/Position", "Size", "Type"])
    df = pd.DataFrame()

    rules = defaultdict()
    with torch.no_grad():
        for count, idx in enumerate(files):
            if count % 100 == 0:
                print("Progress", count)
            with open(str(data_source/ idx), "rb") as f:
                file = pickle.load(f)

            first = torch.tensor(resize(file.questionPanels[:3], (3, 224, 224)))
            second = torch.tensor(resize(file.questionPanels[3:6], (3, 224, 224)))
            third = torch.tensor(resize(np.concatenate([file.questionPanels[6:], file.answerPanels[file.answer - 1: file.answer]], 0), (3, 224, 224)))
            input = torch.stack((first, second, third))

            input = input.to(device).float()

            rule = file.rules.rules[0].rules
            row = [rule["Color"], rule["Number/Position"], rule["Size"], rule["Type"]]
            # rule = json.dumps(sorted(file.rules.rules[0].rules.items()))
            # hash = hashlib.md5(rule.encode()).hexdigest()
            # rules[hash] = rule

            data.append(classifier.cnn(input).cpu().numpy())
            df = pd.concat([df,  pd.DataFrame([row, row, row], columns=['Color', "Number/Position", "Size", "Type"])], ignore_index=True)

    print("Performing TSNE")
    data = np.concatenate(data)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(data)
    tsne_data = pd.DataFrame(data=tsne, columns=["X", "Y"])
    df = pd.concat([tsne_data, df], axis=1)
    g = sns.scatterplot(
        x="X", y= "Y",
        hue="Color",
        palette=sns.color_palette("hls", 4),
        data=df,
        legend="full",
        alpha=0.3
    )
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # Put a legend to the right side
    g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    for k,v in rules.items():
        print(k,v)
    plt.show()

    g = sns.scatterplot(
        x="X", y="Y",
        hue="Number/Position",
        palette=sns.color_palette("hls", 1),
        data=df,
        legend="full",
        alpha=0.3
    )
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # Put a legend to the right side
    g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    for k, v in rules.items():
        print(k, v)
    plt.show()

    g = sns.scatterplot(
        x="X", y="Y",
        hue="Size",
        palette=sns.color_palette("hls", 4),
        data=df,
        legend="full",
        alpha=0.3
    )
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # Put a legend to the right side
    g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    for k, v in rules.items():
        print(k, v)
    plt.show()

    g = sns.scatterplot(
        x="X", y="Y",
        hue="Type",
        palette=sns.color_palette("hls", 3),
        data=df,
        legend="full",
        alpha=0.3
    )
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # Put a legend to the right side
    g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    for k, v in rules.items():
        print(k, v)
    plt.show()
    print()

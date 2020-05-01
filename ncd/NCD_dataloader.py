from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import pickle
import random
from skimage.transform import resize
import torchvision.transforms as transforms

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])


class NCDDataset(Dataset):
    """"
    Return NCD Data
    """

    def __init__(self, data_source,):
        self.data_source = data_source
        self.len = len(os.listdir(str(self.data_source)))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # with open(str(self.data_source / str(idx)), "rb") as f:
        #     file = pickle.load(f)
        #
        # matrix = [file[0][0:3], file[0][3:6]]
        # half3 = file[0][6:]
        #
        # for j in range(file[1].shape[0]):
        #     matrix.append(np.concatenate((half3, file[1][j:j+1,:]), 0))
        # matrix = resize(np.stack(matrix), (10, 3, 224, 224))
        #
        # return torch.tensor(matrix)

        with open(str(self.data_source / str(idx)), "rb") as f:
            f1 = pickle.load(f)

        a = random.choice(range(self.len))
        with open(str(self.data_source / str(a)), "rb") as f:
            f2 = pickle.load(f)

        matrix = [f1.questionPanels[0:3], f1.questionPanels[3:6]]
        half3 = f1.questionPanels[6:]

        random_int = random.sample(range(8), 3)
        for j in range(f1.answerPanels.shape[0]):
            panel = f2.answerPanels[j:j+1] if j in random_int else f1.answerPanels[j:j+1]
            matrix.append(np.concatenate((half3, panel), 0))
        matrix = resize(np.stack(matrix), (10, 3, 224, 224))

        return torch.tensor(matrix)


def NCDLoader(data_path, hyperparams):
    data_set = NCDDataset(data_path)
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                      shuffle=True, num_workers=hyperparams["batch_size"])

class NCDValidationDataset(Dataset):
    def __init__(self, data_source):
        self.data_source = data_source

    def __len__(self):
        return len(os.listdir(str(self.data_source)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(str(self.data_source / str(idx)), "rb") as f:
            file = pickle.load(f)

        matrix = [file.questionPanels[:3], file.questionPanels[3:6]]
        half3 = file.questionPanels[6:]
        for j in range(file.answerPanels.shape[0]):
            matrix.append(np.concatenate((half3, file.answerPanels[j:j+1, :]), 0))
        matrix = resize(np.stack(matrix), (10, 3, 224, 224))

        return torch.tensor(matrix), file.answer

def NCDValidationLoader(data_path, hyperparams):
    data_set = NCDValidationDataset(data_path)
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                      shuffle=True, num_workers=hyperparams["batch_size"])
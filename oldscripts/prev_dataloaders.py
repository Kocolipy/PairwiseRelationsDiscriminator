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

class SampleDataset(Dataset):
    """"
    Return a dataset containing both the question and answer panels (16, 160, 160)
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __len__(self):
        return len(os.listdir(str(self.data_source)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(str(self.data_source / str(idx)), "rb") as f:
            data = pickle.load(f)

        # return torch.tensor(resize(np.concatenate((data.questionPanels, data.answerPanels), axis=0), (16, 28, 28)))
        return torch.tensor(np.concatenate((data.questionPanels, data.answerPanels), axis=0)) / 255.

def SampleLoader(data_path, hyperparams):
    data_set = SampleDataset(data_path)
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                        shuffle=True, num_workers=hyperparams["batch_size"])


class DualRowsDataset(Dataset):
    """"
    Return one dataset containing only two rows of panels (6, 160, 160)
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __len__(self):
        return len(os.listdir(str(self.data_source)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(str(self.data_source / str(idx)), "rb") as f:
            file = pickle.load(f)

        data = torch.cat((norm(torch.tensor(resize(file.questionPanels[:3], (3, 224, 224)))),
                          norm(torch.tensor(resize(file.questionPanels[3:6], (3, 224, 224))))), 0)

        # data = torch.tensor(resize(file.questionPanels[:6], (6, 224, 224)))

        return data, torch.tensor(1.0, dtype=float)


def DualRowsLoader(data_path, hyperparams):

    data_set = DualRowsDataset(data_path)
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                        shuffle=True, num_workers=hyperparams["batch_size"])


class DualRowsRealFakeDataset(Dataset):
    """"
    Return two dataset (real, fake) containing only two rows of panels (6, 160, 160)
    """

    def __init__(self, real_data_source, fake_data_source):
        self.real_data_source = real_data_source
        self.fake_data_source = fake_data_source

    def __len__(self):
        return len(os.listdir(str(self.real_data_source)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        with open(str(self.real_data_source / str(idx)), "rb") as f:
            real = pickle.load(f)

        real_data = torch.cat((norm(torch.tensor(resize(real.first, (3, 224, 224)))),
                               norm(torch.tensor(resize(real.second, (3, 224, 224))))), 0)

        # real_data = torch.tensor(resize(np.concatenate((real.first, real.second), axis=0), (6, 224, 224)))

        with open(str(self.fake_data_source / str(idx)), "rb") as f:
            fake = pickle.load(f)

        fake_data = torch.cat((norm(torch.tensor(resize(fake.first, (3, 224, 224)))),
                               norm(torch.tensor(resize(fake.second, (3, 224, 224))))), 0)

        # fake_data = torch.tensor(resize(np.concatenate((fake.first, fake.second), axis=0), (6, 224, 224)))
        return real_data, real.label, fake_data, fake.label


def DualRowsRealFakeLoader(data_path, hyperparams):
    data_set = DualRowsRealFakeDataset(data_path / "real", data_path / "fake")
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                        shuffle=True, num_workers=hyperparams["batch_size"])


class TripletsDataset(Dataset):
        """"
        Return a triplet (a, p, n)
        """

        def __init__(self, data_source):
            self.data_source = data_source

        def __len__(self):
            return len(os.listdir(str(self.data_source)))

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            with open(str(self.data_source / str(idx)), "rb") as f:
                file = pickle.load(f)

            i = 3 if random.random() > 0.5 else 0
            j = abs(i - 3)
            a = norm(torch.tensor(resize(file.questionPanels[i:i + 3], (3, 224, 224))))
            p = norm(torch.tensor(resize(file.questionPanels[j:j + 3], (3, 224, 224))))
            if random.random() > 0.5:
                n = np.concatenate(
                    [file.questionPanels[j:j + 2], random.choice(file.answerPanels)[np.newaxis, :]])

            else:
                n = np.concatenate(
                    [file.questionPanels[6:8], random.choice(file.questionPanels[j:j + 3])[np.newaxis, :]])

            return a, p, n


def TripletsLoader(data_path, hyperparams):
    data_set = TripletsDataset(data_path)
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                      shuffle=True, num_workers=hyperparams["batch_size"])

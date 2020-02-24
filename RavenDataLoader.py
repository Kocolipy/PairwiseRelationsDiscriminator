from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import pickle
from skimage.transform import resize


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

        return torch.tensor(resize(file.questionPanels[:6], (6, 80, 80))), torch.tensor(1.0, dtype=float)


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
        real_data = torch.tensor(resize(np.concatenate((real.first, real.second), axis=0), (6, 80, 80)))

        with open(str(self.fake_data_source / str(idx)), "rb") as f:
            fake = pickle.load(f)
        fake_data = torch.tensor(resize(np.concatenate((fake.first, fake.second), axis=0), (6, 80, 80)))

        return real_data, real.label, fake_data, fake.label


def DualRowsRealFakeLoader(data_path, hyperparams):
    data_set = DualRowsRealFakeDataset(data_path / "real", data_path / "fake")
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                        shuffle=True, num_workers=hyperparams["batch_size"])


class ValidationDataset(Dataset):
        """"
        Return a dataset of sample arranged in validation format (#answers, 2(for each row), 6, 160, 160)
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
            firstSet = np.repeat(file.questionPanels[:3][np.newaxis, :], 8, 0)
            secondSet = np.repeat(file.questionPanels[3:6][np.newaxis, :], 8, 0)
            thirdSet = file.questionPanels[6:]
            answers = np.array([np.concatenate((thirdSet, ans[np.newaxis, :])) for ans in file.answerPanels])
            data = np.concatenate((np.concatenate((firstSet, answers), 1)[:, np.newaxis, :], np.concatenate((secondSet, answers), 1)[:, np.newaxis, :]), 1)
            data = torch.tensor(resize(data, (8, 2, 6, 80, 80)))

            return data, file.answer


def ValidationLoader(data_path, hyperparams):
    data_set = ValidationDataset(data_path)
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                        shuffle=True, num_workers=hyperparams["batch_size"])


class ValidationTripletsDataset(Dataset):
        """"
        Return both sets and eight answer sets
        """

        def __init__(self, data_source, size):
            self.data_source = data_source
            self.size = size

        def __len__(self):
            return len(os.listdir(str(self.data_source)))

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            with open(str(self.data_source / str(idx)), "rb") as f:
                file = pickle.load(f)
            firstSet = file.questionPanels[:3]
            secondSet = file.questionPanels[3:6]
            thirdSet = file.questionPanels[6:]
            answers = np.array([np.concatenate((thirdSet, ans[np.newaxis, :])) for ans in file.answerPanels])

            firstSet = torch.tensor(resize(firstSet, (3, self.size, self.size)))
            secondSet = torch.tensor(resize(secondSet, (3, self.size, self.size)))
            answers = torch.tensor(resize(answers, (8, 3, self.size, self.size)))

            return firstSet, secondSet, answers, file.answer


def ValidationTripletsLoader(data_path, hyperparams):
    data_set = ValidationTripletsDataset(data_path, hyperparams["image_size"])
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                      shuffle=True, num_workers=hyperparams["batch_size"])



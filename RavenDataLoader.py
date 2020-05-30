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

class ConfigurationDataset(Dataset):
    """"
    Return a dataset of real, fake pair of two sets of 3 panels.

    Real: The first and second rows of the sample (order is shuffled)

    Fake (Alternate Relations):
    The first and second row is selected at random as Set A.
    Set B is generated randomly with 75% option 1, 25% option 2
     - Option 1: constructed from either the unselected row with a random candidate answer
                               or   the third row with a random cell from the unselected row
                 The resulting row is shuffled.
     - Option 2: a random row taken from a random sample
    """
    def __init__(self, data_source, type):
        self.data_source = data_source
        self.offset = ["center", "2x2", "3x3", "oic", "oig", "lr", "ud"].index(type)
        self.len = 10000

    def __len__(self):
        return self.len

    def getFileName(self, idx):
        if idx < 6000:
            return self.data_source / "train" / str(idx)
        elif idx < 8000:
            return self.data_source / "val" / str(idx - 6000)
        else:
            return self.data_source / "test" / str(idx - 8000)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = pickle.load(open(str(self.getFileName(idx)), "rb"))

        i = 3 if random.random() > 0.5 else 0
        j = abs(i-3)
        firstSet = norm(torch.tensor(resize(file.questionPanels[i:i+3], (3, 224, 224))))
        secondSet = norm(torch.tensor(resize(file.questionPanels[j:j+3], (3, 224, 224))))
        real_data = torch.cat([firstSet, secondSet], 0)

        # Both Option 1 and 2
        if random.random() < 1/2:
            if random.random() < 0.5:
                thirdSet = np.concatenate([file.questionPanels[j:j + 2], random.choice(file.answerPanels)[np.newaxis, :]])
            else:
                thirdSet = np.concatenate([file.questionPanels[6:8], random.choice(file.questionPanels[j:j+3])[np.newaxis, :]])
            np.random.shuffle(thirdSet)
        else:
            a = random.choice(range(self.len))
            f2 = pickle.load(open(str(self.getFileName(a)), "rb"))

            k = 3 if random.random() > 0.5 else 0
            thirdSet = f2.questionPanels[k:k + 3]

        thirdSet = norm(torch.tensor(resize(thirdSet, (3, 224, 224))))
        fake_data = torch.cat([firstSet, thirdSet], 0)

        return real_data, torch.tensor(1.0, dtype=float), fake_data, torch.tensor(0.0, dtype=float)

def SpecificConfigurationDataloader(data_path, hyperparams):
    data_set = ConfigurationDataset(data_path, hyperparams["dataset_type"])
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                      shuffle=True, num_workers=hyperparams["batch_size"])


class DiscriminatorDataset(Dataset):
    """"
    Return a dataset of real, fake pair of two sets of 3 panels.

    Real: The first and second rows of the sample (order is shuffled)

    Fake (Alternate Relations):
    The first and second row is selected at random as Set A.
    Set B is generated randomly with 75% option 1, 25% option 2
     - Option 1: constructed from either the unselected row with a random candidate answer
                               or   the third row with a random cell from the unselected row
                 The resulting row is shuffled.
     - Option 2: a random row taken from a random sample
    """
    def __init__(self, data_source, type):
        self.data_source = data_source
        self.type = type

        if type == "full":
            self.len = 70000
        elif type == "train":
            self.len = 42000
        elif type == "test":
            self.len = 14000
        else:
            print("Type must be either full, train or test")

    def __len__(self):
        return self.len

    def getFileName(self, idx):
        if self.type == "test":
            return self.data_source / "test" / str(idx)

        if idx < 42000:
            return self.data_source / "train" / str(idx)
        elif idx < 56000:
            return self.data_source / "val" / str(idx - 42000)
        else:
            return self.data_source / "test" / str(idx - 56000)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(str(self.getFileName(idx)), "rb") as f:
            file = pickle.load(f)

        i = 3 if random.random() > 0.5 else 0
        j = abs(i-3)
        firstSet = norm(torch.tensor(resize(file.questionPanels[i:i+3], (3, 224, 224))))
        secondSet = norm(torch.tensor(resize(file.questionPanels[j:j+3], (3, 224, 224))))
        real_data = torch.cat([firstSet, secondSet], 0)

        # # Only Option 1
        # if random.random() < 0.5:
        #     thirdSet = np.concatenate([file.questionPanels[j:j + 2], random.choice(file.answerPanels)[np.newaxis, :]])
        # else:
        #     thirdSet = np.concatenate([file.questionPanels[6:8], random.choice(file.questionPanels[j:j + 3])[np.newaxis, :]])
        # np.random.shuffle(thirdSet)

        # # Only Option 2
        # a = random.choice(range(self.len))
        # with open(str(self.data_source / str(a)), "rb") as f:
        #     f2 = pickle.load(f)
        #
        # k = 3 if random.random() > 0.5 else 0
        # thirdSet = f2.questionPanels[k:k + 3]

        # Both Option 1 and 2
        if random.random() < 1/2:
            if random.random() < 0.5:
                thirdSet = np.concatenate([file.questionPanels[j:j + 2], random.choice(file.answerPanels)[np.newaxis, :]])
            else:
                thirdSet = np.concatenate([file.questionPanels[6:8], random.choice(file.questionPanels[j:j+3])[np.newaxis, :]])
            np.random.shuffle(thirdSet)
        else:
            a = random.choice(range(self.len))
            with open(str(self.getFileName(a)), "rb") as f:
                f2 = pickle.load(f)

            k = 3 if random.random() > 0.5 else 0
            thirdSet = f2.questionPanels[k:k + 3]

        thirdSet = norm(torch.tensor(resize(thirdSet, (3, 224, 224))))
        fake_data = torch.cat([firstSet, thirdSet], 0)

        return real_data, torch.tensor(1.0, dtype=float), fake_data, torch.tensor(0.0, dtype=float)


def DiscriminatorDataloader(data_path, hyperparams):
    data_set = DiscriminatorDataset(data_path, hyperparams["dataset_type"])
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                      shuffle=True, num_workers=hyperparams["batch_size"])


class InferenceDataset(Dataset):
    """"
    Return a dataset of sample arranged in validation format (#answers, 2(for each row), 6, 160, 160)
    """
    def __init__(self, data_source, config):
        self.data_source = data_source
        self.offset = None
        if not config:
            self.len = len(os.listdir(str(self.data_source)))
        else:
            self.len = 2000
            self.offset = ["center", "2x2", "3x3", "oic", "oig", "lr", "ud"].index(config)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.offset:
            idx = self.offset*2000 + idx

        with open(str(self.data_source / str(idx)), "rb") as f:
            file = pickle.load(f)

        firstSet = norm(torch.tensor(resize(file.questionPanels[:3], (3, 224, 224)))).unsqueeze(0).unsqueeze(0).repeat(8,1,1,1,1)
        secondSet = norm(torch.tensor(resize(file.questionPanels[3:6], (3, 224, 224)))).unsqueeze(0).unsqueeze(0).repeat(8,1,1,1,1)
        thirdSet = file.questionPanels[6:]

        answers = torch.stack([norm(torch.tensor(resize(np.concatenate((thirdSet, ans[np.newaxis, :])), (3, 224, 224)))) for ans in file.answerPanels], 0).unsqueeze(1)
        data = torch.cat((torch.cat((firstSet, answers), 2), torch.cat((secondSet, answers), 2)), 1)

        return data, file.answer


def InferenceLoader(data_path, hyperparams, config=None):
    data_set = InferenceDataset(data_path, config)
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                        shuffle=True, num_workers=2)

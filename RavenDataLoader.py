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

class RandomRealFakeDataset(Dataset):
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
        j = abs(i-3)
        firstSet = norm(torch.tensor(resize(file.questionPanels[i:i+3], (3, 224, 224))))
        secondSet = norm(torch.tensor(resize(file.questionPanels[j:j+3], (3, 224, 224))))
        real_data = torch.cat([firstSet, secondSet], 0)

        if random.random() > 0.5:
            thirdSet = np.concatenate([file.questionPanels[j:j + 2], random.choice(file.answerPanels)[np.newaxis, :]])
            thirdSet = norm(torch.tensor(resize(thirdSet, (3, 224, 224))))
            fake_data = torch.cat([firstSet, thirdSet], 0)
        else:
            thirdSet = np.concatenate([file.questionPanels[6:8], random.choice(file.questionPanels[j:j+3])[np.newaxis, :]])
            thirdSet = norm(torch.tensor(resize(thirdSet, (3, 224, 224))))
            fake_data = torch.cat([firstSet, thirdSet], 0)

        return real_data, torch.tensor(1.0, dtype=float), fake_data, torch.tensor(0.0, dtype=float)

def RandomRealFakeLoader(data_path, hyperparams):
    data_set = RandomRealFakeDataset(data_path)
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

            # firstSet = np.repeat(file.questionPanels[:3][np.newaxis, :], 8, 0)
            # secondSet = np.repeat(file.questionPanels[3:6][np.newaxis, :], 8, 0)
            # thirdSet = file.questionPanels[6:]
            # answers = np.array([np.concatenate((thirdSet, ans[np.newaxis, :])) for ans in file.answerPanels])
            # data = np.concatenate((np.concatenate((firstSet, answers), 1)[:, np.newaxis, :], np.concatenate((secondSet, answers), 1)[:, np.newaxis, :]), 1)
            # data = torch.tensor(resize(data, (8, 2, 6, 224, 224)))

            firstSet = norm(torch.tensor(resize(file.questionPanels[:3], (3, 224, 224)))).unsqueeze(0).unsqueeze(0).repeat(8,1,1,1,1)
            secondSet = norm(torch.tensor(resize(file.questionPanels[3:6], (3, 224, 224)))).unsqueeze(0).unsqueeze(0).repeat(8,1,1,1,1)
            thirdSet = file.questionPanels[6:]

            answers = torch.stack([norm(torch.tensor(resize(np.concatenate((thirdSet, ans[np.newaxis, :])), (3, 224, 224)))) for ans in file.answerPanels], 0).unsqueeze(1)
            data = torch.cat((torch.cat((firstSet, answers), 2), torch.cat((secondSet, answers), 2)), 1)

            return data, file.answer


def ValidationLoader(data_path, hyperparams):
    data_set = ValidationDataset(data_path)
    return DataLoader(data_set, batch_size=hyperparams["batch_size"],
                        shuffle=True, num_workers=8)


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
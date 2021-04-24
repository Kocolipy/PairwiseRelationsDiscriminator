import numpy as np
import random

class DualSets:
    def __init__(self, s1, s2, label):
        self.first = s1
        self.second = s2
        self.label = label

    @classmethod
    def real(cls, s):
        return cls(s.questionPanels[0:3], s.questionPanels[3:6], 1.0)

    @classmethod
    def realwnoise(cls, s):
        return cls(s.questionPanels[0:3], s.questionPanels[3:6], np.random.uniform(0.85, 1.0))

    @classmethod
    def fake5(cls, s1, s2):
        choice = [0,3]
        random.shuffle(choice)
        return cls(s1.questionPanels[choice[0]:choice[0]+3], np.concatenate((s1.questionPanels[choice[1]:choice[1]+2], s2.questionPanels[choice[1]+2:choice[1]+3]),axis=0), 0.0)

    @classmethod
    def fake(cls, s1, s2):
        i = random.randrange(0, 4, 3)
        j = random.randrange(0, 4, 3)
        return cls(s1.questionPanels[i:i+3], s2.questionPanels[j:j+3], 0.0)

    @classmethod
    def fake3(cls, s1, s2, s3):
        choice = [0,3]
        random.shuffle(choice)
        answer = list(range(8))
        random.shuffle(answer)
        return cls(s1.questionPanels[choice[0]:choice[0]+3],
                    np.concatenate((s2.questionPanels[choice[1]:choice[1]+2], s3.questionPanels[answer[0]:answer[0]+1]),axis=0),
                    0.0)
    @classmethod
    def fake1wnoise(cls, s1):
        choice = [0,3]
        random.shuffle(choice)
        answer = list(range(8))
        random.shuffle(answer)
        return cls(s1.questionPanels[choice[0]:choice[0]+3],
                    np.concatenate((s1.questionPanels[choice[1]:choice[1]+2], s1.answerPanels[answer[0]:answer[0]+1]),axis=0),
                    np.random.uniform(0.0, 0.15))

    @classmethod
    def fake1(cls, s1):
        choice = [0,3]
        random.shuffle(choice)
        answer = list(range(8))
        random.shuffle(answer)
        return cls(s1.questionPanels[choice[0]:choice[0]+3],
                    np.concatenate((s1.questionPanels[choice[1]:choice[1]+2], s1.answerPanels[answer[0]:answer[0]+1]),axis=0),
                    0.0)

    def display(self):
        import matplotlib.pyplot as plt
        fig, ims = plt.subplots(3, 3)

        # plot
        for i in range(3):
            image = self.first[i].squeeze()
            ims[0][i].imshow(image, cmap='gray')
            plt.setp(ims[0][i].get_yticklabels(), visible=False)
            plt.setp(ims[0][i].get_xticklabels(), visible=False)
            ims[0][i].tick_params(axis='both', which='both', length=0)

        for i in range(3):
            ims[1][i].axis('off')

        for i in range(3):
            image = self.second[i].squeeze()
            ims[2][i].imshow(image, cmap='gray')
            plt.setp(ims[2][i].get_yticklabels(), visible=False)
            plt.setp(ims[2][i].get_xticklabels(), visible=False)
            ims[2][i].tick_params(axis='both', which='both', length=0)
        plt.show()
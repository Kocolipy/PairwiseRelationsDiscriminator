import matplotlib.pyplot as plt
import torch


def loadAutoEncoder(AutoEncoderClass, ckpt, device):
    ae = AutoEncoderClass()
    checkpoint = torch.load(ckpt)
    ae.load_state_dict(checkpoint["model_state_dict"])
    ae.eval()
    ae.to(device)
    return ae


def displaySample(tensor):
    """"
    Display the entire sample (16, W, H)
    """
    fig, ims = plt.subplots(6, 4)

    # plot
    for i in range(3):
        for j in range(3):
            if 3 * i + j >= 8:
                ims[i][j].axis('off')
                break
            image = tensor[3 * i + j, :].squeeze()
            ims[i][j].imshow(image, cmap='gray')
            plt.setp(ims[i][j].get_yticklabels(), visible=False)
            plt.setp(ims[i][j].get_xticklabels(), visible=False)
            ims[i][j].tick_params(axis='both', which='both', length=0)

        ims[i][3].axis('off')
    for i in range(4):
        ims[3][i].axis('off')

    for i in range(8):
        image = tensor[8 + i, :].squeeze()
        ims[4 + i // 4][i % 4].imshow(image, cmap='gray')
        ims[4 + i // 4][i % 4].set_ylabel(str(i + 1), rotation=0)

        plt.setp(ims[4 + i // 4][i % 4].get_yticklabels(), visible=False)
        plt.setp(ims[4 + i // 4][i % 4].get_xticklabels(), visible=False)
        ims[4 + i // 4][i % 4].tick_params(axis='both', which='both', length=0)
    plt.show()


def displayDualSet(imagetensor):
    """"
    Display the Dual Set (6, W, H)
    """
    fig, ims = plt.subplots(2, 3)

    # plot
    for i in range(2):
        for j in range(3):
            image = imagetensor[3 * i + j, :].squeeze()
            ims[i][j].imshow(image, cmap='gray')
            plt.setp(ims[i][j].get_yticklabels(), visible=False)
            plt.setp(ims[i][j].get_xticklabels(), visible=False)
            ims[i][j].tick_params(axis='both', which='both', length=0)

    plt.show()



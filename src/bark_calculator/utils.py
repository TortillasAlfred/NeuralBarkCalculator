from dataset import RegressionDatasetFolder

from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, SubsetRandomSampler

from math import ceil
import numpy as np
import torch
from torch import nn


def get_train_valid_samplers(dataset, train_percent, seed=42):
    np.random.seed(seed)
    n_items = len(dataset)

    all_idx = np.arange(n_items)

    np.random.shuffle(all_idx)

    n_train = ceil(n_items * train_percent)
    n_valid = n_items - n_train

    train_idx = all_idx[:n_train]
    valid_idx = all_idx[-n_valid:]

    return SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)


def compute_mean_std(working_dir: str):
    train_dataset = RegressionDatasetFolder(working_dir,
                                            transform=Compose(
                                                [Resize((224, 224)), ToTensor()]))
    loader = DataLoader(train_dataset, batch_size=100)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean.tolist(), std.tolist()


def get_mean_std():
    # Util function to not have to recalculate them
    # every single time
    # mean = [0.5495320558547974, 0.46154847741127014, 0.34539610147476196]
    # std = [0.35342904925346375, 0.3120446503162384,
    # 0.25366029143333435]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return mean, std


def get_pos_weight():
    return torch.FloatTensor([1./0.34019524])


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
        self.__name__ = "dice_loss"

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / \
            (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class MixedLoss(nn.Module):

    def __init__(self):
        super(MixedLoss, self).__init__()
        self.dice = SoftDiceLoss()
        self.bce = nn.modules.loss.BCEWithLogitsLoss(
            pos_weight=get_pos_weight())

    def forward(self, predict, true):
        return self.dice(predict, true) + self.bce(predict, true)

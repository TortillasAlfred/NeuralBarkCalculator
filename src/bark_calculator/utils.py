from dataset import RegressionDatasetFolder

from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from torchvision.transforms.functional import pad, resize
from torch.utils.data import DataLoader, SubsetRandomSampler

from math import ceil, floor, sin, cos
import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import rotate, center_crop
import random


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


def rotate_crop(image, angle_range=25):
    angle = random.random() * angle_range * 2 - angle_range

    size, _ = image.size

    image = rotate(image, angle)

    cropped_size = rotatedRectWithMaxArea(size, (angle + 180)/(180*np.pi))

    return center_crop(image, cropped_size)


def rotatedRectWithMaxArea(size, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    sin_a, cos_a = abs(sin(angle)), abs(cos(angle))
    if 1 <= 2.*sin_a*cos_a or abs(sin_a-cos_a) < 1e-10:
        x = 0.5*size
        wr, hr = (x/sin_a, x/cos_a)
    else:
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr, hr = (size*cos_a - size*sin_a)/cos_2a, \
            (size*cos_a - size*sin_a)/cos_2a

    return floor(wr/2), floor(hr/2)


def compute_mean_std(working_dir: str):
    train_dataset = RegressionDatasetFolder(working_dir,
                                            transform=Compose(
                                                [Resize((256, 256)), ToTensor()]))
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
    mean = [0.773, 0.650, 0.487]
    std = [0.112, 0.146, 0.161]

    return mean, std


def get_pos_weight():
    return torch.FloatTensor([1./0.42667 - 1.])


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
        self.mse = WeightedMSELoss()

    def forward(self, predict, true):
        return self.dice(predict, true) + self.bce(predict, true)


class WeightedMSELoss(nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.__name__ = "weighted_mse"
        self.pos_weight = get_pos_weight()

    def forward(self, predict, true):
        logits = torch.sigmoid(predict)
        l2 = (logits - true) ** 2
        weights = torch.ones_like(true)
        weights[true > 0.5] = self.pos_weight
        return torch.mean(weights * l2)


TO_PIL = ToPILImage()
TO_TENSOR = ToTensor()


def pad_resize(image, width, height):
    image = pad(image,
                (ceil((width - image.width)/2),
                 ceil((height - image.height)/2)),
                padding_mode='reflect')

    return resize(image, (height, width))


def pad_to_biggest_image(tensor_data):
    images = [[TO_PIL(t_i) for t_i in t] for t in tensor_data]
    width = max([img[0].width for img in images])
    height = max([img[0].height for img in images])

    def resizer(img): return pad_resize(img, width, height)

    for i, (sample, target) in enumerate(images):
        sample = resizer(sample)
        target = resizer(target)
        tensor_data[i] = (TO_TENSOR(sample), TO_TENSOR(target))

    return torch.stack([t[0] for t in tensor_data]), \
        torch.stack([t[1] for t in tensor_data])

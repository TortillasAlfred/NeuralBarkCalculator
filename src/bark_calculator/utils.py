from sklearn.metrics import f1_score
from dataset import RegressionDatasetFolder

from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from torchvision.transforms.functional import pad, resize
from torch.utils.data import DataLoader, SubsetRandomSampler
from skimage.morphology import remove_small_objects, remove_small_holes
from poutyne.framework.callbacks import Callback

from math import ceil, floor, sin, cos
import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import rotate, center_crop
from PIL import Image
import torch.nn.functional as F
import random


def compute_mean_std(working_dataset):
    loader = DataLoader(working_dataset, batch_size=1)

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
    mean = [0.7399, 0.6139, 0.4401]
    std = [0.1068, 0.1272, 0.1271]

    return mean, std


def compute_pos_weight(working_dataset):
    loader = DataLoader(working_dataset, batch_size=1)

    class_counts = [0, 0, 0]

    for _, targets in loader:
        targets = targets.flatten()

        for y in range(3):
            class_counts[y] += (targets == y).sum().item()

    total_samples = sum(class_counts)

    class_weights = [0, 0, 0]

    for y in range(3):
        class_weights[y] = total_samples / (3 * class_counts[y])

    return torch.tensor(class_weights)


def get_pos_weight():
    return torch.FloatTensor([0.4040, 1.9471, 87.6780])


def get_splits(dataset):
    train_percent = 0.7
    valid_percent = 0.2
    test_percent = 0.1

    total_items = len(dataset)

    wood_type_to_idx = {'epinette_gelee': 0, 'epinette_non_gelee': 1, 'sapin': 2}

    idxs_by_type = [[] for _ in range(3)]

    for i, (_, _, _, wood_type) in enumerate(dataset):
        idxs_by_type[wood_type_to_idx[wood_type]].append(i)

    train_split, valid_split, test_split, train_weights = [], [], [], []

    for idx in range(len(idxs_by_type)):
        idxs_by_type[idx] = np.asarray(idxs_by_type[idx])
        np.random.shuffle(idxs_by_type[idx])
        n_data = len(idxs_by_type[idx])

        idx_weight = total_items / (3 * n_data)

        n_train = int(ceil(train_percent * n_data))
        n_valid = int(floor(valid_percent * n_data))

        train_split.extend(idxs_by_type[idx][:n_train])
        valid_split.extend(idxs_by_type[idx][n_train:n_train + n_valid])
        test_split.extend(idxs_by_type[idx][n_train + n_valid:])
        train_weights.extend([idx_weight] * n_train)

    train_split = np.asarray(train_split)
    valid_split = np.asarray(valid_split)
    test_split = np.asarray(test_split)
    train_weights = np.asarray(train_weights)

    return train_split, valid_split, test_split, train_weights


def remove_small_zones(img):
    device = img.device
    np_image = (img.cpu().numpy() == 0)

    remove_small_holes(np_image, area_threshold=100, connectivity=2, in_place=True)
    remove_small_objects(np_image, min_size=100, connectivity=2, in_place=True)

    img[torch.from_numpy(np_image == 0).to(device).byte() & (img == 0)] = 1
    img[torch.from_numpy(np_image != 0).to(device).byte() & (img != 0)] = 0

    return img


class CustomWeightedCrossEntropy(nn.Module):
    def __init__(self, weights):
        super(CustomWeightedCrossEntropy, self).__init__()
        self.__name__ = "CustomWeightedCrossEntropy"
        self.weights = weights

    def forward(self, predict, true):
        entropies = F.cross_entropy(predict, true, reduction='none')

        max_classes = torch.max(torch.argmax(predict, dim=1), true).flatten()

        class_weights = torch.index_select(self.weights, 0, max_classes).view(true.shape)

        return (entropies * class_weights).mean()


def make_training_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IOU(nn.Module):
    def __init__(self, class_to_watch):
        super().__init__()
        self.class_to_watch = class_to_watch

        if self.class_to_watch is None:
            self.__name__ = "IntersectionOverUnion"
        else:
            self.__name__ = "IntersectionOverUnion_class_{}".format(self.class_to_watch)

    def forward(self, outputs, labels):
        outputs = torch.argmax(outputs, 1)

        outputs = remove_small_zones(outputs)

        outputs = outputs.cpu().reshape(-1)
        labels = labels.cpu().reshape(-1)

        scores = f1_score(labels, outputs, labels=[0, 1, 2], average=None)

        targets_count = np.bincount(labels)

        for i, count_i in enumerate(targets_count):
            if count_i == 0:
                scores[i] = np.delete(scores, i).mean()

        if self.class_to_watch is None:
            return scores.mean()
        elif self.class_to_watch == 'loss':
            return 1 - scores.mean()
        else:
            return scores[self.class_to_watch]


TO_PIL = ToPILImage()
TO_TENSOR = ToTensor()


def pad_resize(image, width, height):
    image = pad(image, (ceil((width - image.width) / 2), ceil((height - image.height) / 2)), padding_mode='reflect')

    return resize(image, (height, width))


def pad_to_biggest_image(tensor_data):
    images = [[TO_PIL(t_i) for t_i in t] for t in tensor_data]
    width = max([img[0].width for img in images])
    height = max([img[0].height for img in images])

    def resizer(img):
        return pad_resize(img, width, height)

    for i, (sample, target) in enumerate(images):
        sample = resizer(sample)
        target = resizer(target)
        tensor_data[i] = (TO_TENSOR(sample), TO_TENSOR(target))

    return torch.stack([t[0] for t in tensor_data]), \
        torch.stack([t[1] for t in tensor_data])

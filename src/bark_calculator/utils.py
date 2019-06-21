from kornia.losses import FocalLoss
from sklearn.metrics import f1_score
from dataset import RegressionDatasetFolder

from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from torchvision.transforms.functional import pad, resize
from torch.utils.data import DataLoader, SubsetRandomSampler
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import remove_small_objects
from poutyne.framework.callbacks import Callback

from math import ceil, floor, sin, cos
import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import rotate, center_crop
from PIL import Image
import torch.nn.functional as F
import random


def get_train_valid_samplers(dataset, train_percent):
    n_items = len(dataset)

    all_idx = np.arange(n_items)

    np.random.shuffle(all_idx)

    n_train = ceil(n_items * train_percent)
    n_valid = n_items - n_train

    train_idx = all_idx[:n_train]
    valid_idx = all_idx[-n_valid:]

    return SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)


def rotate_crop(image, angle_range=20):
    angle = random.random() * angle_range * 2 - angle_range

    size, _ = image.size

    image = rotate(image, angle)

    cropped_size = rotatedRectWithMaxArea(size, (angle + 180) / (180 * np.pi))

    return center_crop(image, cropped_size)


def rotatedRectWithMaxArea(size, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    sin_a, cos_a = abs(sin(angle)), abs(cos(angle))
    if 1 <= 2. * sin_a * cos_a or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * size
        wr, hr = (x / sin_a, x / cos_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (size*cos_a - size*sin_a)/cos_2a, \
            (size*cos_a - size*sin_a)/cos_2a

    return floor(wr / 2), floor(hr / 2)


def elastic_transform(image):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    image = np.array(image)

    image_deformed = elasticdeform.deform_random_grid(image, sigma=10, axis=(0, 1))

    return Image.fromarray(image_deformed)


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
    mean = [0.778, 0.658, 0.502]
    std = [0.095, 0.120, 0.130]

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
    return torch.FloatTensor([0.4138, 1.7491, 84.8554])


def get_splits(dataset):
    train_percent = 0.7
    valid_percent = 0.3
    test_percent = 0
    n_data = len(dataset)

    idxs = np.arange(n_data)
    np.random.shuffle(idxs)

    n_train = int(ceil(train_percent * n_data))
    n_valid = int(floor(valid_percent * n_data))

    return idxs[:n_train], \
        idxs[n_train:n_train + n_valid], \
        idxs[n_train + n_valid:]


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
        self.__name__ = "dice_loss"

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = torch.argmax(logits, dim=1)
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
        self.__name__ = "MixedLoss"
        self.dice = SoftDiceLoss()
        self.bce = nn.modules.loss.CrossEntropyLoss(weight=get_pos_weight())

    def forward(self, predict, true):
        return self.bce(predict, true) + self.dice(predict, true)


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


def remove_class_wise(img, class_idx, shape):
    replace_idx = 1 if class_idx == 0 else 1

    binary_mapping = [0 if i != class_idx else 1 for i in range(3)]
    binary_mapping = torch.tensor(binary_mapping).to(img.device)

    binary_img = torch.index_select(binary_mapping, 0, img.flatten()).reshape(shape)

    masks = torch.stack(list(map(remove_from_img, binary_img)))

    img[(masks == 0) & (img == class_idx)] = replace_idx

    return img


def remove_from_img(img_i):
    img_i = remove_small_objects(img_i.cpu().numpy().astype(bool), min_size=12, connectivity=2)

    return torch.from_numpy(img_i).long()


def remove_small_zones(img):
    shape = img.shape

    img = img.cpu()

    for class_idx in [2, 1]:
        img = remove_class_wise(img, class_idx, shape)

    return img


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


class FocalLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = FocalLoss(alpha=0.25, gamma=torch.tensor(2.0).to('cuda:0'), reduction='mean')
        self.__name__ = "FocalLoss"

    def forward(self, outputs, labels):
        return self.criterion(outputs, labels)


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


class ResetLR(Callback):
    def __init__(self, init_lr):
        super(ResetLR, self).__init__()
        self.init_lr = init_lr

    def on_train_begin(self, logs):
        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] = self.init_lr

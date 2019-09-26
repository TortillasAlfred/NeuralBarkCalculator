from sklearn.metrics import f1_score
from dataset import RegressionDatasetFolder
from lovasz_losses import LovaszSoftmax

from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Lambda
from torchvision.transforms.functional import pad, resize
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.sampler import Sampler, BatchSampler, WeightedRandomSampler
from torch.nn import CrossEntropyLoss
from skimage.morphology import remove_small_objects, remove_small_holes
from poutyne.framework.callbacks import Callback

from math import ceil, floor, sin, cos
import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import rotate, center_crop, adjust_brightness, adjust_contrast, adjust_saturation
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
    mean = [0.7358871102333069, 0.6107082366943359, 0.4379482865333557]
    std = [0.11245555430650711, 0.12965525686740875, 0.12750858068466187]

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
    return torch.FloatTensor([0.4004, 2.0334, 93.1921])


def get_splits(dataset):
    train_percent = 0.7
    valid_percent = 0.2
    test_percent = 0.1

    total_items = len(dataset)

    wood_type_to_idx = {
        'epinette_gelee': 0,
        'epinette_non_gelee': 1,
        'sapin': 2
    }

    idxs_by_type = [[] for _ in range(3)]
    sample_weight = []

    for i, (_, target, _, wood_type) in enumerate(dataset):
        idxs_by_type[wood_type_to_idx[wood_type]].append(i)
        sample_weight.append(
            (target.numel() - target.view(-1).bincount()[0]).float().item())

    sample_weight = torch.tensor(sample_weight)
    sample_weight = sample_weight / sample_weight.sum()

    train_split, valid_split, test_split, train_weights = [], [], [], []
    wood_type_weights = []

    for idx in range(len(idxs_by_type)):
        idxs_by_type[idx] = np.asarray(idxs_by_type[idx])
        np.random.shuffle(idxs_by_type[idx])
        n_data = len(idxs_by_type[idx])

        wood_type_weights.append(total_items / (3 * n_data))

        n_train = int(ceil(train_percent * n_data))
        n_valid = int(floor(valid_percent * n_data))

        train_split.extend(idxs_by_type[idx][:n_train])
        valid_split.extend(idxs_by_type[idx][n_train:n_train + n_valid])
        test_split.extend(idxs_by_type[idx][n_train + n_valid:])

    wood_type_weights = np.asarray(wood_type_weights)
    wood_type_weights /= wood_type_weights.sum()
    train_weights = torch.zeros(len(dataset)).float()

    for i, (_, _, _, wood_type) in enumerate(dataset):
        train_weights[i] = wood_type_weights[
            wood_type_to_idx[wood_type]] * sample_weight[i]

    train_split = np.asarray(train_split)
    valid_split = np.asarray(valid_split)
    test_split = np.asarray(test_split)
    train_weights = np.exp(np.asarray(train_weights))
    train_weights = train_weights[train_split]
    train_weights /= train_weights.sum()

    return train_split, valid_split, test_split, train_weights


def remove_small_zones(img):
    device = img.device
    np_image = (img.cpu().numpy() == 0)

    remove_small_holes(np_image,
                       area_threshold=150,
                       connectivity=2,
                       in_place=True)
    remove_small_objects(np_image, min_size=150, connectivity=2, in_place=True)

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

        class_weights = torch.index_select(self.weights, 0,
                                           max_classes).view(true.shape)

        return (entropies * class_weights).mean()


class JaccardLoss(nn.Module):
    def forward(self, predict, true, eps=1e-7, num_classes=3):
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

        predict = F.softmax(predict, dim=1)
        true_1_hot = true_1_hot.type(predict.type())
        dims = (0, ) + tuple(range(2, true.ndimension()))

        intersection = torch.sum(predict * true_1_hot, dims)
        cardinality = torch.sum(predict + true_1_hot, dims)
        union = cardinality - intersection

        jacc_loss = (intersection / (union + eps)).mean()
        return (1 - jacc_loss)


class MixedLoss(nn.Module):
    def __init__(self, cwe_weights):
        super(MixedLoss, self).__init__()
        self.ce = CustomWeightedCrossEntropy(cwe_weights)
        self.lovasz = LovaszSoftmax()

    def forward(self, predict, true):
        return self.ce(predict, true) / 4 + self.lovasz(predict, true)


def make_training_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class PixelWiseF1(nn.Module):
    def __init__(self, class_to_watch):
        super().__init__()
        self.class_to_watch = class_to_watch

        if self.class_to_watch is None:
            self.__name__ = "PixelWiseF1"
        else:
            self.__name__ = "PixelWiseF1_class_{}".format(self.class_to_watch)

    def forward(self, outputs, labels):
        outputs = torch.argmax(outputs, 1)

        outputs = remove_small_zones(outputs)

        outputs = outputs.cpu().reshape(-1)
        labels = labels.cpu().reshape(-1)

        scores = f1_score(labels, outputs, labels=[0, 1, 2], average=None)

        targets_count = np.bincount(labels, minlength=3)
        outputs_count = np.bincount(outputs, minlength=3)

        for i, count_i in enumerate(targets_count):
            if count_i == 0 and outputs_count[i] == 0:
                scores[i] = np.delete(scores, i).mean()

        if self.class_to_watch is None:
            return scores.mean()
        elif self.class_to_watch == 'loss':
            return 1 - scores.mean()
        elif isinstance(self.class_to_watch, int):
            return scores[self.class_to_watch]
        else:
            return scores


TO_PIL = ToPILImage()
TO_TENSOR = ToTensor()


def pad_resize(image, width, height):
    image = pad(image, (ceil(
        (width - image.width) / 2), ceil((height - image.height) / 2)),
                padding_mode='reflect')

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


class NormColorJitter(object):
    """Copy-pasta from PyTorch except that random values are sampled according to a normal instead of a uniform distribution.

        Args:
            brightness (float): How much to jitter brightness.
                brightness_factor is chosen form a normal distribution centered in 1 with
                variance of the input float.
            contrast (float): How much to jitter contrast.
                contrast_factor is chosen from a normal distribution centered in 1 with
                variance of the input float.
            saturation (float or tuple of float (min, max)): How much to jitter saturation.
                saturation_factor is chosen from a normal distribution centered in 1 with
                variance of the input float.
        """

    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')

    def _check_input(self, value, name, center=1):
        if value < 0:
            raise ValueError(
                "If {} is a single number, it must be non negative.".format(
                    name))
        value = [1, value]

        # if value is 0 for brightness/contrast/saturation
        if value[1] == 0:
            value = None

        return value

    @staticmethod
    def get_params(brightness, contrast, saturation):
        """Get a randomized transform to be applied on image.

            Arguments are same as that of __init__.

            Returns:
                Transform which randomly adjusts brightness, contrast and
                saturation in a random order.
            """
        transforms = []

        if brightness is not None:
            brightness_factor = random.gauss(brightness[0], brightness[1])
            transforms.append(
                Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.gauss(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.gauss(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        print(brightness_factor)

        return transform

    def __call__(self, img):
        """
            Args:
                img (PIL Image): Input image.

            Returns:
                PIL Image: Color jittered image.
            """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        return format_string


class PrioritizedBatchSampler(BatchSampler):
    def __init__(self,
                 num_samples,
                 num_items,
                 batch_size,
                 drop_last,
                 update_callback,
                 replacement=True):
        self.num_items = num_items
        weighted_sampler = WeightedRandomSampler(torch.ones(num_items),
                                                 num_samples, replacement)
        self.sampler = BatchSampler(weighted_sampler, batch_size, drop_last)
        self.update_callback = update_callback
        self.update_callback.connect_sampler(self)

    def __iter__(self):
        for batch_idxs in self.sampler:
            self.update_callback.collect_batch(batch_idxs)

            yield batch_idxs

    def __len__(self):
        return len(self.sampler)


class PrioritizedBatchSamplerUpdate(Callback):
    def __init__(self, metric, metric_mode):
        super().__init__()
        if metric_mode not in ['min', 'max']:
            raise AttributeError("metric_mode has to be either 'min' or 'max'")
        self.metric = metric
        self.metric_mode = metric_mode
        self.sampler = None
        self.num_items = None
        self.running_batch_idxs = None
        self.num_visited = None
        self.weights = None

    def on_train_begin(self, logs):
        self.num_visited = torch.zeros(self.num_items, dtype=torch.float64)

    def connect_sampler(self, sampler):
        self.sampler = sampler
        self.num_items = sampler.num_items
        self.weights = sampler.sampler.sampler.weights

    def collect_batch(self, batch_idxs):
        self.running_batch_idxs = batch_idxs

    def on_batch_end(self, batch, logs):
        self.num_visited[self.running_batch_idxs] += 1
        n_visits = self.num_visited[self.running_batch_idxs]
        weights = self.weights[self.running_batch_idxs]
        metric_value = logs[self.metric]
        if self.metric_mode == 'min':
            metric_value = 1 - metric_value
        metric_value = torch.ones_like(n_visits) * metric_value
        self.weights[self.running_batch_idxs] = weights * (
            n_visits - 1) / n_visits + metric_value / n_visits

    def on_train_end(self, logs):
        print("\n*** Prioritized sampler stats ***")
        most_visits, most_visited_idx = [
            i.item() for i in self.num_visited.max(0)
        ]
        most_visited_final_weight = self.weights[most_visited_idx]
        print(
            "Most visited image was visited {} times. It was example number {} and its final weight was {}."
            .format(most_visits, most_visited_idx, most_visited_final_weight))

        least_visits, least_visited_idx = [
            i.item() for i in self.num_visited.min(0)
        ]
        least_visited_final_weight = self.weights[least_visited_idx]
        print(
            "Least visited image was visited {} times. It was example number {} and its final weight was {}."
            .format(least_visits, least_visited_idx,
                    least_visited_final_weight))

        avg_visits = self.num_visited.mean().item()
        print(
            "The average number of visits was of {} times.".format(avg_visits))

        biggest_weight, biggest_weight_idx = [
            i.item() for i in self.weights.max(0)
        ]
        biggest_weight_n_vists = self.num_visited[biggest_weight_idx]
        print(
            "The biggest final weight was {}. It was example number {} and it was visited {} times."
            .format(biggest_weight, biggest_weight_idx,
                    biggest_weight_n_vists))

        smallest_weight, smallest_weight_idx = [
            i.item() for i in self.weights.min(0)
        ]
        smallest_weight_n_visits = self.num_visited[smallest_weight_idx]
        print(
            "The smallest final weight was {}. It was example number {} and it was visited {} times."
            .format(smallest_weight, smallest_weight_idx,
                    smallest_weight_n_visits))

        avg_weight = self.weights.mean().item()
        print("The average weight was {}.".format(avg_weight))

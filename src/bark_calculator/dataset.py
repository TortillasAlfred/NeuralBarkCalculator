"""
Copy-pasta from torchvision.
"""
import torch.utils.data as data
import torch
import torchvision

from PIL import Image

import os
import os.path
import numpy as np
import random
from matplotlib import cm

from skimage.segmentation import find_boundaries


def make_weight_map(masks):
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.

    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)

    """
    w0 = 10
    sigma = 5

    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1))**2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1))**2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1)**2) / (2 * (sigma**2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * \
            np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss
    return ZZ


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_one_hot(target):
    y = torch.eye(3)

    return y[target.long()]


def make_dataset_for_dir(dir, extensions):
    samples_dir = os.path.join(dir, "samples")
    targets_dir = os.path.join(dir, "duals")

    if not os.path.isdir(samples_dir):
        raise IOError("Root folder should have a 'samples' subfolder !")

    if not os.path.isdir(targets_dir):
        raise IOError("Root folder should have a 'targets' subfolder !")

    images = []

    for wood_type in ["epinette_gelee", "epinette_non_gelee", "sapin"]:
        samples_type_dir = os.path.join(samples_dir, wood_type)
        targets_type_dir = os.path.join(targets_dir, wood_type)

        for _, _, fnames in sorted(os.walk(samples_type_dir)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    sample_path = os.path.join(samples_type_dir, fname)
                    fname = fname.replace("bmp", "png")
                    target_path = os.path.join(targets_type_dir, fname)

                    if not os.path.isfile(target_path):
                        # raise IOError("No file found in 'targets' subfolder" " for image name {} !".format(fname))
                        item = (sample_path, "", fname, wood_type)
                    else:
                        item = (sample_path, target_path, fname, wood_type)

                    images.append(item)

    return images


def make_dataset(dir, extensions):
    dir = os.path.expanduser(dir)

    return make_dataset_for_dir(dir, extensions)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path, grayscale=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if not os.path.isfile(path):
        return None

    with open(path, 'rb') as f:
        img = Image.open(f)
        target_format = 'L' if grayscale else 'RGB'
        return img.convert(target_format)


class RegressionDatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/samples/xxx.ext
        root/samples/xxy.ext
        root/samples/xxz.ext

        root/targets/xxx.ext
        root/targets/xxy.ext
        root/targets/xxz.ext

    Args:
        root (string): Root directory path.
        extensions (list[string], optional): A list of allowed extensions.
        loader (callable, optional): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample or target and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        input_only_transform (callable, optional): A function/transform that takes
            in the sample and transforms it. If given, this will always
            be called prior to the transform argument. Since
            'transform' will be applied after, should always return a
            PILImage.
            E.g, ``transforms.Normalize`` for sample images, where
            target is boolean image.
     Attributes:
        samples (list): List of (sample path, target path) tuples
    """

    def __init__(self,
                 root,
                 extensions=IMG_EXTENSIONS,
                 loader=pil_loader,
                 transform=None,
                 input_only_transform=None,
                 include_fname=False):
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.input_only_transform = input_only_transform
        self.include_fname = include_fname

        self.samples = self.put_samples_in_memory(samples)

    def put_samples_in_memory(self, samples):
        ram_samples = []

        for path, target_path, fname, wood_type in samples:
            sample = self.loader(path)
            target = self.loader(target_path, grayscale=True)

            ram_samples.append((sample, target, fname, wood_type))

        return ram_samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) the sample and target images.
        """
        sample, target, fname, wood_type = self.samples[index]

        if self.transform is not None:
            random_seed = np.random.randint(2147483647)

            random.seed(random_seed)
            sample = self.transform(sample)

            if target is not None:
                random.seed(random_seed)
                target = self.transform(target)

        if self.input_only_transform is not None:
            sample = self.input_only_transform(sample)

        if target is not None:
            if target.max() > 200:
                target /= 255

            if sample.max() > 200:
                sample /= 255

            target = target * 2
            target.round_()

            target = target.long().squeeze()
        else:
            target = torch.zeros(sample.shape[1], sample.shape[2])

        if self.include_fname:
            return sample, target, fname, wood_type
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

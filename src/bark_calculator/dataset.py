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


def make_dataset_for_dir(dir, extensions):
    samples_dir = os.path.join(dir, "samples")
    targets_dir = os.path.join(dir, "duals")

    if not os.path.isdir(samples_dir):
        raise IOError("Root folder should have a 'samples' subfolder !")

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
                        item = (sample_path, "", fname, wood_type)
                    else:
                        item = (sample_path, target_path, fname, wood_type)

                    images.append(item)

    return images


def make_dataset(dir, extensions):
    dir = os.path.expanduser(dir)

    return make_dataset_for_dir(dir, extensions)


IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp'
]


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
                 include_fname=False,
                 in_memory=False):
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root +
                                "\n"
                                "Supported extensions are: " +
                                ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.input_only_transform = input_only_transform
        self.include_fname = include_fname
        self.in_memory = in_memory

        self.filenames = samples

        if self.in_memory:
            self.samples = self.put_samples_in_memory(samples)
        else:
            self.samples = samples

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

        if not self.in_memory:
            sample = self.loader(sample)
            target = self.loader(target, grayscale=True)

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

    def print_filenames(self):
        for idx, filename in enumerate(self.filenames):
            print("{}: {}".format(idx, filename[2]))
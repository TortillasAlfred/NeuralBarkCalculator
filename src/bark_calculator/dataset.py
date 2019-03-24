"""
Copy-pasta from torchvision.
"""
import torch.utils.data as data

from PIL import Image

import os
import os.path
import numpy as np
import random


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


def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)

    samples_dir = os.path.join(dir, "samples")
    targets_dir = os.path.join(dir, "targets")

    if not os.path.isdir(samples_dir):
        raise IOError("Root folder should have a 'samples' subfolder !")

    if not os.path.isdir(targets_dir):
        raise IOError("Root folder should have a 'targets' subfolder !")

    for _, _, fnames in sorted(os.walk(samples_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                sample_path = os.path.join(samples_dir, fname)
                target_path = os.path.join(targets_dir, fname)

                if not os.path.isfile(target_path):
                    raise IOError("No file found in 'targets' subfolder"
                                  " for image name {} !".format(fname))

                item = (sample_path, target_path)
                images.append(item)

    return images


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm',
                  '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


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
            be called prior to the transform argument.
            E.g, ``transforms.Normalize`` for sample images, where
            target is boolean image.
     Attributes:
        samples (list): List of (sample path, target path) tuples
    """

    def __init__(self, root, extensions=IMG_EXTENSIONS, loader=default_loader,
                 transform=None, input_only_transform=None):
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.input_only_transform = input_only_transform

        self.samples = samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) the sample and target images.
        """
        path, target_path = self.samples[index]
        sample = self.loader(path)
        target = self.loader(target_path)

        if self.input_only_transform is not None:
            sample = self.input_only_transform(sample)

        if self.transform is not None:
            random_seed = np.random.randint(2147483647)

            random.seed(random_seed)
            sample = self.transform(sample)

            random.seed(random_seed)
            target = self.transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

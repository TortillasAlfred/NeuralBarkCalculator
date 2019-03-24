from dataset import RegressionDatasetFolder
from utils import compute_mean_std, get_mean_std, get_train_valid_samplers
from models import RegressionVGG19_BN

import matplotlib.pyplot as plt
from torchvision.transforms import *

from pytoune.framework import Experiment
from torch.utils.data import DataLoader
import torch


if __name__ == "__main__":
    dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn",
                                      transform=ToTensor())
    mean, std = get_mean_std()
    augmented_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn",
                                                input_only_transform=Compose(
                                                    [Normalize(mean, std),
                                                     ToPILImage(),
                                                     ColorJitter(),
                                                     ToTensor()]
                                                ),
                                                transform=Compose(
                                                    [RandomRotation(180, expand=True),
                                                     RandomResizedCrop(224),
                                                     ToTensor()]
                                                ))

    # for sample, augmented_sample in zip(iter(dataset), iter(augmented_dataset)):
    #     _,  axs = plt.subplots(2, 2)

    #     sample += augmented_sample

    #     for ax, img in zip(axs.flatten(), sample):
    #         ax.imshow(img)
    #         ax.axis('off')

    #     figManager = plt.get_current_fig_manager()
    #     figManager.window.showMaximized()
    #     plt.tight_layout()
    #     plt.show()

    train_sampler, valid_sampler = get_train_valid_samplers(dataset,
                                                            train_percent=0.8)
    train_loader = DataLoader(augmented_dataset, batch_size=8,
                              sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=8,
                              sampler=valid_sampler)
    exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/exp_vgg19_bn/",
                     module=RegressionVGG19_BN(),
                     device=torch.device("cuda:1"),
                     optimizer="adam",
                     type="reg")

    exp.train(train_loader=train_loader, valid_loader=valid_loader)

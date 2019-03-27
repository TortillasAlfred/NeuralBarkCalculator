from dataset import RegressionDatasetFolder, make_weight_map
from utils import *
from models import vanilla_unet

from torchvision.transforms import *

from poutyne.framework import Experiment, ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch

from math import ceil
import numpy as np


if __name__ == "__main__":
    mean, std = get_mean_std()
    dataset = RegressionDatasetFolder("./Images/nn_cut",
                                      input_only_transform=Compose(
                                          [Normalize(mean, std)]
                                      ),
                                      transform=Compose(
                                          [Resize(256),
                                           ToTensor()]
                                      ))
    pure_dataset = RegressionDatasetFolder("./Images/nn_cut",
                                           transform=ToTensor())
    augmented_dataset = RegressionDatasetFolder("./Images/nn_cut",
                                                input_only_transform=Compose(
                                                    []
                                                ),
                                                transform=Compose([
                                                    Lambda(lambda img: Pad((ceil(1025 - img.size[0]/2),
                                                                            ceil(1025 -
                                                                                 img.size[1]/2)),
                                                                           padding_mode='reflect')(img)),
                                                    Lambda(lambda img: rotate_crop(img)),
                                                    RandomResizeCrop(256),
                                                    ToTensor()]))

    # for sample, augmented_sample in zip(iter(pure_dataset), iter(augmented_dataset)):
    #     _,  axs = plt.subplots(2, 2)

    #     sample += augmented_sample

    #     for ax, img in zip(axs.flatten(), sample):
    #         img = ToPILImage()(img)
    #         ax.imshow(img)
    #         ax.axis('off')

    #     plt.tight_layout()
    #     plt.show()

    train_sampler, valid_sampler = get_train_valid_samplers(dataset,
                                                            train_percent=0.6)
    train_loader = DataLoader(augmented_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(dataset, batch_size=4)
    pure_loader = DataLoader(pure_dataset, batch_size=4)
    module = vanilla_unet()
    optim = torch.optim.Adam(module.parameters(), lr=1e-4)
    exp = Experiment(directory="./mix_unet/",
                     module=module,
                     device=torch.device("cpu"),
                     optimizer=optim,
                     loss_function=MixedLoss())

    # exp.load_checkpoint(1784)
    # module = exp.model.model
    # module.to(torch.device("cuda:0"))

    # to_pil = ToPILImage()

    # for batch, pure_batch in zip(valid_loader, pure_loader):
    #     outputs = module(batch[0].to(torch.device("cuda:0")))
    #     torch.sigmoid(outputs)
    #     outputs[outputs > 0.5] = 1
    #     outputs[outputs <= 0.5] = 0
    #     batch.append(outputs.detach().cpu())
    #     batch[0] = pure_batch[0]
    #     tmp = batch[2]
    #     batch[2] = batch[3]
    #     batch[3] = tmp

    #     names = ["Input", "Target", "Generated image"]

    #     for i in range(batch[1].size(0)):
    #         _, axs = plt.subplots(1, 3)

    #         for j, ax in enumerate(axs.flatten()):
    #             img = to_pil(batch[j][i])
    #             ax.imshow(img)
    #             ax.set_title(names[j])
    #             ax.axis('off')

    #         plt.tight_layout()
    #         plt.savefig("Images/results/nn_mix/{}".format(batch[3][i]),
    #                     format="png",
    #                     dpi=900)

    lr_schedulers = [ReduceLROnPlateau(factor=0.5, min_lr=1e-7)]
    exp.train(train_loader=train_loader,
              valid_loader=valid_loader,
              epochs=1000,
              lr_schedulers=lr_schedulers)

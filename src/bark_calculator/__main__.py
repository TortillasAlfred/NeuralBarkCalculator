from dataset import RegressionDatasetFolder, make_weight_map
from utils import *
from models import vanilla_unet, FCDenseNet103, FCDenseNet57, B2B

from torchvision.transforms import *

from poutyne.framework import Experiment, ExponentialLR, EarlyStopping
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch

from math import ceil
import numpy as np


def load_best_and_show(exp, pure_loader, valid_loader):
    exp.load_best_checkpoint()
    module = exp.model.model
    module.to(torch.device("cuda:0"))
    module.eval()

    to_pil = ToPILImage()

    for batch, pure_batch in zip(valid_loader, pure_loader):
        outputs = module(batch[0].to(torch.device("cuda:0")))
        torch.sigmoid(outputs)
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0
        batch.append(outputs.detach().cpu())
        batch[0] = pure_batch[0]
        tmp = batch[2]
        batch[2] = batch[3]
        batch[3] = tmp

        names = ["Input", "Target", "Generated image"]

        for i in range(batch[1].size(0)):
            _, axs = plt.subplots(1, 3)
            acc = (batch[2][i] == batch[1][i]).sum().item()/(256 * 256)

            for j, ax in enumerate(axs.flatten()):
                img = to_pil(batch[j][i])
                ax.imshow(img)
                ax.set_title(names[j])
                ax.axis('off')

            plt.suptitle("Overall accuracy : {:.3f}".format(acc))
            plt.tight_layout()
            plt.show()
            # plt.savefig("Images/results/nn_cut_mix/{}".format(batch[3][i]),
            #             format="png",
            #             dpi=900)


def show_dataset(dataset, pure_dataset, augmented_dataset):
    for sample, augmented_sample in zip(iter(pure_dataset), iter(augmented_dataset)):
        _,  axs = plt.subplots(2, 2)

        sample += augmented_sample

        for ax, img in zip(axs.flatten(), sample):
            img = ToPILImage()(img)
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


def old_main():
    mean, std = get_mean_std()
    dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                      input_only_transform=Compose(
                                          [Normalize(mean, std)]
                                      ),
                                      transform=Compose(
                                          [Lambda(lambda img:
                                                  pad_resize(img, 256, 256)),
                                           ToTensor()]
                                      ))
    pure_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                           transform=Compose(
                                               [Resize((256, 256)), ToTensor()]
                                           ))
    augmented_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                                input_only_transform=Compose(
                                                    [Normalize(mean, std)]
                                                ),
                                                transform=Compose([
                                                    RandomHorizontalFlip(),
                                                    RandomVerticalFlip(),
                                                    Lambda(lambda img:
                                                           pad_resize(img, 256, 256)),
                                                    ToTensor()]))

    # show_dataset(dataset, pure_dataset, augmented_dataset)

    train_sampler, valid_sampler = get_train_valid_samplers(dataset,
                                                            train_percent=0.8)
    train_loader = DataLoader(augmented_dataset, batch_size=8,
                              sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=8,
                              sampler=valid_sampler)
    pure_loader = DataLoader(pure_dataset, batch_size=8,
                             sampler=valid_sampler)
    module = B2B()
    optim = torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=1e-5)
    exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/b2b/",
                     module=module,
                     device=torch.device("cuda:1"),
                     optimizer=optim,
                     loss_function=MixedLoss())

    # load_best_and_show(exp, pure_loader, valid_loader)

    lr_schedulers = [ExponentialLR(gamma=0.98)]
    exp.train(train_loader=train_loader,
              valid_loader=valid_loader,
              epochs=500,
              lr_schedulers=lr_schedulers)


def new_main():
    mean, std = get_mean_std()
    test_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                           input_only_transform=Compose(
                                               [Normalize(mean, std)]
                                           ),
                                           transform=Compose([
                                               Lambda(lambda img:
                                                      pad_resize(img, 256, 256)),
                                               ToTensor()]),
                                           mode="test")

    # for k in range(1, 6):
    #     train_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
    #                                             input_only_transform=Compose(
    #                                                 [Normalize(mean, std)]
    #                                             ),
    #                                             transform=Compose([
    #                                                 RandomHorizontalFlip(),
    #                                                 RandomVerticalFlip(),
    #                                                 Lambda(lambda img:
    #                                                        pad_resize(img, 256, 256)),
    #                                                 ToTensor()]),
    #                                             k=k,
    #                                             mode="train")
    #     valid_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
    #                                             input_only_transform=Compose(
    #                                                 [Normalize(mean, std)]
    #                                             ),
    #                                             transform=Compose([
    #                                                 Lambda(lambda img:
    #                                                        pad_resize(img, 256, 256)),
    #                                                 ToTensor()]),
    #                                             k=k,
    #                                             mode="valid")
    #     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    #     valid_loader = DataLoader(valid_dataset, batch_size=8)
    #     test_loader = DataLoader(test_dataset, batch_size=8)

    #     module = FCDenseNet57(1)
    #     optim = torch.optim.Adam(
    #         module.parameters(), lr=1e-3, weight_decay=1e-5)
    #     exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/b2b/{}/".format(k),
    #                      module=module,
    #                      device=torch.device("cuda:1"),
    #                      optimizer=optim,
    #                      metrics=['mse'],
    #                      loss_function=MixedLoss())

    #     lr_schedulers = [ExponentialLR(gamma=0.90)]
    #     callbacks = [EarlyStopping(patience=12)]
    #     exp.train(train_loader=train_loader,
    #               valid_loader=valid_loader,
    #               epochs=100,
    #               lr_schedulers=lr_schedulers,
    #               callbacks=callbacks)
    #     exp.test(test_loader)

    test_loader = DataLoader(test_dataset, batch_size=1)
    module = B2B("/mnt/storage/mgodbout/Ecorcage/b2b/", 5)
    exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/b2b/",
                     module=module,
                     device=torch.device("cuda:1"),
                     metrics=['mse'],
                     loss_function=MixedLoss())
    exp.test(test_loader)


if __name__ == "__main__":
    new_main()

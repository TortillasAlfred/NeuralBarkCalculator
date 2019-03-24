import matplotlib.pyplot as plt
from dataset import RegressionDatasetFolder
from utils import compute_mean_std, get_mean_std

from torchvision.transforms import *


if __name__ == "__main__":
    dataset = RegressionDatasetFolder("./Images/nn")
    mean, std = get_mean_std()
    augmented_dataset = RegressionDatasetFolder("./Images/nn",
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

    for sample, augmented_sample in zip(iter(dataset), iter(augmented_dataset)):
        _,  axs = plt.subplots(2, 2)

        sample += augmented_sample

        for ax, img in zip(axs.flatten(), sample):
            ax.imshow(img)
            ax.axis('off')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.tight_layout()
        plt.show()

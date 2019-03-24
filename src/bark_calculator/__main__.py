from dataset import RegressionDatasetFolder
from utils import compute_mean_std

import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset = RegressionDatasetFolder("./Images/nn")

    for sample in iter(dataset):
        _,  axs = plt.subplots(1, 2)

        for ax, img in zip(axs, sample):
            ax.imshow(img)
            ax.axis('off')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.tight_layout()
        plt.show()

from skimage.io import show, imshow

from math import ceil

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class Processor:

    def process(self, image_loader, treatment_method):
        for image, image_name in image_loader:
            treated_image = treatment_method.treat_image(image)
            self.processor_handle(treated_image, image_name)


class DisplayProcessor(Processor):

    def processor_handle(self, treated_images, image_name):
        n_images = len(treated_images)
        n_cols = 3
        n_rows = ceil(n_images/3)
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows)

        for idx, image in enumerate(treated_images):
            ax = axes[idx] if n_rows == 1 else axes[idx // n_cols][idx % n_cols]
            ax.imshow(image, cmap=plt.get_cmap('binary'))
            ax.axis('off')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.tight_layout()
        plt.show()

class Saver(Processor):

    def processor_handle(self, treated_images, image_name):
        target_folder = "Images/v1_1/"

        n_images = len(treated_images)
        n_cols = 3
        n_rows = ceil(n_images/3)
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows)

        for idx, image in enumerate(treated_images):
            ax = axes[idx] if n_rows == 1 else axes[idx // n_cols][idx % n_cols]
            ax.imshow(image, cmap=plt.get_cmap('binary'))
            ax.axis('off')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.tight_layout()
        image_name = image_name.replace("/", "_")
        image_name = image_name.replace('\\', "_")
        image_name = image_name.replace(" ", "_")
        image_name = image_name.replace(".bmp", "")
        plt.savefig(target_folder + image_name + ".pdf", format="pdf", dpi=900)

class DataViewing(Processor):

    def processor_handle(self, treated_images, image_name):
        imshow(treated_images[0])
        show()

        for treated_image in treated_images[1:]:
            reshaped = treated_image.reshape((treated_image.shape[0] ** 2, 2))
            x = reshaped[:, 0]
            y = reshaped[:, 1]
            sns.scatterplot(x, y)
    
        plt.show()
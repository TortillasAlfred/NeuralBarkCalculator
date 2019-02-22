from skimage.io import show, imshow

from math import ceil

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class Processor:

    def process(self, image_loader, treatment_method):
        for image in image_loader:
            treated_image = treatment_method.treat_image(image)
            self.processor_handle(treated_image)


class DisplayProcessor(Processor):

    def processor_handle(self, treated_images):
        n_images = len(treated_images)
        fig, axes = plt.subplots(ncols=3, nrows=ceil(n_images/3))

        for idx, image in enumerate(treated_images):
            axes[idx].imshow(image, cmap=plt.gray())
            axes[idx].axis('off')

        fig.tight_layout()
        plt.show()

class DataViewing(Processor):

    def processor_handle(self, treated_images):
        imshow(treated_images[0])
        show()

        for treated_image in treated_images[1:]:
            reshaped = treated_image.reshape((treated_image.shape[0] ** 2, 2))
            x = reshaped[:, 0]
            y = reshaped[:, 1]
            sns.scatterplot(x, y)
    
        plt.show()
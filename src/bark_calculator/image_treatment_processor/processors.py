from skimage.io import show, imshow

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
        imshow(np.concatenate(treated_images, axis=1)) 
        show()

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
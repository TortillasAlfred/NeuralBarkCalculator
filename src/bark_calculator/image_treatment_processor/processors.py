from skimage.io import show, imshow

import matplotlib.pyplot as plt


class Processor:

    def process(self, image_loader, treatment_method):
        for image in image_loader:
            treated_image = treatment_method.treat_image(image)
            self.processor_handle(treated_image)


class DisplayProcessor(Processor):

    def processor_handle(self, treated_images):
        for treated_image in treated_images:
            imshow(treated_image)
            show()
        
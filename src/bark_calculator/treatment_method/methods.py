import numpy as np

from skimage.color import rgb2grey

class TreatmentMethod:

    def treat_image(self, image):
        raise NotImplementedError("Should be implemented by children classes.")
        
    def treat_images(self, images):
        return [self.treat_image(image) for image in images]
        

class EdgeDetection(TreatmentMethod):

    def treat_image(self, image):
        pass


class Identity(TreatmentMethod):

    def treat_image(self, image):
        return image


class BlackFilter(TreatmentMethod):

    def treat_image(self, image):
        return image, self.remove_all_black_lines(image)

    def remove_all_black_lines(self, image):
        black_image = rgb2grey(image)

        return image[np.mean(black_image <= 10**-5, axis=1) < 10**-3]

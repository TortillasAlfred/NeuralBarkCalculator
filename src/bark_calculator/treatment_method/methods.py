import numpy as np

from skimage.color import rgb2grey, grey2rgb
import cv2

class TreatmentMethod:

    def treat_image(self, image):
        raise NotImplementedError("Should be implemented by children classes.")
        
    def treat_images(self, images):
        return [self.treat_image(image) for image in images]
        

class EdgeDetection(TreatmentMethod):

    def __init__(self):
        super().__init__()
        self.black_masker = BlackMask()


    def treat_image(self, image):
        treated_list = [rgb2grey(image)]

        # Add edge detection
        treated_list.append(self.auto_canny(image))
        
        return treated_list

    def auto_canny(self, image):
        sigma = 0.33

        black_mask = self.black_masker.treat_image(image)

        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        grey_image_masked = np.ma.array(grey_image, mask=black_mask)

        v = np.median(grey_image_masked)

        #---- apply automatic Canny edge detection using the computed median----
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        return cv2.Canny(grey_image_masked, lower, upper)

class Identity(TreatmentMethod):

    def treat_image(self, image):
        return [image]


class BlackFilter(TreatmentMethod):

    def treat_image(self, image):
        return [image, self.remove_black_regions(image)]

    def remove_black_regions(self, image):
        black_image = rgb2grey(image)

        black_points = black_image < 0.15

        black_lines = np.mean(black_image, axis=1) < 10 ** -1
        
        black_mask = np.ones_like(image, dtype=bool)

        black_mask[black_points] = False
        black_mask[black_lines, :, :] = False

        masked_image = np.copy(image)
        masked_image[~black_mask] = 100

        return masked_image

class BlackMask(TreatmentMethod):

    def treat_image(self, image):
        return self.make_mask(image)

    def make_mask(self, image):
        black_image = rgb2grey(image)

        black_points = black_image < 0.15

        black_lines = np.mean(black_image, axis=1) < 10 ** -1
        
        black_mask = np.ones(image.shape[:-1], dtype=bool)

        black_mask[black_points] = False
        black_mask[black_lines, :] = False

        return black_mask
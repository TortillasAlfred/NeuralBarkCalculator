import numpy as np

from skimage.color import rgb2grey, grey2rgb
from skimage.feature import canny
from skimage.transform import rescale

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

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
        image = rescale(image, 1/8)

        treated_list = [rgb2grey(image)]

        # Add edge detection
        treated_list.extend(self.auto_canny(image))
        
        return treated_list

    def auto_canny(self, image):
        black_mask = self.black_masker.make_mask(image)

        pca = PCA(n_components=1)

        final_image = np.zeros((image.shape[0], image.shape[0]))
        pca_treated = pca.fit_transform(image[black_mask])
        black_white_pca = 1 - minmax_scale(pca_treated)
        np.put(final_image, np.where(black_mask.flatten()), black_white_pca)
        
        return [canny(final_image, sigma=1.6)]

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
        pipi = np.ma.array(rgb2grey(image), mask=self.make_mask(image))
        return [image, pipi]

    def make_mask(self, image):
        black_image = rgb2grey(image)

        black_points = black_image < 0.15

        black_lines = np.mean(black_image, axis=1) < 10 ** -1
        
        black_mask = np.ones(image.shape[:-1], dtype=bool)

        black_mask[black_points] = False
        black_mask[black_lines, :] = False

        return black_mask

class ColorFilter(TreatmentMethod):

    def treat_image(self, image):
        image = rescale(image, 1/8)

        mask = BlackMask().make_mask(image)

        pca = PCA(n_components=1)

        final_image = np.ones_like(image) * 0.5
        pca_treated = pca.fit_transform(image[mask])
        black_white_pca = 1 - minmax_scale(pca_treated)
        final_image[mask] = black_white_pca
        
        return [image, final_image]
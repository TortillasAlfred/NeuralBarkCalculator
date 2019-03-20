import numpy as np

from skimage.color import rgb2grey, grey2rgb, rgb2hsv, hsv2rgb
from skimage.feature import canny
from skimage.filters import sobel, threshold_otsu, threshold_adaptive, \
    laplace, scharr, prewitt, roberts, gabor_kernel
from skimage.feature.texture import local_binary_pattern
from skimage.transform import rescale
from skimage.measure import label
from skimage.morphology import disk, label
from skimage.filters.rank import entropy
from skimage.segmentation import watershed
from skimage.exposure import histogram
from skimage import img_as_float

from scipy import ndimage as ndi

from cv2 import Laplacian

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

import cv2


class TreatmentMethod:

    def treat_image(self, image):
        raise NotImplementedError("Should be implemented by children classes.")

    def treat_images(self, images):
        return [self.treat_image(image) for image in images]


class Grey(TreatmentMethod):

    def treat_image(self, image):
        return rgb2grey(image)


class EdgeDetection(TreatmentMethod):

    def __init__(self):
        super().__init__()
        self.black_masker = BlackMask()

    def treat_image(self, image):
        image = rescale(image, 1/8)

        treated_list = [rgb2grey(image)]

        # Add edge detection
        image = self.filtered_image(image)
        treated_list.append(sobel(image))
        treated_list.append(laplace(image))
        treated_list.append(prewitt(image))
        treated_list.append(scharr(image))
        treated_list.append(roberts(image))

        return treated_list

    def filtered_image(self, image):
        black_mask = self.black_masker.make_mask(image)

        pca = PCA(n_components=1)

        final_image = np.zeros((image.shape[0], image.shape[0]))
        pca_treated = pca.fit_transform(image[black_mask])
        black_white_pca = 1 - minmax_scale(pca_treated)
        np.put(final_image, np.where(black_mask.flatten()), black_white_pca)

        return final_image

    def auto_canny(self, image):
        return [canny(final_image, sigma=1.6)]


class ComponentDetection(TreatmentMethod):

    def treat_image(self, image):
        image = rescale(image, 1/8)

        black_mask = BlackMask().make_mask(image)

        pca = PCA(n_components=1)

        final_image = np.zeros((image.shape[0], image.shape[0]))
        pca_treated = pca.fit_transform(image[black_mask])
        black_white_pca = 1 - minmax_scale(pca_treated)
        np.put(final_image, np.where(black_mask.flatten()), black_white_pca)

        edges = sobel(final_image)

        markers = np.zeros_like(final_image)
        markers[final_image > 0.6] = 2
        markers[final_image < 0.45] = 1
        markers[~black_mask] = 0

        ws_image = grey2rgb(watershed(edges, markers, mask=black_mask)/2)

        return [image, (image + ws_image)/2, ws_image]

    def treat_image_with_markers(self, image, markers):
        image = rescale(image, 1/8)

        black_mask = BlackMask().make_mask(image)

        pca = PCA(n_components=1)

        final_image = np.zeros((image.shape[0], image.shape[0]))
        pca_treated = pca.fit_transform(image[black_mask])
        black_white_pca = 1 - minmax_scale(pca_treated)
        np.put(final_image, np.where(black_mask.flatten()), black_white_pca)

        markers[~black_mask] = 0

        edges = sobel(final_image)

        ws_image = grey2rgb(watershed(edges, markers, mask=black_mask)/2)

        return [image, (image + ws_image)/2, ws_image]


class Thresholding(TreatmentMethod):

    def treat_image(self, image):
        image = rescale(image, 1/8)

        black_mask = BlackMask().make_mask(image)

        pca = PCA(n_components=1)

        final_image = np.zeros((image.shape[0], image.shape[0]))
        pca_treated = pca.fit_transform(image[black_mask])
        black_white_pca = 1 - minmax_scale(pca_treated)
        np.put(final_image, np.where(black_mask.flatten()), black_white_pca)

        edges = sobel(final_image)

        markers = threshold_adaptive(
            final_image, block_size=35).astype(np.float)
        markers += 1
        markers[~black_mask] = 0

        ws_image = watershed(edges, markers, mask=black_mask)

        return [final_image, ws_image]


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

        im = image.reshape((image.shape[0] * image.shape[0], 3))

        kmeans = KMeans(n_clusters=4, n_jobs=-1, n_init=50)

        k_means_4_image = kmeans.fit_predict(
            im).reshape((image.shape[0], image.shape[0]))
        sorted_centers_args = np.argsort(np.linalg.norm(
            kmeans.cluster_centers_, axis=-1))[::-1]

        for prev_number, sorted_number in enumerate(sorted_centers_args):
            np.put(k_means_4_image, kmeans.labels_ ==
                   prev_number, sorted_number)

        k_means_4_image_treated = grey2rgb(
            1 - (k_means_4_image / k_means_4_image.max()))

        return [image, (k_means_4_image_treated + image)/2, k_means_4_image_treated, k_means_4_image]


class Entropy(TreatmentMethod):

    def treat_image(self, image):
        final_image = rgb2grey(image)

        treated_list = [image]

        for r in [10, 12, 15, 20, 25]:
            filtered = entropy(final_image, disk(r))
            treated_list.append(filtered)

        return treated_list


class V1(TreatmentMethod):

    def treat_image(self, image):
        treated_list = [rescale(image, 1/8)]

        color_filtered = ColorFilter().treat_image(image)[1:]

        treated_list.extend(color_filtered[:-1])

        treated_list.extend(ComponentDetection().treat_image(image)[1:])

        markers = color_filtered[-1]

        black_cluster = np.median(markers[-1])

        markers_for_component_detection = np.zeros_like(markers)
        markers_for_component_detection[markers == markers.max()] = 2
        markers_for_component_detection[markers == markers.min()] = 1

        treated_list.append(ComponentDetection().
                            treat_image_with_markers(image, markers_for_component_detection)[2])

        return treated_list


class V2(TreatmentMethod):

    def power(self, image, kernel):
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

    def treat_image(self, image):
        black_white_big = rgb2grey(image)
        image = rescale(image, 1/8)

        black_white = rgb2grey(image)

        treated_images = [image]
        for h, l in [(0.55, 0.70), (0.55, 0.72), (0.55, 0.74),
                     (0.55, 0.76), (0.55, 0.78)]:
            hist = self.get_hist_and_threshhold(black_white, h, l)
            # treated_images.append(hist)

        hist = self.get_hist_and_threshhold(black_white, 0.55, 0.70)
        edges = sobel(hist)

        black_mask = BlackMask().make_mask(image)
        markers = np.copy(hist)
        markers[~black_mask] = 0

        ws_image = grey2rgb(watershed(edges, markers, mask=black_mask)/2)

        treated_images.append(hist)
        treated_images.append(ws_image)

        gabors = []
        for freq in [0.4, 1.0, 2.0, 5.0]:
            for theta in range(8):
                theta = theta / 4. * np.pi
                kernel = gabor_kernel(freq, theta=theta)
                gabors.append(self.power(black_mask, kernel))

            treated_images.append(np.sum(gabors, axis=0))

        # black_mask = BlackMask().make_mask(image)
        # edges = sobel(black_white, mask=black_mask)
        # treated_images.append(edges)

        # for thresh in [0.15, 0.20, 0.25]:
        #     pipi = np.ones_like(edges)
        #     pipi[edges > thresh] = 0
        #     treated_images.append(pipi)

        # kernel = np.array([[-1, -1, 0, -1, -1],
        #                    [-1, 0, 2, 0, -1],
        #                    [0, 2, 4, 2, 0],
        #                    [-1, 0, 2, 0, -1],
        #                    [-1, -1, 0, -1, -1]])

        # hsv_image = rgb2hsv(image)

        # dst_h = cv2.filter2D(hsv_image[:, :, 0], -1, kernel)
        # dst_s = cv2.filter2D(hsv_image[:, :, 1], -1, kernel)
        # dst_v = cv2.filter2D(hsv_image[:, :, 2], -1, kernel)

        # dst_hsv = np.stack([dst_h, dst_s, dst_v], axis=-1)

        # treated_images.append(hsv2rgb(dst_hsv))
        # treated_images.append(rgb2grey(hsv2rgb(dst_hsv)))

        # win_rows, win_cols = 150, 150
        # win_mean = ndimage.uniform_filter(
        #     black_white_big, (win_rows, win_cols))
        # win_sqr_mean = ndimage.uniform_filter(
        #     black_white_big**2, (win_rows, win_cols))
        # win_var = win_sqr_mean - win_mean**2

        # treated_images.append(win_var)

        return treated_images

    def get_hist_and_threshhold(self, image, h, l):
        returned_image = np.zeros_like(image)
        returned_image[image < h] = 1
        returned_image[image > l] = 2

        return returned_image

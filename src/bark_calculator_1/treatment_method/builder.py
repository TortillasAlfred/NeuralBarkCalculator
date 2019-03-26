from treatment_method.methods import *


def build_from_treatment_method_arg(treatment_method):
    if treatment_method == "edge_detection":
        return EdgeDetection()

    if treatment_method == "id":
        return Identity()

    if treatment_method == "black_filter":
        return BlackFilter()

    if treatment_method == "black_mask":
        return BlackMask()

    if treatment_method == "color_filter":
        return ColorFilter()

    if treatment_method == "component_detection":
        return ComponentDetection()

    if treatment_method == "threshold":
        return Thresholding()

    if treatment_method == "v1":
        return V1()

    if treatment_method == "entropy":
        return Entropy()

    if treatment_method == "v2":
        return V2()

    if treatment_method == "grey":
        return Grey()

    if treatment_method == "eq":
        return Equalizer()

    if treatment_method == "black_trim":
        return BlackTrimmer()

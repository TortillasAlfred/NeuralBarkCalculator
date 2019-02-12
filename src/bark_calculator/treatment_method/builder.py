from treatment_method.methods import *


def build_from_treatment_method_arg(treatment_method):
    if treatment_method == "edge_detection":
        return EdgeDetection()

    if treatment_method == "id":
        return Identity()

    if treatment_method == "black_filter":
        return BlackFilter()

from image_treatment_processor.processors import *


def build_from_image_processor_arg(image_processor):
    if image_processor == "display":
        return DisplayProcessor()

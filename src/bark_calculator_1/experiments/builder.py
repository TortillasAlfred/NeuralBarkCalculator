from experiments.experiments import *

from image_loader.builder import build_from_image_loader_arg
from treatment_method.builder import build_from_treatment_method_arg
from image_treatment_processor.builder import build_from_image_processor_arg

def build_experiment_from_args(args):
    image_loader = build_from_image_loader_arg(args.image_loader)
    treatment_method = build_from_treatment_method_arg(args.treatment_method)
    processor = build_from_image_processor_arg(args.image_processor)
    
    return GenericExperiment(image_loader, treatment_method, processor)
    
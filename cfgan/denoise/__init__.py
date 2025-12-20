import os
from typing import Any, Dict, Union

import numpy as np
import torch

from ..common.utils import get_image_dimension, normalize, set_image_dimension, set_image_size
from .filter import CrossFilter
from .model import CFGANModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILTER_ARGS = {"fraction": (0.15, 0.60), "rotation": -22.5, "types": "ri", "gaussian_radius": 1.25}
IMAGE_DIMENSION = 4
IMAGE_SIZE = [256, 256]


def cross_filter_denoise_image(
    image: Union[np.ndarray, torch.Tensor], filter_args: Dict[str, Any], model_path: Union[str, os.PathLike]
) -> Union[np.ndarray, torch.Tensor]:
    dimension = get_image_dimension(image)
    image = set_image_dimension(image, IMAGE_DIMENSION)
    image = set_image_size(image, IMAGE_SIZE)

    # Cross-Filter
    if filter_args:
        filter_args = {
            "fraction": filter_args.pop("fraction", FILTER_ARGS["fraction"]),
            "rotation": filter_args.pop("rotation", FILTER_ARGS["rotation"]),
            "types": filter_args.pop("types", FILTER_ARGS["types"]),
            "gaussian_radius": filter_args.pop("gaussian_radius", FILTER_ARGS["gaussian_radius"]),
        }
    else:
        filter_args = FILTER_ARGS
    cross_filter = CrossFilter(shape=IMAGE_SIZE, **filter_args)
    for i in range(len(image)):
        image[i, 0] = cross_filter.denoise(image[i, 0])

    # CFGAN
    cfgan_model = CFGANModel(model_path)
    image_max = image.max()
    image = cfgan_model.denoise(normalize(image))
    image *= image_max

    image = set_image_dimension(image, dimension)

    return image

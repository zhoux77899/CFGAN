from typing import List, Tuple, Union

import numpy as np
import torch

from ..common import restrict_values


@restrict_values("num_dimensions", [1, 2, 3, 4])
def get_image_size(image: Union[np.ndarray, torch.Tensor], num_dimensions: int = 2) -> List[int]:
    """Get image size

    :param image: [``numpy.ndarray`` or ``torch.Tensor``]: Image.
    :param num_dimensions:
    :return: ``List[int]`` Image dimension.
    """
    if isinstance(image, np.ndarray):
        return list(image.shape)[-num_dimensions:]
    elif isinstance(image, torch.Tensor):
        return list(image.size())[-num_dimensions:]
    else:
        raise TypeError("Input must be a `numpy.ndarray` or `torch.Tensor`")


def get_image_dimension(image: Union[np.ndarray, torch.Tensor]) -> int:
    if isinstance(image, np.ndarray):
        return image.ndim
    elif isinstance(image, torch.Tensor):
        return image.dim()
    else:
        raise TypeError("Input must be a `numpy.ndarray` or `torch.Tensor`")


def set_image_dimension(
    image: Union[np.ndarray, torch.Tensor],
    dimension: int = 4,
) -> Union[np.ndarray, torch.Tensor]:
    """Set image to input dimension.

    :param image: [``numpy.ndarray`` or ``torch.Tensor``]: Image.
    :param dimension: [``int``]: Expected dimension of image.
    :return: [``numpy.ndarray`` or ``torch.Tensor``]: Dimension set image.
    """
    if isinstance(image, np.ndarray):
        if image.ndim < dimension:
            while image.ndim < dimension:
                image = np.expand_dims(image, axis=0)
        if image.ndim > dimension:
            while image.ndim > dimension:
                image = np.squeeze(image, axis=-3)
    elif isinstance(image, torch.Tensor):
        if image.dim() < dimension:
            while image.dim() < dimension:
                image = torch.unsqueeze(image, dim=0)
        if image.dim() > dimension:
            while image.dim() > dimension:
                image = torch.squeeze(image, dim=-3)
    else:
        raise TypeError("Input must be a `numpy.ndarray` or `torch.Tensor`")
    return image


def set_image_size(
    image: Union[np.ndarray, torch.Tensor],
    size: Union[List[int], Tuple[int, int]],
) -> Union[np.ndarray, torch.Tensor]:
    """Set image to input size.

    :param image: [``numpy.ndarray`` or ``torch.Tensor``]: Image.
    :param size: [``List[int]`` or ``Tuple[int, int]``]: Expected image size.
    :return: [``numpy.ndarray`` or ``torch.Tensor``]: Size set image.
    """
    cur_height, cur_width = get_image_size(image, 2)
    tar_height, tar_width = size
    dimension = get_image_dimension(image)
    image = set_image_dimension(image, 2)

    if cur_height > tar_height:
        image = image[(cur_height - tar_height) // 2 : (cur_height + tar_height) // 2, :]
    if cur_height < tar_height:
        if isinstance(image, np.ndarray):
            expanded_image = np.zeros((tar_height, cur_width), dtype=image.dtype)
        elif isinstance(image, torch.Tensor):
            expanded_image = torch.zeros((tar_height, cur_width), dtype=image.dtype)
        else:
            raise TypeError("Input must be a `numpy.ndarray` or `torch.Tensor`")
        expanded_image[(tar_height - cur_height) // 2 : (tar_height + cur_height) // 2, :] = image
        image = expanded_image

    cur_height, cur_width = get_image_size(image, 2)

    if cur_width > tar_width:
        image = image[:, (cur_width - tar_width) // 2 : (cur_width + tar_width) // 2]
    if cur_width < tar_width:
        if isinstance(image, np.ndarray):
            expanded_image = np.zeros((cur_height, tar_width), dtype=image.dtype)
        elif isinstance(image, torch.Tensor):
            expanded_image = torch.zeros((cur_height, tar_width), dtype=image.dtype)
        else:
            raise TypeError("Input must be a `numpy.ndarray` or `torch.Tensor`")
        expanded_image[:, (tar_width - cur_width) // 2 : (tar_width + cur_width) // 2] = image
        image = expanded_image

    image = set_image_dimension(image, dimension)

    return image


def normalize(image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    return (image - image.min()) / (image.max() - image.min())


def unify_positions_instance(positions: Union[np.ndarray, torch.Tensor, List[List[float]]]) -> np.ndarray:
    if isinstance(positions, np.ndarray):
        return positions
    elif isinstance(positions, torch.Tensor):
        return positions.detach().cpu().numpy()
    elif isinstance(positions, list):
        return np.array(positions)
    else:
        raise ValueError(f"Unsupported type `{type(positions)}` of positions")

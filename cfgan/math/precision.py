from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..common.env import ENV
from ..common.utils import unify_positions_instance


@dataclass
class Error:
    errors: np.ndarray = field(default=None)
    total_error: np.ndarray = field(default=None)
    mean_error: np.ndarray = field(default=None)
    std_error: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        if self.errors is None:
            raise ValueError("errors must be provided")
        if self.total_error is None:
            self.total_error = self.errors.sum()
        if self.mean_error is None:
            self.mean_error = self.errors.mean()
        if self.std_error is None:
            self.std_error = self.errors.std()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} (\n\t"
            f"errors={self.errors}\n\t"
            f"total_error={self.total_error}\n\t"
            f"mean_error={self.mean_error}\n\t"
            f"std_error={self.std_error}\n"
            f")"
        )


def convert_pixel_to_um(
    positions: Union[np.ndarray, torch.Tensor, List[List[float]]],
    pitch: float = ENV.pitch,
    std_size: Tuple[int, int] = ENV.std_size,
    clip_size: Tuple[int, int] = ENV.clip_size,
    clip_bias: Tuple[int, int] = ENV.clip_bias,
) -> Union[np.ndarray, torch.Tensor, List[List[float]]]:
    centers = tuple(size / 2 * pitch for size in std_size)
    edges = tuple(center - (size / 4 - bias) * pitch for center, size, bias in zip(centers, clip_size, clip_bias))
    conversions = [lambda x: x * pitch + edges[0], lambda y: y * pitch + edges[1], lambda z: z]
    um_positions = [[conversion(p) for p, conversion in zip(position, conversions)] for position in positions]

    if isinstance(positions, np.ndarray):
        return np.array(um_positions)
    if isinstance(positions, torch.Tensor):
        return torch.tensor(um_positions)
    return um_positions


def calc_prediction_error(
    real_positions: Union[np.ndarray, torch.Tensor, List[List[float]]],
    pred_positions: Union[np.ndarray, torch.Tensor, List[List[float]]],
) -> Error:
    distance_matrix = cdist(real_positions, pred_positions, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    errors = distance_matrix[row_ind, col_ind]
    return Error(errors)


def localization_prediction_errors(
    real_positions: Union[np.ndarray, torch.Tensor, List[List[float]]],
    pred_positions: Union[np.ndarray, torch.Tensor, List[List[float]]],
    is_pixel: Optional[List[bool]] = None,
) -> Tuple[Error, Error, Error]:
    if is_pixel is None:
        is_pixel = [False, True]
    if is_pixel[0]:
        real_positions = convert_pixel_to_um(real_positions)
    if is_pixel[1]:
        pred_positions = convert_pixel_to_um(pred_positions)

    real_positions = unify_positions_instance(real_positions)
    pred_positions = unify_positions_instance(pred_positions)

    radial_errors = calc_prediction_error(real_positions, pred_positions)
    lateral_errors = calc_prediction_error(real_positions[:, :2], pred_positions[:, :2])
    axial_errors = calc_prediction_error(real_positions[:, 2:], pred_positions[:, 2:])

    return radial_errors, lateral_errors, axial_errors

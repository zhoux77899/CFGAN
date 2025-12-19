import math
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from .colors import BLACK, WHITE, get_color
from .utils import axis_style_process_3d
from ..common.env import ENV
from ..common.utils import unify_positions_instance
from ..math.precision import convert_pixel_to_um


PITCH = ENV.pitch
STD_SIZE = ENV.std_size
IMAGE_SIZE = ENV.clip_size
CENTERS = tuple(size / 2 * PITCH for size in STD_SIZE)
EDGES = tuple(center - image_size / 4 * PITCH for center, image_size in zip(CENTERS, IMAGE_SIZE))


def plot_positions_grid(
        *pred_positions: Union[np.ndarray, torch.Tensor, List[List[float]]],
        real_positions: Union[np.ndarray, torch.Tensor, List[List[float]]],
        labels: Optional[List[str]] = None,
        is_pixel: Optional[List[bool]] = None,
        figsize: Tuple[int, int] = (6, 3)
) -> None:
    """
    Plot the predicted and real positions in a 3D grid.

    This function visualizes the real and predicted positions in a 3D coordinate system.
    It can handle multiple sets of predicted positions and distinguish them with different colors.

    :param pred_positions: Variable arguments, representing multiple sets of predicted positions.
    :param real_positions: The real positions, used as a reference for comparison.
    :param labels: Optional, labels for each set of positions.
        Defaults to "Ground Truth" for real positions and "Prediction" for predicted positions.
    :param is_pixel: Optional, indicates whether the position data is in pixel units.
        Defaults to True for predicted positions and False for real positions.
    :param figsize: Optional, the size of the figure.
        Defaults to (6, 3).
    """

    if is_pixel is None:
        is_pixel = [True] * len(pred_positions) + [False]
    if labels is None:
        labels = ["Prediction"] * len(pred_positions) + ["Ground Truth"]

    scatter_args = {"alpha": 0.5, "linewidths": 0.5}

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.view_init(elev=20, azim=60)

    # plot real positions
    if is_pixel[-1]:
        real_positions = convert_pixel_to_um(real_positions)
    real_positions = unify_positions_instance(real_positions)
    scatter = ax.scatter(
        real_positions[:, 0],
        real_positions[:, 1],
        real_positions[:, 2],
        s=30,
        facecolor=(1, 1, 1, 0),
        edgecolors=BLACK,
        label=labels[-1],
        **scatter_args
    )
    # plot prediction positions
    for i, pred_position in enumerate(pred_positions):
        if is_pixel[i]:
            pred_position = convert_pixel_to_um(pred_position)
        pred_position = unify_positions_instance(pred_position)
        ax.scatter(
            pred_position[:, 0],
            pred_position[:, 1],
            pred_position[:, 2],
            s=15,
            color=get_color(i),
            label=labels[i],
            **scatter_args
        )
    scatter.set_markerfacecolor = (1, 1, 1, 0)

    ax.legend(bbox_to_anchor=(1.35, 0.9), fontsize="small", fancybox=False)
    ax.set_xlabel("X (µm)", color=BLACK)
    ax.set_ylabel("Y (µm)", color=BLACK)
    ax.set_zlabel("Z (µm)", color=BLACK)
    ax.set_xlim(math.floor(EDGES[0]), math.ceil(2 * CENTERS[0] - EDGES[0]))
    ax.set_ylim(math.floor(EDGES[1]), math.ceil(2 * CENTERS[1] - EDGES[1]))
    ax.set_zlim(ENV.limit - ENV.bias, ENV.limit + ENV.bias)

    plt.rcParams.update({
        "grid.linewidth": 0.5,
        "grid.color": [0, 0, 0, 0.1],
    })

    axis_style_process_3d(ax)

    plt.show()

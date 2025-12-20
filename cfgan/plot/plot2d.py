from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from ..math.precision import Error
from .colors import BLACK, BLUE, SILVER, WHITE, YELLOW
from .utils import axis_style_process_2d, set_figsize


def plot_localization_errors(
    errors: Tuple[Error, Error, Error], ylimit: float = 0.1, figsize: Tuple[int, int] = (4, 2)
) -> None:
    radial_errors, lateral_errors, axial_errors = errors
    idx = np.arange(radial_errors.errors.size)
    width = 0.25

    fig, ax = plt.subplots()

    ax.bar(x=idx - width, height=radial_errors.errors, width=width, color=SILVER, label="Radial")
    ax.bar(x=idx, height=lateral_errors.errors, width=width, color=YELLOW, label="Lateral")
    ax.bar(x=idx + width, height=axial_errors.errors, width=width, color=BLUE, label="Axial")

    ax.legend(
        loc="upper left" if radial_errors.errors[0] <= radial_errors.errors[-1] else "upper right",
        ncol=1,
        fontsize="x-small",
        frameon=False,
        fancybox=False,
        framealpha=1,
        handlelength=2.5,
        handletextpad=0.5,
        columnspacing=0.5,
    )
    ax.set_xticks(idx, idx)
    ax.set_ylabel("Localization Error (µm)", color=BLACK)
    ax.set_xlim([idx.min() - 1, idx.max() + 1])
    ax.set_ylim([0, ylimit])

    xlimn = idx.size
    ylimn = 5
    xtick = 1
    ytick = ylimit / ylimn
    axis_style_process_2d(ax, xlimn=xlimn, ylimn=ylimn, xticks=[xtick, xtick], yticks=[ytick, ytick])
    set_figsize(*figsize, ax)

    plt.show()

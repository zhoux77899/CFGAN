from typing import List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .colors import BLACK, SILVER


def axis_style_process_2d(
        ax: plt.Axes,
        xlimn: int,
        ylimn: int,
        xticks: Optional[Union[List[int], List[float]]],
        yticks: Optional[Union[List[int], List[float]]],
        grid: bool = False,
        fullframe: bool = False
) -> None:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(xlimn))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(ylimn))
    if xticks:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xticks[0]))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xticks[1]))
    else:
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    if yticks:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yticks[0]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(yticks[1]))

    if grid is True:
        ax.grid(True, axis="both", linewidth=0.5, color="lightgray")
        ax.set_axisbelow(True)

    ax.spines["left"].set_color(BLACK)
    ax.spines["right"].set_color(SILVER if not fullframe else BLACK)
    ax.spines["top"].set_color(SILVER if not fullframe else BLACK)
    ax.spines["bottom"].set_color(BLACK)

    ax.tick_params(which="both", axis="both", direction="in")


def axis_style_process_3d(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3, steps=[1, 2, 4, 5, 10]))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3, steps=[1, 2, 4, 5, 10]))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(3, steps=[1, 2, 4, 5, 10]))

    ax.xaxis.line.set_color(BLACK)
    ax.yaxis.line.set_color(BLACK)
    ax.zaxis.line.set_color(BLACK)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.tick_params(which="both", axis="both", colors=BLACK, width=2.5)
    for tickline in ax.xaxis.get_majorticklines():
        tickline.set_solid_capstyle("round")
        tickline.set_linewidth(2)
    for tickline in ax.yaxis.get_majorticklines():
        tickline.set_solid_capstyle("round")
        tickline.set_linewidth(2)
    for tickline in ax.zaxis.get_ticklines():
        tickline.set_solid_capstyle("round")
        tickline.set_linewidth(2)


def set_figsize(w: float, h: float, ax: Optional[plt.Axes] = None) -> None:
    if not ax:
        ax = plt.gca()

    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom

    ax.figure.set_size_inches(w / (r - l), h / (t - b))

from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow(
    *imgs: np.ndarray,
    figheight: float = 3,
    px_size: float = 8e-8,
    cmap: str = "viridis",
    titles: Union[List[str], str] = None,
    normalization: bool = False,
    scalebar: bool = False,
    colorbar: Union[bool, List[bool]] = False,
    rms: bool = False,
    **kwargs,
) -> None:
    """
    Display one or more images with customizable options.
    This function visualizes a series of images using Matplotlib,
    with options to set the figure height, pixel size, color map, titles, normalization, scalebar, colorbar, and more.
    It also supports adding scale bars and color bars to the images.

    :param imgs: One or more images to be displayed.
    :param figheight: The height of the figure in inches. (default is 3)
    :param px_size: The size of a pixel in meters. (default is 8e-8)
    :param cmap: The colormap to use for displaying the images. (default is "viridis")
    :param titles: Titles for the images. Can be a list of titles or a single title. (default is None)
    :param normalization: If True, normalize the images to the maximum value among them. (default is False)
    :param scalebar: If True, add a scalebar to the first image. (default is False)
    :param colorbar: If True, add a colorbar to the images.
        Can be a single boolean or a list of booleans specifying whether to add a colorbar for each image.
        (default is False).
    :param rms: If True, calculate the root-mean-square (RMS) values of the images. (default is False)
    :param kwargs: Additional keyword arguments to pass to `imshow`.
    :return: None
    """

    figwidth = figheight * len(imgs)
    subplots = len(imgs)
    fig, axes = plt.subplots(1, subplots, figsize=(figwidth, figheight))

    for idx, ax in enumerate(axes if subplots > 1 else [axes]):
        vmin = kwargs.pop("vmin", imgs[idx].min() if normalization else min([img.min() for img in imgs]))
        vmax = kwargs.pop("vmax", imgs[idx].max() if normalization else max([img.max() for img in imgs]))
        im = ax.imshow(imgs[idx], vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none", **kwargs)

        if idx == 0 and scalebar:
            scale_bar = ScaleBar(
                dx=px_size,
                units="m",
                dimension="si-length",
                height_fraction=0.02,
                length_fraction=0.25,
                location="lower left",
                scale_loc="top",
                frameon=False,
                color="white",
                label_formatter=lambda value, unit: "{} {}".format(value, unit.replace("$\\mathrm{\\mu}$", "µ")),
            )
            ax.add_artist(scale_bar)

        if any(
            [
                isinstance(colorbar, bool) and colorbar and idx == len(imgs) - 1,
                isinstance(colorbar, list) and colorbar[idx],
            ]
        ):
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            fig = ax.get_figure()
            fig.add_axes(ax_cb)
            plt.colorbar(im, cax=ax_cb)

        ax.set_title(titles[idx] if isinstance(titles, list) else "")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(titles if isinstance(titles, str) else "")
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, hspace=0)
    plt.show()


def get_background(image: Union[torch.Tensor, np.ndarray], background_threshold: float = 10.0) -> float:
    """
    Calculate the background value of an image based on a given threshold.
    This function takes an image input, either as a PyTorch tensor or a NumPy array,
    and computes the background value by finding the specified percentile in the pixel values of the image.

    :param image: The input image, which can be a PyTorch tensor or a NumPy array.
    :param background_threshold: The percentile to use for calculating the background value, by default 10.0.
    :return: The calculated background value at the specified percentile.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    return np.percentile(image, background_threshold).item()

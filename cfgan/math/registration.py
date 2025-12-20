import copy
from typing import List, Tuple, Union

import numpy as np
import scipy
import torch

from cfgan.common.img import get_background
from cfgan.common.logging import logger

STD_IMAGES_STEP = 0.01
EPS = 1e-09
SI = 1e03


def cross_correlation_pre_process(
    image_a: Union[np.ndarray, torch.Tensor], image_b: Union[np.ndarray, torch.Tensor], threshold: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-process two images for cross-correlation analysis.

    :param image_a: The first image.
    :param image_b: The second image.
    :param threshold: A float value that determines the sensitivity of the displacement calculation.
    :return: The pre-processed images as NumPy arrays.
    """

    image_a = copy.deepcopy(image_a)
    image_b = copy.deepcopy(image_b)

    if isinstance(image_a, torch.Tensor):
        image_a = image_a.cpu().numpy()
    if isinstance(image_b, torch.Tensor):
        image_b = image_b.cpu().numpy()

    if threshold:
        roi = image_a < image_a.max() * threshold
        image_a[roi] = get_background(image_a, threshold)
        image_b[roi] = get_background(image_b, threshold)

    return image_a, image_b


def normalized_cross_correlation(
    reference_image: Union[np.ndarray, torch.Tensor],
    moving_image: Union[np.ndarray, torch.Tensor],
    threshold: float = None,
) -> float:
    r"""
    Calculate the Normalized Cross Correlation (NCC) similarity between two images as

    .. math::
        NCC = \frac{\sum \sum I_r I_m}{\sqrt{\sum \sum I_r^2} \sqrt{\sum \sum I_m^2}}.

    :param reference_image: The reference image to which the moving image will be compared.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param moving_image: The image that needs to be compared to the reference image.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param threshold: A float value that determines the sensitivity of the displacement calculation.
    :return: NCC similarity between the two images.
    """

    reference_image, moving_image = cross_correlation_pre_process(reference_image, moving_image, threshold)

    cross_corr = np.sum(reference_image * moving_image)
    normalized_corr = cross_corr / (np.sqrt(np.sum(reference_image**2)) * np.sqrt(np.sum(moving_image**2)))

    return normalized_corr


def differential_cross_correlation(
    moving_image: Union[np.ndarray, torch.Tensor],
    fitting_image: Union[np.ndarray, torch.Tensor],
    threshold: float = None,
) -> float:
    r"""
    Calculate the Differential Cross Correlation (DCC) similarity between two images as

    .. math::
        DCC = \frac{\sum \sum (I_m - I_f) I_f}{\sqrt{\sum \sum I_f^2}}.

    :param moving_image: The image that needs to be compared.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param fitting_image: The image that has fitting molecule in the moving image.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param threshold: A float value that determines the sensitivity of the displacement calculation.
    :return: DCC similarity between the two images.
    """

    moving_image, fitting_image = cross_correlation_pre_process(moving_image, fitting_image, threshold)

    cross_corr = np.sum(np.abs(moving_image - fitting_image) * np.abs(fitting_image))
    differential_corr = cross_corr / np.sqrt(np.sum(fitting_image**2))

    return differential_corr


def get_argmax(
    x: Union[List[Union[int, float]], np.ndarray, torch.Tensor],
    y: Union[List[Union[int, float]], np.ndarray, torch.Tensor],
    x_fit: Union[List[Union[int, float]], np.ndarray, torch.Tensor],
    x_left: int,
    x_right: int,
    ratio: int,
    deg: int = 4,
) -> Tuple[int, np.ndarray]:
    """
    Fits a polynomial of degree `deg` to the data points (`x`, `y`), evaluates the polynomial at `x_fit`,
    and finds the x-value within the interval [`x_left`, `x_right`] that maximizes the fitted polynomial.

    :param x: List or array of x-coordinates of the data points.
    :param y: List or array of y-coordinates of the data points.
    :param x_fit: List or array of x-coordinates where the polynomial is evaluated.
    :param x_left: Left boundary of the interval to search for the maximum.
    :param x_right: Right boundary of the interval to search for the maximum.
    :param ratio: Ratio to scale the x-coordinates for indexing.
    :param deg: Degree of the polynomial to fit (default is 4).
    :return: A tuple containing the x-value that maximizes the polynomial within the interval and the fitted y-values.
    """

    p = np.polyfit(x, y, deg=deg)
    y_fit = np.poly1d(p)(x_fit)
    x_roots = np.roots([p[i] * (len(p) - 1 - i) for i in range(0, len(p) - 1)]).real
    x_argmax = x_roots[
        np.squeeze(
            np.argmax([y_fit[int((r - x_left) * ratio)] if x_left <= r <= x_right else -np.inf for r in x_roots])
        )
    ]
    return x_argmax, y_fit


def axial_position_registration(
    reference_images: Union[np.ndarray, torch.Tensor],
    moving_image: Union[np.ndarray, torch.Tensor],
    lateral_shift: Union[List[float], Tuple[float, float]],
    start: float,
    reference_range: Union[List[float], np.ndarray, torch.Tensor],
    search_length: float = 0.1,
    search_step: float = STD_IMAGES_STEP,
    threshold: float = None,
) -> Tuple[float, float, List[float]]:
    """
    Registers the axial position of a moving image relative to a set of reference images by finding the position
    that maximizes the normalized cross-correlation.

    :param reference_images: Sequence of reference images containing depth information.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param moving_image: Moving image whose depth position needs to be determined.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param lateral_shift: A tuple containing the lateral displacement in pixel values :math:`(d_x, d_y)`
        between the two images.
    :param start: The starting position for the search.
    :param reference_range: The range of positions corresponding to the reference images.
    :param search_length: The length of the search range on one side.
        Default value is `0.1` μm.
    :param search_step: The step size for the search.
        Default value is `0.01` μm.
    :param threshold: A float value that determines the sensitivity of the displacement calculation.
    :return: A tuple containing the registered axial position, the maximum normalized cross-correlation value,
        and the list of normalized cross-correlation values.
    """

    start = np.round(start, 2)
    start_idx = np.squeeze(np.argwhere(np.round(reference_range, 2) == start))
    search_range_idx: int = int(search_length / STD_IMAGES_STEP)
    left_idx = start_idx - search_range_idx if start_idx > search_range_idx else 0
    right_idx = (
        start_idx + search_range_idx
        if search_range_idx < len(reference_range) - 1 - start_idx
        else len(reference_range) - 1
    )

    step_ratio: int = int(search_step / STD_IMAGES_STEP)
    search_ratio: int = int(SI / step_ratio)
    reference_z_idx = np.arange(left_idx, right_idx + EPS, step_ratio, dtype=int)
    fitting_z_idx = np.arange(left_idx, right_idx + EPS, step_ratio / SI)

    normalized_corr_list: List[float] = []
    for i in reference_z_idx:
        reference_image = scipy.ndimage.shift(reference_images[i], lateral_shift[::-1], order=1, mode="nearest")
        normalized_corr = normalized_cross_correlation(reference_image, moving_image, threshold)
        normalized_corr_list.append(normalized_corr)

    try:
        normalized_corr_argmax, normalized_corr_fitting = get_argmax(
            reference_z_idx, normalized_corr_list, fitting_z_idx, left_idx, right_idx, search_ratio
        )
    except ValueError:
        logger.error("attempt to get argmax of an empty cross-correlation sequence")
        normalized_corr_argmax = start_idx
        normalized_corr_fitting = np.zeros_like(fitting_z_idx)

    if normalized_corr_argmax < 0:
        normalized_corr_argmax = 0
    elif normalized_corr_argmax > len(reference_range) - 1:
        normalized_corr_argmax = len(reference_range) - 1

    normalized_corr_idx = int(normalized_corr_argmax)
    normalized_corr_idx_diff = normalized_corr_argmax - normalized_corr_idx
    z_registration: float = reference_range[normalized_corr_idx] + STD_IMAGES_STEP * normalized_corr_idx_diff
    normalized_corr_registration: float = (
        normalized_corr_fitting[int(search_ratio * (normalized_corr_argmax - left_idx))]
        if left_idx <= normalized_corr_argmax <= right_idx
        else 0
    )

    return z_registration, normalized_corr_registration, normalized_corr_list

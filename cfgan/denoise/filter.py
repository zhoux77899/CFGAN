from typing import List, Tuple, Union

import scipy
import numpy as np
import torch
from PIL import Image, ImageDraw


def filter_denoise(
        img: Union[np.ndarray, torch.Tensor],
        filter: "CrossFilter",
        eps: float = 1e-09
) -> Union[np.ndarray, torch.Tensor]:
    use_torch_tensor = isinstance(img, torch.Tensor)
    if use_torch_tensor:
        img = img.numpy()

    # calculate FFT
    image_fft = np.fft.fftshift(np.fft.fft2(img))
    image_fft_log = 20 * np.log(np.abs(image_fft))
    inner_fft = image_fft * filter.mask
    tiny_matrix = np.zeros_like(img, dtype=complex)
    tiny_matrix[filter.mask == 0] = eps
    inner_fft_log = 20 * np.log(np.abs(inner_fft + tiny_matrix))
    # calculate iFFT
    denoised_img = np.abs(np.fft.ifft2(np.fft.ifftshift(inner_fft)))

    if use_torch_tensor:
        denoised_img = torch.from_numpy(denoised_img)

    return denoised_img


class CrossFilter:
    """Cross filter for molecule images denoising

    Args:
        shape: image size `(H, W)`
        fraction: mask size / image shape
        rotation: angle
        types: `r` for rounded
    """

    def __init__(
            self,
            shape: Union[List[int], Tuple[int, ...]],
            fraction: Union[float, List[float], Tuple[float, float]],
            rotation: float = 0.0,
            types: str = '',
            **kwargs: float
    ):
        self._mask, self._shape, self._edges = self._new_mask(shape, fraction, rotation, types, **kwargs)

    @staticmethod
    def _new_mask(
            shape: Union[List[int], Tuple[int, int]],
            fraction: Union[float, List[float], Tuple[float, float]],
            rotation: float = 0.0,
            types: str = '',
            **kwargs: float
    ) -> Tuple[np.ndarray, Tuple[int, int], List[List[int]]]:
        """Generate a cross-filter mask

        Args:
            shape: image size `(H, W)`
            fraction: mask size / image shape
            rotation: angle
            types: `r` for rounded

        Returns:
            mask: cross-filter mask image
            shape: image size `(H, W)`
            edges: list of edges
        """

        # mask shape
        try:
            y_pxs, x_pxs = shape
        except:
            raise TypeError('Invalid shape {}'.format(np.array(shape).shape))

        # mask fraction
        if isinstance(fraction, float):
            short_fraction = fraction
            long_fraction = fraction
        elif isinstance(fraction, list) or isinstance(fraction, tuple):
            short_fraction = min(fraction)
            long_fraction = max(fraction)
        else:
            raise TypeError('Should receive float, list or tuple, but received {}'.format(fraction.__class__))

        # mask parameters
        aspect_ratio = x_pxs / y_pxs
        x_rot = np.degrees(np.arctan(np.tan(np.radians(rotation)) / aspect_ratio))
        y_rot = np.degrees(np.arctan(np.tan(np.radians(rotation)) * aspect_ratio))
        c_row, c_col = y_pxs // 2, x_pxs // 2
        x_short_edge = int(x_pxs * min(short_fraction, 1) // 2)
        x_long_edge = int(x_pxs * min(long_fraction, 1) // 2)
        y_short_edge = int(y_pxs * min(short_fraction, 1) // 2)
        y_long_edge = int(y_pxs * min(long_fraction, 1) // 2)

        # mask type
        mask_types = {
            'c': False,  # circle mask
            'r': False,  # rounded mask
            'i': False,  # inner rounded mask
        }
        for t in types:
            if t not in mask_types.keys():
                raise ValueError('Unknown mask type {}'.format(t))
            mask_types[t] = True

        # basic mask
        mask, x_mask, y_mask = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        x_mask[c_row - y_short_edge:c_row + y_short_edge + 1, c_col - x_long_edge:c_col + x_long_edge + 1] = 1
        y_mask[c_row - y_long_edge:c_row + y_long_edge + 1, c_col - x_short_edge:c_col + x_short_edge + 1] = 1
        mask = x_mask + y_mask
        mask[mask > 0] = 1

        # circle mask
        if mask_types['c']:
            x_circle = np.linspace(-x_pxs / (2 * (x_long_edge + 0.5)), x_pxs / (2 * (x_long_edge + 0.5)), x_pxs)
            y_circle = np.linspace(-y_pxs / (2 * (y_long_edge + 0.5)), y_pxs / (2 * (y_long_edge + 0.5)), y_pxs)
            X_circle, Y_circle = np.meshgrid(x_circle, y_circle)
            R_circle = np.sqrt(X_circle ** 2 + Y_circle ** 2)
            circle_mask = np.ones(shape)
            circle_mask[R_circle >= 1] = 0.0
            x_mask *= circle_mask
            y_mask *= circle_mask
            mask = x_mask + y_mask
            mask[mask > 0] = 1

        # rounded mask
        if mask_types['r']:
            x_mask_img = Image.new(mode='1', size=(x_pxs, y_pxs), color=0)
            x_rounded = ImageDraw.Draw(x_mask_img, '1')
            x_rounded.rounded_rectangle(
                xy=(c_col - x_long_edge, c_row - y_short_edge, c_col + x_long_edge + 1, c_row + y_short_edge + 1),
                radius=y_short_edge,
                fill=1
            )
            rounded_x_mask = np.array(x_mask_img) * 1
            y_mask_img = Image.new(mode='1', size=(x_pxs, y_pxs), color=0)
            y_rounded = ImageDraw.Draw(y_mask_img, '1')
            y_rounded.rounded_rectangle(
                xy=(c_col - x_short_edge, c_row - y_long_edge, c_col + x_short_edge + 1, c_row + y_long_edge + 1),
                radius=x_short_edge,
                fill=1
            )
            rounded_y_mask = np.array(y_mask_img) * 1
            x_mask *= rounded_x_mask
            y_mask *= rounded_y_mask
            mask = x_mask + y_mask
            mask[mask > 0] = 1

        # inner mask
        if mask_types['i']:
            if aspect_ratio == 1 and x_long_edge >= 3 * y_short_edge:
                inner_radius = np.minimum(x_short_edge // 2, y_short_edge // 2)
                inner_circle = np.zeros((2 * inner_radius + 1, 2 * inner_radius + 1))
                x_inner = y_inner = np.linspace(-inner_radius, inner_radius, 2 * inner_radius + 1)
                X_inner, Y_inner = np.meshgrid(x_inner, y_inner)
                R_inner = np.sqrt(X_inner ** 2 + Y_inner ** 2)
                inner_circle[R_inner >= inner_radius] = 1.0
                inner_circle_corners = inner_circle[X_inner * Y_inner != 0].reshape(
                    (2 * inner_radius, 2 * inner_radius)
                )
                for quad in ([1, 1], [0, 1], [1, 0], [0, 0]):
                    up, bottom = quad[0] * inner_radius, (quad[0] + 1) * inner_radius
                    left, right = quad[1] * inner_radius, (quad[1] + 1) * inner_radius
                    corner = inner_circle_corners[up:bottom, left:right]
                    x_flag = 1 if quad[1] == 1 else -1
                    y_flag = 1 if quad[0] == 1 else -1
                    c_up = c_row - y_flag * (y_short_edge + inner_radius)
                    c_bottom = c_row - y_flag * y_short_edge
                    c_left = c_col - x_flag * (x_short_edge + inner_radius)
                    c_right = c_col - x_flag * x_short_edge
                    mask[
                        min(c_up, c_bottom):max(c_up, c_bottom),
                        min(c_left, c_right):max(c_left, c_right)
                    ] = corner
                    mask[mask > 0] = 1

        # gaussian mask
        if 'gaussian_radius' in kwargs or 'gaussian_sigma' in kwargs:
            gaussian_radius = 1 / kwargs['gaussian_radius'] if 'gaussian_radius' in kwargs else 1
            gaussian_sigma = 2 * kwargs['gaussian_sigma'] ** 2 if 'gaussian_sigma' in kwargs else 2 * (1 / 3) ** 2 
            x_gaussian = np.linspace(
                -gaussian_radius / (2 * (x_long_edge + 0.5) / x_pxs),
                gaussian_radius / (2 * (x_long_edge + 0.5) / x_pxs),
                x_pxs
            )
            y_gaussian = np.linspace(
                -gaussian_radius / (2 * (y_long_edge + 0.5) / y_pxs),
                gaussian_radius / (2 * (y_long_edge + 0.5) / y_pxs),
                y_pxs
            )
            X_gaussian, Y_gaussian = np.meshgrid(x_gaussian, y_gaussian)
            gaussian_a = 1 / np.sqrt(np.pi * gaussian_sigma)
            gaussian_exp = np.exp(-(X_gaussian ** 2 + Y_gaussian ** 2) / gaussian_sigma)
            gaussian_mask = gaussian_a * gaussian_exp
            gaussian_mask[gaussian_mask < 0] = 0
            gaussian_mask /= gaussian_mask.max()
            gaussian_mask = scipy.ndimage.rotate(
                gaussian_mask,
                (x_rot + y_rot / aspect_ratio) / 2,
                reshape=False,
                order=0
            )
            mask *= gaussian_mask

        mask = scipy.ndimage.rotate(mask, rotation, reshape=False, order=0, mode='nearest')
        edges = [[y_short_edge, y_long_edge], [x_short_edge, x_long_edge]]

        return mask, mask.shape, edges

    def denoise(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return filter_denoise(img, self)

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def edges(self) -> List[List[int]]:
        return self._edges

    def __repr__(self) -> str:
        return str(self._mask)

    def __eq__(self, other: 'CrossFilter') -> bool:
        return self._mask == other._mask

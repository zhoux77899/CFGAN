# TODO)) Include in `slitflow` pipeline in the future
import copy
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch
from skimage.registration import phase_cross_correlation
from sklearn.linear_model import LinearRegression

from ..common.env import ENV
from ..common.img import get_background
from ..common.logging import logger
from ..common.utils import get_image_size, normalize, set_image_dimension
from ..math.geometry import Line, Point, get_intersection
from ..math.registration import (
    axial_position_registration,
    differential_cross_correlation,
    normalized_cross_correlation,
)
from ..models.modeling_terminator import Terminator


def lateral_localization(
    reference_image: Union[np.ndarray, torch.Tensor],
    moving_image: Union[np.ndarray, torch.Tensor],
    centers: Union[float, Tuple[float, float], List[float]] = 0.0,
    pitch: Optional[float] = ENV.pitch,
    threshold: Optional[float] = ENV.threshold,
) -> Tuple[float, float]:
    r"""
    Calculate the lateral displacement between two images.

    This function takes a reference image and a moving image, both of which have a lateral displacement,
    and returns the value of this displacement. The displacement is calculated based on a specified threshold.

    :param reference_image: The reference image to which the moving image will be compared.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param moving_image: The image that needs to be compared to the reference image.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param centers: Central coordinates of the reference image.
        If provides a single float value, this value determines both lateral central coordinates.
    :param pitch: A float value that determines the physical size of image pixels.
        Defaults value is `0.08` μm.
    :param threshold: A float value that determines the sensitivity of the displacement calculation.
        Default value is `0.2`.
    :return: A tuple containing the lateral displacement values :math:`(d_x, d_y)` between the two images.
    """

    if threshold and (not isinstance(threshold, (int, float)) or threshold <= 0):
        raise ValueError("Threshold must be a positive number.")
    if pitch and not isinstance(pitch, (int, float)):
        raise ValueError("Pitch must be a number or None.")
    if isinstance(centers, (list, tuple)) and len(centers) != 2:
        raise ValueError("Centers must be a single float or a tuple/list of length 2.")

    if isinstance(reference_image, torch.Tensor):
        reference_image = reference_image.cpu().numpy()
    if isinstance(moving_image, torch.Tensor):
        moving_image = moving_image.cpu().numpy()

    reference_image = copy.deepcopy(set_image_dimension(reference_image, 2))
    moving_image = copy.deepcopy(set_image_dimension(moving_image, 2))

    if threshold:
        reference_intensity_threshold = threshold * reference_image.max()
        moving_intensity_threshold = threshold * moving_image.max()
        reference_mask = reference_image < reference_intensity_threshold
        moving_mask = moving_image < moving_intensity_threshold
        reference_image[reference_mask] = 0.0
        moving_image[moving_mask] = 0.0

    shifts, error, phase_diff = phase_cross_correlation(
        reference_image, moving_image, disambiguate=True, upsample_factor=8
    )
    x_shift, y_shift = [-s for s in shifts[::-1]]

    if isinstance(centers, float):
        x_center = y_center = centers
    else:
        x_center, y_center = centers

    x_pos = x_center + x_shift * pitch if pitch else x_center + x_shift
    y_pos = y_center + y_shift * pitch if pitch else y_center + y_shift

    return x_pos, y_pos


def axial_localization(
    reference_images: Union[np.ndarray, torch.Tensor],
    moving_image: Union[np.ndarray, torch.Tensor],
    lateral_shift: Union[List[float], Tuple[float, float]],
    start: float,
    reference_range: Union[List[float], np.ndarray, torch.Tensor],
    search_length: float = 0.1,
    delay: int = 100,
    centers: Union[float, Tuple[float, float], Tuple[float, float, float], List[float]] = 0.0,
    pitch: Optional[float] = ENV.pitch,
    threshold: float = ENV.threshold,
) -> float:
    r"""
    Calculate the depth position of the moving image within the reference image sequence.

    :param reference_images: Sequence of reference images containing depth information.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param moving_image: Moving image whose depth position needs to be determined.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param lateral_shift: A tuple containing the lateral displacement in pixel values :math:`(d_x, d_y)`
        between the two images.
    :param centers: Central coordinates of the reference image.
        If provides a single float value, this value determines both lateral central coordinates.
    :param start: A float value that determines the starting position for depth search.
    :param reference_range: Range for depth search.
        It can be a List of `float`, `NumPy` array or a `PyTorch` tensor.
    :param search_length: The length of the search range on one side.
        Default value is `0.1` μm.
    :param delay: The maximum time spend on the search.
        Default value is `100` s.
    :param pitch: A float value that determines the physical size of image pixels.
        Defaults value is `0.08` μm.
    :param threshold: A float value that determines the sensitivity of the displacement calculation.
        Default value is `0.2`.
    :return: Depth position :math:`z` of the moving image within the reference image sequence.
    """

    if len(reference_images) != len(reference_range):
        raise ValueError(
            f"Reference images length {len(reference_images)} "
            f"does not match "
            f"reference range length {len(reference_range)}"
        )

    if isinstance(reference_images, torch.Tensor):
        reference_images = reference_images.cpu().numpy()
    if isinstance(moving_image, torch.Tensor):
        moving_image = moving_image.cpu().numpy()

    # convert `lateral_shift` from absolute to relative coordinates
    if isinstance(centers, float):
        lateral_shift = [s - centers for s in lateral_shift]
    else:
        lateral_shift = [s - centers[i] for i, s in enumerate(lateral_shift)]
    # convert `lateral_shift` from μm to pixel if `pitch`
    lateral_shift = [s // pitch if pitch else s for s in lateral_shift]

    delay = max(delay, 0)
    z_registration = None
    search_flag = True  # flag of whether continue search
    start_time = time.perf_counter()  # start search time

    while search_flag and time.perf_counter() - start_time < delay:
        z_registration, normalized_corr_registration, normalized_corr_list = axial_position_registration(
            reference_images, moving_image, lateral_shift, start, reference_range, search_length, threshold=threshold
        )
        position_condition = np.abs(z_registration - start) <= search_length
        correlation_condition = np.abs(normalized_corr_registration - np.max(normalized_corr_list)) < np.abs(
            normalized_corr_registration - np.min(normalized_corr_list)
        )
        if normalized_corr_registration == 0:
            search_flag = False
        elif position_condition and correlation_condition:
            if np.abs(z_registration - start) > search_length / 2:
                start = np.round(z_registration, 2)
            else:
                search_flag = False
        elif normalized_corr_list[0] > normalized_corr_list[-1]:
            start -= search_length
        else:
            start += search_length

    return z_registration if z_registration else start


def relocate_image(image_a: np.ndarray, image_b: np.ndarray, shift_limit: Optional[int] = None) -> np.ndarray:
    """
    Aligns `image_b` to `image_a` using phase cross-correlation with optional shift limiting.

    Uses phase correlation to calculate optimal translation vector between two images,
    applies constrained shift to image_b based on calculated displacement. Designed for
    image registration tasks with optional movement restriction.

    :param image_a: Reference image array (2D/3D) for alignment target
    :param image_b: Source image array (2D/3D) to be aligned with reference
    :param shift_limit: Maximum allowed pixel shift magnitude (None for unconstrained)
    :return: Shifted version of `image_b` aligned to `image_a`
    """

    shifts, error, phase_diff = phase_cross_correlation(image_a, image_b, upsample_factor=1)
    if shift_limit:
        shifts = [min(max(s, -abs(shift_limit)), abs(shift_limit)) for s in shifts]
    return scipy.ndimage.shift(image_b, shifts, order=1, mode="nearest")


def intensity_fitting(
    moving_image: Union[np.ndarray, torch.Tensor],
    fitting_image: Union[np.ndarray, torch.Tensor],
    threshold: float = None,
    use_fitting: bool = False,
) -> float:
    """
    Perform intensity fitting between two images to find the optimal scaling factor.

    :param moving_image: The image that needs to be compared.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param fitting_image: The image that has fitting molecule in the moving image.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param threshold: A float value that determines the sensitivity of the displacement calculation.
    :param use_fitting: A flag that indicates whether to use linear fitting.
    :return: The optimal intensity ratio.
    """

    moving_image = copy.deepcopy(moving_image)
    fitting_image = copy.deepcopy(fitting_image)

    intensity_ratio_range: np.ndarray = np.arange(0.01, 2 * moving_image.max() / fitting_image.max(), 0.01)
    intensity_ratio_corr: List[float] = [
        differential_cross_correlation(moving_image, fitting_image * i, threshold) for i in intensity_ratio_range
    ]

    if not use_fitting:
        return max(0.0, intensity_ratio_range[np.argmin(intensity_ratio_corr)].item())

    split_idx = max(int(0.2 * len(intensity_ratio_corr)), 2)  # at least 2 points
    x_array_l = intensity_ratio_range[:split_idx].reshape((split_idx, 1))
    x_array_r = intensity_ratio_range[-split_idx:].reshape((split_idx, 1))
    y_array_l = np.array(intensity_ratio_corr[:split_idx]).reshape((split_idx, 1))
    y_array_r = np.array(intensity_ratio_corr[-split_idx:]).reshape((split_idx, 1))
    line_reg_l = LinearRegression().fit(x_array_l, y_array_l)
    line_reg_r = LinearRegression().fit(x_array_r, y_array_r)
    coef = (abs(float(line_reg_l.coef_.item())) + abs(float(line_reg_r.coef_.item()))) / 2
    line_l = Line(-coef, Point(intensity_ratio_range.tolist()[0], intensity_ratio_corr[0]))
    line_r = Line(coef, Point(intensity_ratio_range.tolist()[-1], intensity_ratio_corr[-1]))

    return max(0.0, get_intersection(line_l, line_r).x)


class Localizer:
    """
    A class used to perform localization of a moving image with reference images.

    This class provides methods to preprocess and align a moving image with a set of reference images.
    It allows for optional 90-degree rotations and normalization of the input images.

    :param reference_images: Sequence of reference images containing depth information.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param moving_image: Moving image whose depth position needs to be determined.
        It can be a `NumPy` array or a `PyTorch` tensor.
    """

    def __init__(
        self,
        reference_images: Union[np.ndarray, torch.Tensor],
        moving_image: Union[np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        self.reference_rot90 = kwargs.pop("reference_rot90", False)
        self.moving_rot90 = kwargs.pop("moving_rot90", False)
        self.normalize = kwargs.pop("normalize", False)

        self.reference_images = np.rot90(reference_images, axes=(-2, -1)) if self.reference_rot90 else reference_images
        self.moving_image = np.rot90(moving_image, axes=(-2, -1)) if self.moving_rot90 else moving_image
        self.reference_range = np.arange(len(self.reference_images)) * ENV.std_step

        self.ref_len, self.mov_len, self.pad_len = self.get_length()
        self.dxy, self.dz = self.set_distance()
        self.x_shift, self.y_shift, self.z_shift = self.init_shift()

    def get_length(self) -> Tuple[int, int, int]:
        # reference image size must be smaller than moving image
        reference_image_size = get_image_size(self.reference_images, num_dimensions=2)
        moving_image_size = get_image_size(self.moving_image, num_dimensions=2)
        for r, m in zip(reference_image_size, moving_image_size):
            if r >= m:
                raise ValueError(
                    f"Cannot crop from "
                    f"reference image size {reference_image_size} to "
                    f"moving image size {moving_image_size}"
                )

        # reference and moving image must be square
        ref_all_same_size = all(ref_len == reference_image_size[0] for ref_len in reference_image_size)
        mov_all_same_size = all(mov_len == moving_image_size[0] for mov_len in moving_image_size)
        if not ref_all_same_size and not mov_all_same_size:
            raise ValueError(
                f"Reference image size {reference_image_size} or/and "
                f"moving image size {moving_image_size} "
                f"are not all the same size."
            )

        ref_len, ref_len = reference_image_size
        mov_len, mov_len = moving_image_size
        pad_len = mov_len - ref_len

        return ref_len, mov_len, pad_len

    def set_distance(self, lateral_nums: int = 20, dz: int = 10) -> Tuple[int, int]:
        return self.pad_len // lateral_nums + 1, dz

    def init_shift(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_shift = y_shift = np.arange(0, self.pad_len + ENV.tiny, self.dxy, dtype=np.int32)
        z_shift = np.arange(0, len(self.reference_images), self.dz, dtype=np.int32)

        return x_shift, y_shift, z_shift

    def get_3d_correlation_grid(self) -> np.ndarray:
        correlation_grid: List[float] = []
        for z in self.z_shift:
            for y in self.y_shift:
                for x in self.x_shift:
                    ref_img = normalize(copy.deepcopy(self.reference_images[z]))
                    mov_img = normalize(copy.deepcopy(self.moving_image)[y : y + self.ref_len, x : x + self.ref_len])
                    loc_img = relocate_image(mov_img, ref_img, 2 * self.dxy)
                    corr = normalized_cross_correlation(loc_img, mov_img, 0.00)
                    correlation_grid.append(corr)

        correlation_grid: np.ndarray = np.asarray(correlation_grid).reshape(
            len(self.z_shift), len(self.y_shift), len(self.x_shift)
        )
        correlation_grid: np.ndarray = np.nan_to_num(correlation_grid, nan=0)

        return correlation_grid

    def lateral_position_fitting(self, corr_3d_grid: np.ndarray, axial_corr_arg: int) -> Tuple[float, float]:
        corr_2d_grid = corr_3d_grid[axial_corr_arg].squeeze()
        interp = scipy.interpolate.RegularGridInterpolator(
            (self.x_shift, self.y_shift), corr_2d_grid, method="linear", bounds_error=False, fill_value=None
        )

        x_shift_fit = y_shift_fit = np.round(np.arange(0, self.pad_len + ENV.tiny, 0.1), 1)
        x_shift_fit_grid, y_shift_fit_grid = np.meshgrid(x_shift_fit, y_shift_fit, indexing="ij")
        corr_2d_grid_fit = interp((x_shift_fit_grid, y_shift_fit_grid))
        fit_arg = np.argmax(corr_2d_grid_fit)

        x_pos = float(x_shift_fit[fit_arg % len(x_shift_fit)])
        y_pos = float(y_shift_fit[fit_arg // len(y_shift_fit)])

        return x_pos, y_pos

    def all_position_localization(self, corr_3d_grid: np.ndarray, axial_corr_arg: int) -> Tuple[float, float, float]:
        reference_image = self.reference_images[self.z_shift[axial_corr_arg].item()]

        x_pos, y_pos = self.lateral_position_fitting(corr_3d_grid, axial_corr_arg)
        moving_image = self.moving_image[int(y_pos) : int(y_pos) + self.ref_len, int(x_pos) : int(x_pos) + self.ref_len]
        x_calib, y_calib = lateral_localization(reference_image, moving_image, pitch=None, threshold=None)
        calib_limit = self.dxy * 2
        x_calib, y_calib = [min(max(c, -abs(calib_limit)), abs(calib_limit)) for c in [x_calib, y_calib]]
        moving_image = scipy.ndimage.shift(moving_image, [-y_calib, -x_calib], order=1, mode="nearest")
        z_pos = axial_localization(
            self.reference_images,
            moving_image,
            lateral_shift=[0, 0],
            start=self.reference_range[self.z_shift[axial_corr_arg]].item(),
            reference_range=self.reference_range,
            pitch=None,
        )

        return x_pos + x_calib, y_pos + y_calib, z_pos

    def needle_in_a_haystack_localization(self) -> Tuple[float, float, float]:
        corr_3d_grid = self.get_3d_correlation_grid()
        axial_max_corr = np.max(corr_3d_grid, axis=(1, 2))

        image_diff = []

        for axial_corr_arg in np.argsort(axial_max_corr)[::-1]:
            x_pos, y_pos = self.lateral_position_fitting(corr_3d_grid, axial_corr_arg)
            z_pos = float(self.reference_range[self.z_shift[axial_corr_arg]])
            x_start, x_end = int(x_pos), int(x_pos) + self.ref_len
            y_start, y_end = int(y_pos), int(y_pos) + self.ref_len

            # image size of [ref_len, ref_len]
            reference_image = self.reference_images[
                np.argwhere(np.round(self.reference_range, 2) == np.round(z_pos, 2)).squeeze()
            ].astype(self.moving_image.dtype)
            moving_image = self.moving_image[y_start:y_end, x_start:x_end]

            if self.normalize:
                reference_image = normalize(reference_image)
                moving_image = normalize(moving_image)

            intensity = intensity_fitting(moving_image, reference_image)
            residual_image = moving_image - reference_image * intensity
            image_diff.append(residual_image.min())

        image_diff = np.array(image_diff)
        diff_threshold = np.percentile(image_diff, 0.0)
        best_arg = np.argwhere(image_diff >= diff_threshold)[0].squeeze()
        axial_corr_arg = np.argwhere(axial_max_corr == np.sort(axial_max_corr)[-1 - best_arg]).squeeze().item()

        return self.all_position_localization(corr_3d_grid, axial_corr_arg)

    def right_under_your_nose_localization(self) -> Tuple[float, float, float]:
        corr_3d_grid = self.get_3d_correlation_grid()
        axial_corr_arg = int(np.argmax(corr_3d_grid)) // (len(self.x_shift) * len(self.y_shift))

        return self.all_position_localization(corr_3d_grid, axial_corr_arg)


class LocalizationPipeline:
    """
    A class to manage the localization process of aligning a moving image with reference images.

    This class encapsulates the configuration and execution of a localization pipeline,
    allowing for coarse and fine iterations of alignment.
    Various parameters can be customized to adjust the behavior of the pipeline.

    :param reference_images: Sequence of reference images containing depth information.
        It can be a `NumPy` array or a `PyTorch` tensor.
    :param moving_image: Moving image whose depth position needs to be determined.
        It can be a `NumPy` array or a `PyTorch` tensor.
    """

    def __init__(
        self,
        reference_images: Union[np.ndarray, torch.Tensor],
        moving_image: Union[np.ndarray, torch.Tensor],
        terminator: str,
        **kwargs: Any,
    ) -> None:
        self.reference_images = reference_images
        self.moving_image = moving_image

        try:
            self.device = torch.device(kwargs.pop("device", "cpu"))
        except RuntimeError as e:
            logger.warning(f"set device error: {e}, default set to cpu")
            self.device = torch.device("cpu")

        self.coarse_max_iters = kwargs.pop("coarse_max_iters", 10)
        self.fine_max_iters = kwargs.pop("fine_max_iters", 5)
        self.remove_fitting_area = kwargs.pop("remove_fitting_area", False)
        self.localizer_args = kwargs

        self.terminator = self.load_model(terminator)

        self.coarse_positions: List[List[float]] = []
        self.fine_positions: List[List[float]] = []

    def __call__(self) -> Tuple[List[List[float]], List[List[float]]]:
        self.coarse_localization_round()
        for i in range(self.fine_max_iters):
            self.fine_localization_round()
        return self.coarse_positions, self.fine_positions

    def load_model(self, weight_path: str) -> Terminator:
        torch.serialization.add_safe_globals([Terminator])

        terminator = Terminator()
        weights = torch.load(weight_path, weights_only=True)
        terminator.load_state_dict(weights)
        terminator.to(self.device).eval()

        return terminator

    def coarse_localization_round(self) -> None:
        self.coarse_positions.clear()
        reference_images = copy.deepcopy(self.reference_images)
        moving_image = copy.deepcopy(self.moving_image)

        iters: int = 0
        has_emitters: bool = (
            self.terminator(set_image_dimension(torch.from_numpy(moving_image).to(torch.float32), 4)) > 0.5
        )
        reach_max_iters: bool = iters >= self.coarse_max_iters > 0
        while has_emitters and not reach_max_iters:
            iters += 1
            localizer = Localizer(reference_images, moving_image, **self.localizer_args)
            x_pos, y_pos, z_pos = localizer.needle_in_a_haystack_localization()
            moving_image = self._fit_residual_image(reference_images, moving_image, [x_pos, y_pos, z_pos], localizer)
            self.coarse_positions.append([x_pos, y_pos, z_pos])
            has_emitters = (
                self.terminator(set_image_dimension(torch.from_numpy(moving_image).to(torch.float32), 4)) > 0.5
            )
            reach_max_iters = iters >= self.coarse_max_iters > 0

    def fine_localization_round(self) -> None:
        if not self.coarse_positions:
            logger.warning(f"call `{self.__class__.__name__}.coarse_localization_round()` first")
            self.coarse_localization_round()
            logger.warning(f"continue calling `{self.__class__.__name__}`.fine_localization_round")

        positions = self.fine_positions.copy() if self.fine_positions else self.coarse_positions
        self.fine_positions.clear()
        reference_images = copy.deepcopy(self.reference_images)

        for i in range(len(positions)):
            moving_image = copy.deepcopy(self.moving_image)
            localizer = Localizer(reference_images, moving_image, **self.localizer_args)
            for j in range(len(positions)):
                if i != j:
                    moving_image = self._fit_residual_image(reference_images, moving_image, positions[j], localizer)
            localizer = Localizer(reference_images, moving_image, **self.localizer_args)
            x_pos, y_pos, z_pos = localizer.right_under_your_nose_localization()
            self.fine_positions.append([x_pos, y_pos, z_pos])

    @staticmethod
    def check_position(
        positions: Union[List[float], Tuple[float, float, float]],
        limits: Union[List[float], Tuple[float, float, float]],
    ) -> List[float]:
        return [min(max(position, 0), limit) for position, limit in zip(positions, limits)]

    def _fit_residual_image(
        self,
        reference_images: Union[np.ndarray, torch.Tensor],
        moving_image: Union[np.ndarray, torch.Tensor],
        positions: Union[List[float], Tuple[float, float, float]],
        localizer: Localizer,
        remove_ratio: float = 0.0,
    ) -> Union[np.ndarray, torch.Tensor]:
        ref_rot90 = self.localizer_args.get("reference_rot90", False)
        mov_rot90 = self.localizer_args.get("moving_rot90", False)

        x_pos, y_pos, z_pos = self.check_position(
            positions, [localizer.pad_len, localizer.pad_len, ENV.bias + ENV.limit]
        )
        x_start, x_end = int(x_pos), int(x_pos) + localizer.ref_len
        y_start, y_end = int(y_pos), int(y_pos) + localizer.ref_len

        moving_image = np.rot90(moving_image, axes=(0, 1)) if mov_rot90 else moving_image

        # image size of [ref_len, ref_len]
        clipped_moving_image = moving_image[y_start:y_end, x_start:x_end]
        reference_image = reference_images[
            np.argwhere(np.round(localizer.reference_range, 2) == np.round(z_pos, 2)).squeeze()
        ].astype(moving_image.dtype)
        reference_image = np.rot90(reference_image, axes=(0, 1)) if ref_rot90 else reference_image
        ref_shifts: np.ndarray | None = None
        for normalization_param in ["phase", None]:
            ref_shifts, _, _ = phase_cross_correlation(
                clipped_moving_image,
                reference_image,
                disambiguate=True,
                upsample_factor=1,
                normalization=normalization_param,
            )
            shifted_reference_image = scipy.ndimage.shift(reference_image, ref_shifts, order=1, mode="nearest")
            intensity = intensity_fitting(clipped_moving_image, shifted_reference_image, use_fitting=True)
            if np.linalg.norm(ref_shifts) <= np.sqrt(8) * localizer.dxy and intensity > 0:
                break
        else:
            logger.info(f"shifts of reference image {ref_shifts.tolist()} exceeds limit, changed to default [0, 0]")
            shifted_reference_image = scipy.ndimage.shift(reference_image, [0, 0], order=1, mode="nearest")
            intensity = intensity_fitting(clipped_moving_image, shifted_reference_image, use_fitting=True)
        intensity = max(0.0, intensity)

        # image size of [clip_size, clip_size]
        fitting_image = np.full_like(moving_image, get_background(reference_image), dtype=moving_image.dtype)
        fitting_image[y_start:y_end, x_start:x_end] = shifted_reference_image
        if self.localizer_args.get("normalize", False):
            fitting_image = normalize(fitting_image)
        residual_image = moving_image - fitting_image * intensity

        if self.remove_fitting_area:
            fitting_area = residual_image[y_start:y_end, x_start:x_end]
            residual_condition = fitting_area < residual_image.max() * remove_ratio
            fitting_area[residual_condition] = get_background(reference_image)
            residual_image[y_start:y_end, x_start:x_end] = fitting_area
        else:
            residual_image[residual_image < residual_image.max() * remove_ratio] = get_background(reference_image)

        return residual_image

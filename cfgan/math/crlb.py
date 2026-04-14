from abc import ABC, abstractmethod

import numpy as np
import torch

from cfgan.common.utils import get_image_dimension


class NormalizationStrategy(ABC):
    """
    Abstract base class for PSF image normalization strategies.

    Subclasses must implement the ``_normalize`` method to define how
    PSF images are normalized. Common validation (dimension check,
    non-negativity, non-zero intensity) is handled by the base class
    via the Template Method pattern.

    Usage:
        >>> strategy = GlobalNormalization()
        >>> normalized = strategy(psf_images)
    """

    def __call__(
        self, imgs: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        self._validate(imgs)
        return self._normalize(imgs)

    @staticmethod
    def _validate(imgs: np.ndarray | torch.Tensor) -> None:
        """Validate PSF images before normalization."""
        assert get_image_dimension(imgs) == 3, "`imgs` must be a $z$-stack of images"

        if isinstance(imgs, np.ndarray):
            assert (imgs >= 0).all(), "PSF images must be non-negative"
            slice_sums = imgs.sum(axis=(1, 2), keepdims=True)
            assert (slice_sums > 0).all(), "Each PSF slice must have non-zero total intensity"
        elif isinstance(imgs, torch.Tensor):
            assert (imgs >= 0).all(), "PSF images must be non-negative"
            slice_sums = imgs.sum(dim=(1, 2), keepdim=True)
            assert (slice_sums > 0).all(), "Each PSF slice must have non-zero total intensity"
        else:
            raise ValueError(f"Unknown imgs type: {type(imgs)}.")

    @abstractmethod
    def _normalize(
        self, imgs: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Apply normalization to validated PSF images."""
        ...


class GlobalNormalization(NormalizationStrategy):
    """
    Normalize PSF images to a global probability distribution.

    The entire 3D stack is normalized so that the global sum of all
    elements equals 1.

    Note:
        For CRLB calculation in single-molecule localization with a 2D
        detector, ``PerSliceNormalization`` is the physically correct choice.
    """

    def _normalize(
        self, imgs: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        total = imgs.sum()
        return imgs / total


class PerSliceNormalization(NormalizationStrategy):
    """
    Normalize PSF images to per-slice probability distributions.

    Each 2D slice (z-plane) is independently normalized so that the sum
    of its elements equals 1. This is the physically correct normalization
    for CRLB calculation in single-molecule localization with a 2D detector.
    """

    def _normalize(
        self, imgs: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(imgs, np.ndarray):
            return imgs / imgs.sum(axis=(1, 2), keepdims=True)
        return imgs / imgs.sum(dim=(1, 2), keepdim=True)


class NoneNormalization(NormalizationStrategy):
    """
    Pass through PSF images without normalization.

    Use this when the input PSF data has already been properly normalized
    (e.g., per-slice probability distribution from PSF generation code)
    and re-normalization would introduce errors.
    """

    def _normalize(
        self, imgs: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        return imgs


_NORMALIZATION_REGISTRY: dict[str, type[NormalizationStrategy]] = {
    "global": GlobalNormalization,
    "per_slice": PerSliceNormalization,
    "none": NoneNormalization,
}


def get_normalization(name: str) -> NormalizationStrategy:
    """
    Factory function that returns a normalization strategy instance by name.

    This function looks up the normalization strategy in the internal
    registry and returns a new instance. To add a new normalization method,
    create a subclass of ``NormalizationStrategy`` and register it in
    ``_NORMALIZATION_REGISTRY``.

    Args:
        name: Normalization method name. Supported values:

            - ``"global"``: Normalize the entire 3D stack so that the
              global sum equals 1.
            - ``"per_slice"``: Normalize each 2D z-slice independently
              so that each slice sums to 1. Physically correct for CRLB
              calculation with a 2D detector.
            - ``"none"``: No normalization applied. Use when PSF data
              is already properly normalized.

    Returns:
        A ``NormalizationStrategy`` instance corresponding to the given name.

    Raises:
        ValueError: If the normalization name is not registered.

    Example:
        >>> strategy = get_normalization("per_slice")
        >>> normalized = strategy(psf_images)
    """
    if name not in _NORMALIZATION_REGISTRY:
        supported = ", ".join(f"'{k}'" for k in _NORMALIZATION_REGISTRY)
        raise ValueError(f"Unknown normalization '{name}'. Supported: {supported}")
    return _NORMALIZATION_REGISTRY[name]()


class CRLB:
    """
    Calculate the Cramer-Rao Lower Bound (CRLB) for 3D localization.

    The CRLB provides a lower bound on the variance of any unbiased estimator.
    For single molecule localization, it gives the theoretical limit on localization
    precision.

    Args:
        psf_imgs: 3D PSF array of shape (nz, ny, nx)
        delta: Tuple of (dx, dz) pixel sizes
        eps: Small value for numerical stability (default: 1e-6)
        normalization: Normalization strategy for PSF images. Options:

            - ``"global"``: Normalize the entire 3D stack so that the global sum
              equals 1.
            - ``"per_slice"``: Normalize each 2D z-slice independently so that each
              slice sums to 1. This is the physically correct choice for CRLB
              calculation with a 2D detector.
            - ``"none"``: No normalization applied. Use when PSF data is already
              properly normalized.

            Default: ``"none"``.
    """

    def __init__(
        self,
        psf_imgs: np.ndarray | torch.Tensor,
        delta: tuple[float, float],
        eps: float = 1e-6,
        normalization: str = "none",
    ) -> None:
        assert get_image_dimension(psf_imgs) == 3, "`psf_imgs` must be a $z$-stack of images"
        assert len(delta) == 2, "`delta` must be a 2-tuple of (dx, dz)"

        normalizer = get_normalization(normalization)
        self.psf_imgs = normalizer(psf_imgs)

        self.dx, self.dz = delta
        self.eps = eps

    def __call__(
        self,
        total_photons: int,
        background_photons: int,
        mode: str,
    ) -> np.ndarray | torch.Tensor:
        return self.forward(total_photons, background_photons, mode)

    def pad(self, grad: np.ndarray | torch.Tensor, padding: tuple[int, ...]) -> np.ndarray | torch.Tensor:
        assert len(padding) == 6, "`padding` must be a 6-tuple"

        if isinstance(grad, np.ndarray):
            pad_width = tuple(zip(padding[::2], padding[1::2]))
            pad_width = pad_width[::-1]
            return np.pad(grad, pad_width, mode="edge")
        if isinstance(grad, torch.Tensor):
            padded_grad = grad.unsqueeze(0).unsqueeze(0)  # [D, H, W] -> [1, 1, D, H, W]
            torch_pad_func = torch.nn.ReplicationPad3d(padding)
            padded_grad = torch_pad_func(padded_grad)
            return padded_grad.squeeze(0).squeeze(0)  # [1, 1, D, H, W] -> [D, H, W]

        raise NotImplementedError(f"CRLB.pad() does not support {type(grad)}")

    def maximum(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(x, np.ndarray):
            return np.maximum(x, self.eps)
        if isinstance(x, torch.Tensor):
            return torch.maximum(x, torch.full_like(x, self.eps))
        raise NotImplementedError(f"CRLB.maximum() does not support {type(x)}")

    def sqrt(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(x, np.ndarray):
            return np.sqrt(x)
        if isinstance(x, torch.Tensor):
            return torch.sqrt(x)
        raise NotImplementedError(f"CRLB.sqrt() does not support {type(x)}")

    def grad(
        self,
        direction: str,
        psf_imgs: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray | torch.Tensor:
        """
        Calculate the gradient of the CRLB with respect to the given direction.
        """
        if psf_imgs is None:
            psf_imgs = self.psf_imgs

        if direction == "x":
            grad = (psf_imgs[:, :, 2:] - psf_imgs[:, :, :-2]) / (2 * self.dx)
            return self.pad(grad, (1, 1, 0, 0, 0, 0))
        if direction == "y":
            grad = (psf_imgs[:, 2:, :] - psf_imgs[:, :-2, :]) / (2 * self.dx)
            return self.pad(grad, (0, 0, 1, 1, 0, 0))
        if direction == "z":
            grad = (psf_imgs[2:, :, :] - psf_imgs[:-2, :, :]) / (2 * self.dz)
            return self.pad(grad, (0, 0, 0, 0, 1, 1))

        raise NotImplementedError(f"CRLB.grad() does not support direction {direction}")

    def forward(
        self,
        total_photons: int,
        background_photons: int,
        mode: str,
    ) -> np.ndarray | torch.Tensor:
        psf_imgs = self.psf_imgs * total_photons
        denominator = self.maximum(psf_imgs + background_photons)

        if mode == "lateral":
            fx = self.maximum((self.grad("x", psf_imgs) ** 2 / denominator).sum((1, 2)))
            fy = self.maximum((self.grad("y", psf_imgs) ** 2 / denominator).sum((1, 2)))
            return self.sqrt(1 / fx + 1 / fy)
        if mode == "axial":
            fz = self.maximum((self.grad("z", psf_imgs) ** 2 / denominator).sum((1, 2)))
            return self.sqrt(1 / fz)
        if mode == "radial":
            fx = self.maximum((self.grad("x", psf_imgs) ** 2 / denominator).sum((1, 2)))
            fy = self.maximum((self.grad("y", psf_imgs) ** 2 / denominator).sum((1, 2)))
            fz = self.maximum((self.grad("z", psf_imgs) ** 2 / denominator).sum((1, 2)))
            return self.sqrt(1 / fx + 1 / fy + 1 / fz)

        raise NotImplementedError(f"CRLB.forward() does not support mode {mode}")
